"""
src/possession_engine_v2.py - CHALLENGE.md control physics for Stage 2 (v2).

Per-frame control evidence from tactical ATD only:
    d_home[f], d_away[f]   distance (m) from the TRUSTED ball (src/ball_speed.py) to the
                           nearest home / away player
    speed[f]               ball speed, m/s (src/ball_speed.py; NaN when missing/untrusted)

Regimes (CHALLENGE.md "controlled possession" + "loose ball" rules):
    CTRL_STRONG  d_nearest < 1.2 m AND speed < 5 m/s AND the other team is not equally
                 close (margin >= 0.5 m)              -> strong emission for that team
    CTRL_WEAK    a player within 2.5 m of a slow ball, not strong (closing in, shielding)
                                                      -> weak emission for that team
    DEFLECT      d_nearest < 1.5 m but speed > 7 m/s  -> flat: touch / deflection / rebound
                                                         / won header is NOT a change
    LOOSE        both teams farther than 2.5 m        -> flat: last team keeps it
    CONTESTED    both teams within CONTEST_MARGIN of each other at the ball -> flat
    MISSING      ball unseen / untrusted              -> flat (+ gap interception below)

Flat = 0 on both H and A, so the HMM's P(stay) carries the last possession through.

Missing-ball trajectory interception (intercept_gaps):
    For each alive gap, join the last visible ball (x1, y1) to the reappearance (x2, y2).
    If a player of the team that did NOT have the ball at the gap start comes within
    1.2 m of that segment, and the ball reappears at a controlled speed (< 4 m/s), the
    transition is placed at that player's closest-approach frame. Otherwise the last
    possession is preserved across the whole gap.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FPS = 25

# ---- control regimes (metres, m/s) -----------------------------------------
CTRL_D = 1.2             # ball within a stride of the foot: on the player's foot
CTRL_SPEED = 5.0         # a dribble / settled first touch moves the ball < ~5 m/s
DEFLECT_D = 1.5          # close enough to touch it ...
DEFLECT_SPEED = 7.0      # ... but this fast = it bounced off / was flicked on, not controlled
LOOSE_D = 2.5            # nobody within ~1 step + reach: loose ball
CONTEST_MARGIN_M = 0.5   # both teams this close to each other at the ball = 50/50 duel

# ---- emission strengths (nats per frame; the HMM unit is log(P_stay/P_switch)) ----
# Default switch penalty is log(0.985/0.015) = 4.18 nats. A controlled reception (first
# touch + settle) takes ~0.4 s = 10 frames, so 10 frames of clean control should just pay
# for a switch: 4.18 / 10 = 0.42 per frame.
STRONG_LLR = 0.42
WEAK_LLR = 0.25 * STRONG_LLR

# ---- missing-ball interception ---------------------------------------------
INTERCEPT_R = 1.2        # opponent this close to the straight ball path = could cut it out
INTERCEPT_SPEED = 4.0    # m/s: ball reappears this slow = it was brought under control
GAP_MIN_FRAMES = 3       # shorter dropouts are tracker flicker
REAPPEAR_FRAMES = 3      # frames used to measure the reappearance speed
GAP_SWITCH_LLR = STRONG_LLR   # per frame, for the interceptor after the cut-out frame
GAP_HOLD_LLR = 0.05      # per frame, for the last team: a tie-breaker so a later switch
                         # lands at the reappearance, not at an arbitrary gap frame

REGIMES = ("MISSING", "LOOSE", "CONTESTED", "DEFLECT", "CTRL_WEAK", "CTRL_STRONG")
R_MISSING, R_LOOSE, R_CONTESTED, R_DEFLECT, R_WEAK, R_STRONG = range(6)


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def distances(P, slot_team, bx, by):
    """d_home, d_away (n,) from the ball to the nearest player of each team; NaN where
    the ball is missing. P is (n, slots, 2) metres, slot_team 0 = home, 1 = away."""
    home = slot_team == 0
    with np.errstate(invalid="ignore"):
        d = np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    d_home = d[:, home].min(axis=1)
    d_away = d[:, ~home].min(axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan
    return d_home, d_away


def control_regimes(d_home, d_away, speed):
    """Returns (regime code per frame, controlling team per frame: +1 home, -1 away, 0)."""
    n = len(d_home)
    dh = np.where(np.isnan(d_home), np.inf, d_home)
    da = np.where(np.isnan(d_away), np.inf, d_away)
    d_near = np.minimum(dh, da)
    d_far = np.maximum(dh, da)
    team = np.where(dh < da, 1, -1).astype(np.int8)
    seen = ~np.isnan(speed) & np.isfinite(d_near)
    s = np.nan_to_num(speed, nan=0.0)

    reg = np.full(n, R_MISSING, dtype=np.int8)
    reg[seen] = R_LOOSE
    near = seen & (d_near <= LOOSE_D)
    reg[near & (s <= DEFLECT_SPEED)] = R_WEAK
    reg[seen & (d_near < CTRL_D) & (s < CTRL_SPEED)] = R_STRONG
    reg[seen & (d_near < DEFLECT_D) & (s > DEFLECT_SPEED)] = R_DEFLECT
    with np.errstate(invalid="ignore"):          # inf - inf on frames with no players
        contested = near & (d_far - d_near < CONTEST_MARGIN_M) & (reg != R_DEFLECT)
    reg[contested] = R_CONTESTED
    team[(reg != R_STRONG) & (reg != R_WEAK)] = 0
    return reg, team


def control_llr(reg, team):
    """Signed per-frame log-likelihood ratio log P(obs|H) - log P(obs|A)."""
    llr = np.zeros(len(reg))
    llr[reg == R_STRONG] = STRONG_LLR
    llr[reg == R_WEAK] = WEAK_LLR
    return llr * team


def _seg_dist(px, py, x1, y1, x2, y2):
    """Distance from points (px, py) to the segment (x1, y1)-(x2, y2)."""
    vx, vy = x2 - x1, y2 - y1
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return np.hypot(px - x1, py - y1)
    t = np.clip(((px - x1) * vx + (py - y1) * vy) / L2, 0.0, 1.0)
    return np.hypot(px - (x1 + t * vx), py - (y1 + t * vy))


def intercept_gaps(P, slot_team, bx, by, speed, alive, team_before):
    """Missing-ball trajectory interception.

    team_before[f]: +1 home / -1 away possession from a first-pass decode.
    Returns (llr (n,) signed H-vs-A evidence over the gaps, list of gap records).
    """
    n = len(bx)
    llr = np.zeros(n)
    records = []
    missing = np.isnan(bx)
    home = slot_team == 0
    for g0, g1 in _runs(missing & alive):
        if g1 - g0 < GAP_MIN_FRAMES or g0 == 0 or g1 >= n:
            continue
        if missing[g0 - 1] or missing[g1] or not alive[g0 - 1]:
            continue                       # gap must be bracketed by visible, alive ball
        prev = int(team_before[g0 - 1])
        if prev == 0:
            continue
        x1, y1, x2, y2 = bx[g0 - 1], by[g0 - 1], bx[g1], by[g1]
        opp_slots = ~home if prev == 1 else home
        Pg = P[g0:g1][:, opp_slots]                        # (gap, k, 2)
        d = _seg_dist(Pg[..., 0], Pg[..., 1], x1, y1, x2, y2)
        d = np.where(np.isnan(d), np.inf, d)
        dmin_f = d.min(axis=1)
        k = int(np.argmin(dmin_f))
        reap = speed[g1:min(n, g1 + REAPPEAR_FRAMES)]
        reap = reap[~np.isnan(reap)]
        v_reap = float(np.median(reap)) if len(reap) else np.inf
        hit = dmin_f[k] < INTERCEPT_R and v_reap < INTERCEPT_SPEED
        if hit:
            tc = g0 + k
            llr[g0:tc] += prev * GAP_HOLD_LLR
            llr[tc:g1] += -prev * GAP_SWITCH_LLR
        else:
            llr[g0:g1] += prev * GAP_HOLD_LLR
        records.append(dict(g0=int(g0), g1=int(g1), prev=prev, d_min=float(dmin_f[k]),
                            v_reappear=v_reap, intercepted=bool(hit),
                            cut_frame=int(g0 + k) if hit else None))
    return llr, records
