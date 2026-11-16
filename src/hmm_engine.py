"""
Module 3 - HMM DECODER (final stage: globally coherent possession spells)

Takes Module 1's dead mask (hard constraint) and Module 2's physics signals
(nearest home / away distance to the ball, ball speed, restart priors) and decodes
the whole 199,319-frame sequence at once with a 3-state log-space Viterbi:

    states   0 = H (home in possession), 1 = A (away), 2 = D (dead / ungraded)

Why an HMM on top of the dwell logic: Module 2 decides causally, frame by frame, so
a switch is only accepted DWELL frames *after* the challenger took the ball and a
run can start unlabelled until the first controlled touch. Viterbi sees the whole
sequence: it places each switch at the frame where the challenger's control
actually began, fills alive frames that have no direct evidence from the
neighbouring spell, and rejects any excursion that does not pay for two switches,
which is exactly the flicker that destroys spellF1.

Hard constraints
  * Module 1 dead frames  -> emission(H) = emission(A) = -inf, emission(D) = 0
  * ungraded frames       -> decoded D, written as X by the orchestrator
  * graded alive frames   -> emission(D) = -inf, so they always decode to H or A

TACTICAL SOURCE ONLY: every input reaches this module through Module 1 / Module 2,
which load via games.require()/asset_paths() with the default source="tactical".
This file opens no files itself.

numpy + standard library only (repo contract: numpy + pandas).
"""

import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def import_engines():
    """Return (deadball_engine, possession_engine) modules.

    The engines live in src/module_1.py and src/module_2.py, but module_2 imports its
    sibling as `deadball_engine`. Register the aliases in sys.modules so both names
    resolve without renaming or editing the original files.
    """
    for alias, filename in (("deadball_engine", "module_1"),
                            ("possession_engine", "module_2")):
        if alias not in sys.modules:
            try:
                sys.modules[alias] = importlib.import_module(alias)
            except ImportError:
                sys.modules[alias] = importlib.import_module(filename)
    return sys.modules["deadball_engine"], sys.modules["possession_engine"]


_db, _pe = import_engines()

# ----------------------------------------------------------------------------
# Parameters (physical meaning; shared constants come from Modules 1/2 so the
# three stages cannot drift apart)
# ----------------------------------------------------------------------------
FPS = _db.FPS                       # 25 frames / s
CONTROL_R = _pe.CONTROL_R           # 4.5 m: a player this close can play the ball
FLIGHT_SPEED = _pe.FLIGHT_SPEED     # 9.0 m/s: faster than this = ball travelling

HOME, AWAY, DEAD, UNGRADED = "H", "A", "D", "X"
S_H, S_A, S_D = 0, 1, 2
STATE_LABELS = np.array([HOME, AWAY, DEAD])

# --- transitions -------------------------------------------------------------
# Expected length of a possession spell. A geometric dwell with this mean gives
# P(switch) = 1 / (MIN_POSSESSION_S * FPS) per frame: 3 s -> 1/75 -> P(stay)=0.9867.
MIN_POSSESSION_S = 3.0
P_SWITCH = 1.0 / (MIN_POSSESSION_S * FPS)
P_STAY = 1.0 - P_SWITCH
# Cost (nats) of one H<->A switch relative to staying. 3 s -> log(74) = 4.30 nats.
SWITCH_PENALTY = float(np.log(P_STAY / P_SWITCH))
# Alive <-> D transitions carry no cost: D is decided entirely by Module 1's mask
# through the emissions, so these entries are structural. The first alive frame
# after a D run is a clean hand-off and may start in H or A freely.
LOG_ALIVE_TO_DEAD = 0.0
LOG_DEAD_TO_ALIVE = 0.0

# --- emissions -----------------------------------------------------------------
# How long a challenger must show clear control before a switch is plausible.
# Same value as Module 2's dwell hysteresis (16 frames = 0.64 s).
CHALLENGE_DWELL_S = _pe.DWELL / FPS
# Per-frame log-likelihood ratio of one frame of *clear* control. Chosen so that
# CHALLENGE_DWELL_S of clear challenger control exactly pays one SWITCH_PENALTY:
# 4.30 nats / 16 frames = 0.27 nats/frame. A temporary excursion has to pay two
# switches (out and back), so a deflection needs ~1.3 s of clear control to
# register - shorter touches are absorbed into the surrounding spell.
CONTROL_LLR = SWITCH_PENALTY / (CHALLENGE_DWELL_S * FPS)
# Within this distance of the ball a player has it at his feet (full evidence);
# evidence fades linearly to zero at CONTROL_R.
TOUCH_R = 1.5
# Contest margin: the closer team's evidence scales with how much closer it is.
# Equal distances (a 50/50 duel) are neutral, and possession stays with the
# current team, per the rules. 2 m ahead of the nearest opponent = full evidence.
CONTEST_MARGIN_M = 2.0
# A ball moving faster than FLIGHT_SPEED next to a player is a pass going past or
# a deflection, not control: such frames count at this fraction.
FAST_TOUCH_WEIGHT = 0.35
# Ball in flight (fast, nobody within CONTROL_R), ball loose, or ball unseen:
# flat emissions. The hold toward the current state comes from P_STAY.

# --- restart prior (kept from Module 2) -----------------------------------------
# A Smart Tagging set piece whose window contains the start of an alive run tells
# us who restarts. The prior is worth one switch: the opponent must show a full
# dwell of clear control to override the tag.
RESTART_PRIOR_LLR = SWITCH_PENALTY
# Tag-to-restart matching tolerance (s) either side of the tag window (Module 2
# uses 3 s after the tag start).
RESTART_MATCH_S = 3.0


# ----------------------------------------------------------------------------
# Evidence (Module 2's physics signals as arrays)
# ----------------------------------------------------------------------------
def possession_evidence(ball_x, ball_y, home_x, home_y, away_x, away_y):
    """Per-frame d_home, d_away: distance (m) from the ball to the nearest home /
    away tracked player. NaN when the ball is unseen or the team has no tracks.
    Same geometry as Module 2's _nearest(), kept per team instead of only the min."""
    n = len(ball_x)
    d_home = np.full(n, np.nan)
    d_away = np.full(n, np.nan)
    for f in range(n):
        bx = ball_x[f]
        if bx != bx:                    # NaN: ball unseen
            continue
        by = ball_y[f]
        xs = home_x[f]
        if xs is not None and len(xs):
            d_home[f] = np.sqrt(((xs - bx) ** 2 + (home_y[f] - by) ** 2).min())
        xs = away_x[f]
        if xs is not None and len(xs):
            d_away[f] = np.sqrt(((xs - bx) ** 2 + (away_y[f] - by) ** 2).min())
    return d_home, d_away


def evidence_from_tracking(trk, dynamic_team=None):
    """Vectorised possession_evidence() on Module 1's slot matrix (same metres,
    same anisotropic scaling): returns d_home, d_away.
    Supports dynamic_team (n, 46) for pass-flow reattributed slots."""
    P = trk["P"]
    dx = P[..., 0] - trk["ball_x"][:, None]
    dy = P[..., 1] - trk["ball_y"][:, None]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)
    if dynamic_team is not None:
        home = (dynamic_team == 0)
        d_home = np.where(home, d, np.inf).min(axis=1)
        d_away = np.where(~home, d, np.inf).min(axis=1)
    else:
        home = trk["slot_team"] == 0
        d_home = d[:, home].min(axis=1)
        d_away = d[:, ~home].min(axis=1)
    d_home[~np.isfinite(d_home)] = np.nan
    d_away[~np.isfinite(d_away)] = np.nan
    return d_home, d_away


def control_score(d_home, d_away, ball_s):
    """Signed control evidence in [-1, 1]: +1 = clear home control, -1 = clear away,
    0 = no information (ball unseen, in flight, loose, or a 50/50 contest)."""
    dh = np.where(np.isnan(d_home), np.inf, d_home)
    da = np.where(np.isnan(d_away), np.inf, d_away)
    home_closer = dh < da
    d_win = np.minimum(dh, da)
    d_lose = np.maximum(dh, da)

    with np.errstate(invalid="ignore"):
        proximity = np.clip((CONTROL_R - d_win) / (CONTROL_R - TOUCH_R), 0.0, 1.0)
        margin = np.clip((d_lose - d_win) / CONTEST_MARGIN_M, 0.0, 1.0)
    strength = np.where(d_win < CONTROL_R, proximity * margin, 0.0)
    strength = np.nan_to_num(strength)

    fast = ~np.isnan(ball_s) & (ball_s > FLIGHT_SPEED)
    strength = np.where(fast, strength * FAST_TOUCH_WEIGHT, strength)
    return np.where(home_closer, strength, -strength)


def restart_priors(alive, game, data_dir=None):
    """{first alive frame of a run: S_H | S_A} from Smart Tagging set-piece Team tags.
    Throw-ins and other N/A tags give no prior (the emissions decide)."""
    restarts, _ = _db.smart_tag_intervals(game, data_dir)
    meta_path = _db.require(["atd_meta"], game, data_dir)[0]
    root = _db.ET.parse(meta_path).getroot()
    tnames = [(t.find("Name").text or "").strip().lower() for t in root.iter("Team") if t.find("Name") is not None]
    h_name = tnames[0] if len(tnames) > 0 else "barcelona"
    a_name = tnames[1] if len(tnames) > 1 else "athletic"
    h_tok = h_name.split()[0] if h_name else "barcelona"
    a_tok = a_name.split()[0] if a_name else "athletic"

    tagged = []
    for (s0, s1, team, _typ) in restarts:
        if h_name in team or h_tok in team:
            tagged.append((s0, s1, S_H))
        elif a_name in team or a_tok in team or "atletic" in team or "athletic" in team:
            tagged.append((s0, s1, S_A))
        else:
            tagged.append((s0, s1, None))

    starts = np.where(alive & ~np.r_[False, alive[:-1]])[0]
    tol = int(RESTART_MATCH_S * FPS)
    priors = {}
    for f0 in starts:
        best = None
        for (s0, s1, st) in tagged:
            if s0 - tol <= f0 <= s1 + tol:
                gap = abs(f0 - s0)
                if best is None or gap < best[0]:
                    best = (gap, st)
        if best is not None and best[1] is not None:
            priors[int(f0)] = best[1]
    return priors


# ----------------------------------------------------------------------------
# HMM
# ----------------------------------------------------------------------------
def emission_loglik(score, dead, graded, priors):
    """(n, 3) log emission scores. Only differences between states matter."""
    n = len(score)
    logB = np.zeros((n, 3))
    llr = CONTROL_LLR * score
    logB[:, S_H] = 0.5 * llr
    logB[:, S_A] = -0.5 * llr
    logB[:, S_D] = -np.inf               # graded alive frames can never be D

    for f0, st in priors.items():
        logB[f0, st] += RESTART_PRIOR_LLR

    forced_d = dead | ~graded            # hard constraint: Module 1 owns D
    logB[forced_d, S_H] = -np.inf
    logB[forced_d, S_A] = -np.inf
    logB[forced_d, S_D] = 0.0
    return logB


def transition_logmatrix():
    """log A[from, to]. Alive<->D entries are structural (mask-gated, free)."""
    ls, lw = np.log(P_STAY), np.log(P_SWITCH)
    return np.array([
        [ls, lw, LOG_ALIVE_TO_DEAD],
        [lw, ls, LOG_ALIVE_TO_DEAD],
        [LOG_DEAD_TO_ALIVE, LOG_DEAD_TO_ALIVE, 0.0],
    ])


def viterbi(logB, logA):
    """Most likely state path, log space. O(n * K^2), K = 3."""
    n, K = logB.shape
    back = np.zeros((n, K), dtype=np.int8)
    delta = logB[0].copy()               # flat initial distribution
    idx = np.arange(K)
    for t in range(1, n):
        cand = delta[:, None] + logA     # [from, to]
        b = cand.argmax(axis=0)
        back[t] = b
        delta = cand[b, idx] + logB[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(d_home, d_away, ball_s, dead, game="BAR-ATH", data_dir=None, verbose=True):
    """Full HMM stage. Returns labels (H/A/D on graded frames, X on ungraded)."""
    graded = _db.graded_mask(game)
    dead = dead & graded
    alive = graded & ~dead

    score = control_score(d_home, d_away, ball_s)
    priors = restart_priors(alive, game, data_dir)
    logB = emission_loglik(score, dead, graded, priors)
    path = viterbi(logB, transition_logmatrix())

    labels = STATE_LABELS[path].astype("<U1")
    labels[~graded] = UNGRADED

    # contract checks
    assert np.all(labels[dead] == DEAD), "dead frame decoded alive"
    assert np.all(np.isin(labels[alive], (HOME, AWAY))), "graded alive frame not H/A"

    if verbose:
        ev = alive & (score != 0)
        n_sw = int(np.sum((labels[1:] != labels[:-1]) & alive[1:] & alive[:-1]))
        print("  P(stay)={:.4f}  switch penalty={:.2f} nats  control LLR={:.3f}/frame".format(
            P_STAY, SWITCH_PENALTY, CONTROL_LLR))
        print("  alive frames with control evidence: {} of {} ({:.1f}%)".format(
            int(ev.sum()), int(alive.sum()), 100.0 * ev.sum() / max(1, alive.sum())))
        print("  restart priors applied: {}   in-play H<->A switches: {}".format(
            len(priors), n_sw))
    return labels
