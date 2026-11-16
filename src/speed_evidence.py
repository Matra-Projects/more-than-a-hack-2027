"""
Speed evidence stage (ADDITIONAL, optional) - "who holds the ball while it is
unseen, and who receives it".

Sits between Stage 1 (module_1 dead mask, unchanged) and the HMM decoder
(hmm_engine, unchanged). It only ADDS per-frame log-likelihood terms:

    llr_home[f], llr_away[f]   added to the H / A emission scores on alive frames
    reception_mask[f]          predicted reception frames (opponent hand-over)
    static_dead[f]             extra D candidates from STATIC exits (ball static
                               > 2 s before vanishing + whole field calm)

and is consumed by emission_loglik_speed(), which wraps hmm_engine's emission
builder. The Viterbi core and module_1's dead mask are not modified, so the
frozen pipeline (build_submission.py -> team_engine_hmm.csv) is unaffected.

Pipeline
  1. Ball speed from consecutive raw ATD positions (dx*105.3, dy*68.0, *25 fps),
     3-frame rolling median. The ATD's own 3rd ball column is cross-checked and
     reported (it is not a usable speed on this file - see module_1).
  2. Homography trust gate: ball projected to pixels through the frame's 3x3 H;
     outside the projected pitch quadrilateral, or reprojection_error > 1.0 ->
     ball UNTRUSTED, treated exactly like a missing ball (flat emissions).
  3. Exit-speed classifier on the last visible frames before every gap:
     FAST (median > 9 m/s, decelerating) / STATIC (< 1.5 m/s) / AMBIGUOUS.
  4. FAST: hold prior toward the kicking team until the predicted reception
     (2D extrapolation with grass friction / air drag until speed < 6 m/s or the
     path reaches a tracked player); an opponent at the reception point gets a
     switch discount so the HMM can hand over without the full dwell.
  5. Reappearance check: the team actually on the ball when it reappears is
     compared with the prediction; a match gets the dwell credit, a mismatch
     gets nothing (normal dwell applies).

The "last possession team" at a gap is read from a first, unmodified HMM decode
(two-pass), because possession is a property of the decoded sequence.

TACTICAL SOURCE ONLY. numpy + pandas + stdlib.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import hmm_engine as he  # noqa: E402
from games import GAME, GAMES, require  # noqa: E402
import load_homography as lh  # noqa: E402

_db, _pe = he.import_engines()

# ----------------------------------------------------------------------------
# Parameters (physical meaning)
# ----------------------------------------------------------------------------
FPS = he.FPS
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]          # anisotropic scaling, metres
SPEED_MEDIAN_FRAMES = 3       # rolling median on the raw frame-to-frame speed
REPROJ_UNTRUSTED = 1.0        # homography reprojection error above this -> distrust ball

EXIT_FRAMES = 4               # visible frames inspected at the start of a gap
FAST_EXIT_SPEED = 9.0         # m/s: a kicked ball (matches Module 2's FLIGHT_SPEED)
STATIC_EXIT_SPEED = 1.5       # m/s: ball at rest / being gathered
STATIC_EXIT_FRAMES = 3        # ... over the last 3 visible frames
DECEL_TOL = 1.15              # "decelerating": last speed <= first speed * this

RECEIVE_SPEED = 6.0           # m/s: below this a pass is controllable = received
RECEIVE_R = 1.5               # m: path within this of a tracked player = reached him
ROLL_DECEL = _db.ROLL_DECEL   # 1.5 m/s^2 grass friction
AERIAL_SPEED = _db.AERIAL_SPEED   # 15 m/s: above this assume airborne
AIR_DRAG_K = _db.AIR_DRAG_K   # 0.013 /m quadratic drag
EXTRAP_MAX_S = 3.0            # a pass is received within ~3 s (40 m at 13 m/s)

# Log-likelihood terms, all expressed in units of the HMM's own switch penalty
# so they stay consistent with its dwell logic (penalty = one switch).
FAST_HOLD_LLR = 0.5 * he.CONTROL_LLR     # per frame toward the kicking team in flight
STATIC_HOLD_LLR = 0.1 * he.CONTROL_LLR   # weak per-frame hold after a static exit
RECEPTION_DISCOUNT = 0.6                 # share of the switch penalty waived at a
                                         # predicted opponent reception (the remaining
                                         # 40% must still come from real control)
RECEPTION_LLR = RECEPTION_DISCOUNT * he.SWITCH_PENALTY
REAPPEAR_FRAMES = 5                      # frames after reappearance that are checked
MATCH_LLR = 0.5 * he.SWITCH_PENALTY      # credit when reappearance confirms the prediction

# STATIC-exit D candidate (handed to the D side as an extra candidate)
STATIC_DEAD_S = 2.0           # ball static this long before it vanished ...
STATIC_DEAD_MOTION = _db.STOP_MOTION     # ... and the whole field this calm (0.9 m/s)

FAST, STATIC, AMBIGUOUS, NO_EXIT = "FAST", "STATIC", "AMBIGUOUS", "NO_EXIT"


# ----------------------------------------------------------------------------
# 1. Ball speed + ATD cross-check
# ----------------------------------------------------------------------------
def load_raw_ball(game=GAME, data_dir=None):
    """Raw ATD ball: normalised x, y and the ATD's own 3rd column."""
    (raw_path,) = require(["atd_raw"], game, data_dir)
    n = GAMES[game]["n_video_frames"]
    nx = np.full(n, np.nan); ny = np.full(n, np.nan); col = np.full(n, np.nan)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            b = parts[2]
            if 0 <= f < n and b[0] != "N":
                x, y, s = b.split(",")
                nx[f] = float(x); ny[f] = float(y); col[f] = float(s)
    return nx, ny, col


def ball_speed(nx, ny):
    """m/s from consecutive positions, anisotropic metres, 3-frame rolling median.
    NaN wherever either neighbour is unseen."""
    raw = np.full(len(nx), np.nan)
    raw[1:] = np.hypot(np.diff(nx) * PITCH_L, np.diff(ny) * PITCH_W) * FPS
    med = pd.Series(raw).rolling(SPEED_MEDIAN_FRAMES, center=True, min_periods=SPEED_MEDIAN_FRAMES)
    out = med.median().to_numpy(dtype=float, copy=True)
    out[np.isnan(raw)] = np.nan
    return out, raw


def atd_column_check(speed, col):
    """Correlation of our speed with the ATD column (raw and as kph -> m/s)."""
    ok = ~np.isnan(speed) & ~np.isnan(col) & (speed < 45)
    if ok.sum() < 100:
        return {"n": int(ok.sum()), "corr": float("nan")}
    return {"n": int(ok.sum()),
            "corr": float(np.corrcoef(speed[ok], col[ok])[0, 1]),
            "median_ours": float(np.median(speed[ok])),
            "median_col_as_mps": float(np.median(col[ok] / 3.6))}


# ----------------------------------------------------------------------------
# 2. Homography trust gate
# ----------------------------------------------------------------------------
def _project(Hm, x, y):
    w = Hm[:, 2, 0] * x + Hm[:, 2, 1] * y + Hm[:, 2, 2]
    u = (Hm[:, 0, 0] * x + Hm[:, 0, 1] * y + Hm[:, 0, 2]) / w
    v = (Hm[:, 1, 0] * x + Hm[:, 1, 1] * y + Hm[:, 1, 2]) / w
    return u, v


def trust_gate(nx, ny, game=GAME, data_dir=None):
    """True where the ball position is trusted (inside the projected pitch quad
    and reprojection error <= REPROJ_UNTRUSTED). Missing balls are untrusted."""
    Hm, reproj, valid = lh.load(game, data_dir)
    n = len(nx)
    Hm = np.asarray(Hm, dtype=float)[:n]
    reproj = np.asarray(reproj, dtype=float)[:n]
    valid = np.asarray(valid, dtype=bool)[:n]
    seen = ~np.isnan(nx) & valid
    bx = np.where(seen, nx, 0.5); by = np.where(seen, ny, 0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        pu, pv = _project(Hm, bx, by)
        corners = [_project(Hm, np.full(n, cx), np.full(n, cy))
                   for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))]
        # point inside a convex quad: all edge cross products share one sign
        sgn = []
        for k in range(4):
            (ax, ay), (bx2, by2) = corners[k], corners[(k + 1) % 4]
            sgn.append((bx2 - ax) * (pv - ay) - (by2 - ay) * (pu - ax))
        sgn = np.array(sgn)
        inside = np.all(sgn >= 0, axis=0) | np.all(sgn <= 0, axis=0)
    rp = np.nan_to_num(reproj, nan=np.inf)
    return seen & inside & (rp <= REPROJ_UNTRUSTED)


# ----------------------------------------------------------------------------
# 3-5. Exit classifier, reception prediction, reappearance check
# ----------------------------------------------------------------------------
def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def classify_exit(speed, trusted, g0):
    """Class of the gap starting at g0 from the last visible frames' speeds."""
    lo = g0 - EXIT_FRAMES
    if lo < 0 or not trusted[lo:g0].all():
        return NO_EXIT
    prof = speed[lo:g0]
    if np.isnan(prof).any():
        prof = prof[~np.isnan(prof)]
        if len(prof) < STATIC_EXIT_FRAMES:
            return NO_EXIT
    if np.median(prof) > FAST_EXIT_SPEED and prof[-1] <= prof[0] * DECEL_TOL:
        return FAST
    if np.all(prof[-STATIC_EXIT_FRAMES:] < STATIC_EXIT_SPEED):
        return STATIC
    return AMBIGUOUS


def _predict_reception(g0, g1, mx, my, trk):
    """Extrapolate the exit trajectory; return (frame, team) of the predicted
    reception or (None, None)."""
    x1, y1 = mx[g0 - 1], my[g0 - 1]
    x3, y3 = mx[g0 - 3], my[g0 - 3]
    vx, vy = (x1 - x3) * FPS / 2.0, (y1 - y3) * FPS / 2.0
    speed = float(np.hypot(vx, vy))
    if not np.isfinite(speed) or speed <= 0:
        return None, None
    ux, uy = vx / speed, vy / speed
    P = trk["P"]; team_of = trk["slot_team"]
    dt = 1.0 / FPS
    x, y = x1, y1
    for k in range(int(EXTRAP_MAX_S * FPS)):
        f = g0 + k
        if f >= len(P):
            break
        if speed > AERIAL_SPEED:
            speed -= AIR_DRAG_K * speed * speed * dt
        else:
            speed = max(0.0, speed - ROLL_DECEL * dt)
        x += ux * speed * dt
        y += uy * speed * dt
        d = np.hypot(P[f, :, 0] - x, P[f, :, 1] - y)
        d = np.where(np.isnan(d), np.inf, d)
        j = int(np.argmin(d))
        reached = d[j] <= RECEIVE_R
        if speed < RECEIVE_SPEED or reached:
            if not np.isfinite(d[j]):
                return None, None
            return f, int(team_of[j])
        if x < 0 or x > PITCH_L or y < 0 or y > PITCH_W:
            return None, None          # left the pitch: Stage 1's business
    return None, None


def build(trk, dead, labels_pass1, d_home, d_away, game=GAME, data_dir=None, verbose=True):
    """All per-frame speed-evidence arrays. labels_pass1 = unmodified HMM decode."""
    n = len(trk["ball_x"])
    graded = _db.graded_mask(game)
    alive = graded & ~dead

    nx, ny, col = load_raw_ball(game, data_dir)
    speed, _raw = ball_speed(nx, ny)
    trusted = trust_gate(nx, ny, game, data_dir)
    speed_t = np.where(trusted, speed, np.nan)
    mx = nx * PITCH_L; my = ny * PITCH_W

    llr_home = np.zeros(n); llr_away = np.zeros(n)
    reception = np.zeros(n, dtype=bool)
    static_dead = np.zeros(n, dtype=bool)
    counts = {FAST: 0, STATIC: 0, AMBIGUOUS: 0, NO_EXIT: 0}
    recv = {"same": 0, "other": 0, "none": 0}
    reap = {"match": 0, "mismatch": 0, "unknown": 0}
    team_lab = {0: he.HOME, 1: he.AWAY}

    def add(team, sl, val):
        (llr_home if team == 0 else llr_away)[sl] += val

    motion = trk.get("motion")
    if motion is None:
        motion = _db.field_motion(trk["P"])
    mot = np.nan_to_num(motion, nan=np.inf)

    for g0, g1 in _runs(~trusted):
        if not alive[g0:g1].any() or g0 == 0:
            continue
        cls = classify_exit(speed_t, trusted, g0)
        counts[cls] += 1
        last = labels_pass1[g0 - 1]
        last_team = 0 if last == he.HOME else 1 if last == he.AWAY else None
        predicted = last_team

        if cls == FAST and last_team is not None:
            rf, rteam = _predict_reception(g0, g1, mx, my, trk)
            hold_end = g1 if rf is None else min(g1, rf)
            add(last_team, slice(g0, hold_end), FAST_HOLD_LLR)
            if rf is None:
                recv["none"] += 1
            elif rteam == last_team:
                recv["same"] += 1
            else:
                recv["other"] += 1
                predicted = rteam
                reception[rf] = True
                add(rteam, slice(rf, rf + 1), RECEPTION_LLR)
        elif cls == STATIC and last_team is not None:
            add(last_team, slice(g0, g1), STATIC_HOLD_LLR)
            need = int(STATIC_DEAD_S * FPS)
            lo = g0 - need
            if lo >= 0 and trusted[lo:g0].all() and np.all(np.nan_to_num(speed_t[lo:g0], nan=0)
                                                          < STATIC_EXIT_SPEED):
                if np.all(mot[lo:g1] < STATIC_DEAD_MOTION):
                    static_dead[lo:g1] = True

        # reappearance check
        if predicted is None or g1 >= n:
            reap["unknown"] += 1
            continue
        sl = slice(g1, min(n, g1 + REAPPEAR_FRAMES))
        dh = np.nanmedian(d_home[sl]) if np.any(~np.isnan(d_home[sl])) else np.nan
        da = np.nanmedian(d_away[sl]) if np.any(~np.isnan(d_away[sl])) else np.nan
        if not (np.isfinite(dh) or np.isfinite(da)) or min(np.nan_to_num(dh, nan=np.inf),
                                                          np.nan_to_num(da, nan=np.inf)) > he.CONTROL_R:
            reap["unknown"] += 1
            continue
        actual = 0 if np.nan_to_num(dh, nan=np.inf) < np.nan_to_num(da, nan=np.inf) else 1
        if actual == predicted:
            reap["match"] += 1
            add(actual, slice(g1, g1 + 1), MATCH_LLR)
        else:
            reap["mismatch"] += 1

    # only alive frames carry evidence; Stage 1 D frames are forced anyway
    llr_home[~alive] = 0.0
    llr_away[~alive] = 0.0
    static_dead &= graded

    info = {"gap_classes": counts, "reception": recv, "reappearance": reap,
            "atd_check": atd_column_check(speed, col),
            "untrusted_seen": int((~np.isnan(nx) & ~trusted & graded).sum())}
    if verbose:
        tot = max(1, sum(counts.values()))
        print("  ATD 3rd column vs position speed: corr {:.3f} (n={})".format(
            info["atd_check"]["corr"], info["atd_check"]["n"]))
        print("  untrusted visible ball frames (outside pitch quad / reproj > {}): {}".format(
            REPROJ_UNTRUSTED, info["untrusted_seen"]))
        print("  gaps: " + "  ".join("{} {} ({:.1f}%)".format(k, v, 100.0 * v / tot)
                                     for k, v in counts.items()))
        print("  FAST receptions: same team {same}, opponent {other}, none {none}".format(**recv))
        print("  reappearance vs prediction: match {match}, mismatch {mismatch}, "
              "unknown {unknown}".format(**reap))
        print("  STATIC-exit D candidates: {} frames".format(int(static_dead.sum())))
    return {"llr_home": llr_home, "llr_away": llr_away, "reception_mask": reception,
            "static_dead": static_dead, "trusted": trusted, "speed": speed, "info": info}


# ----------------------------------------------------------------------------
# Emission builder wrapper + decode (Viterbi core untouched)
# ----------------------------------------------------------------------------
def emission_loglik_speed(score, dead, graded, priors, llr_home, llr_away):
    logB = he.emission_loglik(score, dead, graded, priors)
    alive = graded & ~dead
    logB[alive, he.S_H] += llr_home[alive]
    logB[alive, he.S_A] += llr_away[alive]
    assert not np.isnan(logB).any(), "NaN leaked into the Viterbi log-likelihoods"
    assert np.isfinite(logB[alive][:, :2]).all(), "non-finite H/A score on an alive frame"
    return logB


def decode_speed(trk, dead_stage1, game=GAME, data_dir=None, use_static_dead=True, verbose=True):
    """Two-pass decode: pass 1 = unmodified HMM, pass 2 = HMM + speed evidence."""
    graded = _db.graded_mask(game)
    d_home, d_away = he.evidence_from_tracking(trk)
    pass1 = he.decode(d_home, d_away, trk["ball_s"], dead_stage1, game, data_dir, verbose=False)

    ev = build(trk, dead_stage1, pass1, d_home, d_away, game, data_dir, verbose)
    dead = dead_stage1 | (ev["static_dead"] if use_static_dead else False)
    alive = graded & ~dead

    score = he.control_score(d_home, d_away, trk["ball_s"])
    score = np.where(ev["trusted"], score, 0.0)          # untrusted ball = missing ball
    priors = he.restart_priors(alive, game, data_dir)
    logB = emission_loglik_speed(score, dead, graded, priors, ev["llr_home"], ev["llr_away"])
    path = he.viterbi(logB, he.transition_logmatrix())

    labels = he.STATE_LABELS[path].astype("<U1")
    labels[~graded] = he.UNGRADED
    assert np.all(labels[dead] == he.DEAD)
    assert np.all(np.isin(labels[alive], (he.HOME, he.AWAY)))
    return labels, dead, ev
