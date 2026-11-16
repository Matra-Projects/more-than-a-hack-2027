"""
src/hmm_engine_v2.py - 3-state (H / A / D) HMM with a physics-conditioned,
frame-dependent transition matrix.

Transition from frame t-1 to frame t, for the alive states:
    default                         P(stay) = 0.985  (a switch costs log(.985/.015) = 4.18 nats)
    sustained control by team c     row c:      P(stay) = 0.998  (c's player within 1.0 m of the
      (d_c < 1.0 m for > 10 frames)              ball for > 0.4 s: a dribble is not given away)
    clean turnover towards team T   row not-T:  P(stay) = 0.85   (T's player within 1.0 m of a
      (d_T < 1.0 m, speed < 3 m/s)               ball slower than 3 m/s: switch cost drops to
                                                 1.73 nats, so the boundary does not lag)
Rows are per team, not global: a home dribble makes staying H cheap without making the
switch A -> H expensive, and a turnover towards T only cheapens the switch INTO T.

D is owned by Stage 1 (dead mask = hard constraint, alive <-> D transitions are free),
exactly as in hmm_engine.py. Smart Tagging restart team tags stay a soft prior.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hmm_engine as v1            # noqa: E402  (restart_priors, constants; not modified)
import possession_engine_v2 as pe  # noqa: E402

FPS = pe.FPS
S_H, S_A, S_D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"])

P_STAY_DEFAULT = 0.985
P_STAY_DRIBBLE = 0.998
P_STAY_TURNOVER = 0.85
DRIBBLE_D = 1.0          # m: ball at the feet
DRIBBLE_MIN_FRAMES = 10  # > 0.4 s of continuous close control
TURNOVER_D = 1.0         # m
TURNOVER_SPEED = 3.0     # m/s: ball stopped / taken off the opponent's foot, not a pass

SWITCH_PENALTY = float(np.log(P_STAY_DEFAULT / (1 - P_STAY_DEFAULT)))
RESTART_PRIOR_LLR = SWITCH_PENALTY    # soft: one switch worth of evidence


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def stay_probabilities(d_home, d_away, speed):
    """Per-frame P(stay) for the H row and the A row, plus diagnostic flags."""
    n = len(d_home)
    dh = np.where(np.isnan(d_home), np.inf, d_home)
    da = np.where(np.isnan(d_away), np.inf, d_away)
    s = np.where(np.isnan(speed), np.inf, speed)
    stay = np.full((n, 2), P_STAY_DEFAULT)

    dribble = np.zeros((n, 2), dtype=bool)
    for col, (mine, other) in enumerate(((dh, da), (da, dh))):
        close = (mine < DRIBBLE_D) & (mine < other)
        for a, b in _runs(close):
            if b - a > DRIBBLE_MIN_FRAMES:
                dribble[a + DRIBBLE_MIN_FRAMES:b, col] = True
    stay[dribble] = P_STAY_DRIBBLE

    turnover = np.zeros((n, 2), dtype=bool)            # column = team gaining the ball
    turnover[:, 0] = (dh < TURNOVER_D) & (dh < da) & (s < TURNOVER_SPEED)
    turnover[:, 1] = (da < TURNOVER_D) & (da < dh) & (s < TURNOVER_SPEED)
    stay[turnover[:, 0], 1] = np.minimum(stay[turnover[:, 0], 1], P_STAY_TURNOVER)  # A row
    stay[turnover[:, 1], 0] = np.minimum(stay[turnover[:, 1], 0], P_STAY_TURNOVER)  # H row
    return stay, dribble, turnover


def emission_loglik(llr, dead, graded, priors):
    n = len(llr)
    logB = np.zeros((n, 3))
    logB[:, S_H] = 0.5 * llr
    logB[:, S_A] = -0.5 * llr
    logB[:, S_D] = -np.inf
    for f0, st in priors.items():
        logB[f0, st] += RESTART_PRIOR_LLR
    forced = dead | ~graded
    logB[forced, S_H] = -np.inf
    logB[forced, S_A] = -np.inf
    logB[forced, S_D] = 0.0
    assert not np.isnan(logB).any(), "NaN in emissions"
    return logB


def viterbi_dynamic(logB, stay):
    """Viterbi with a frame-dependent transition matrix (log space).
    stay[t] = (P(H->H), P(A->A)) for the step t-1 -> t; alive<->D is free."""
    n = len(logB)
    ls = np.log(stay)
    lw = np.log1p(-stay)
    back = np.zeros((n, 3), dtype=np.int8)
    dH, dA, dD = logB[0]
    for t in range(1, n):
        lsh, lsa = ls[t]
        lwh, lwa = lw[t]
        # to H: from H stay, from A switch, from D free
        cands = (dH + lsh, dA + lwa, dD)
        bH = 0 if cands[0] >= cands[1] else 1
        if cands[2] > cands[bH]:
            bH = 2
        nH = cands[bH]
        cands_a = (dH + lwh, dA + lsa, dD)
        bA = 1 if cands_a[1] >= cands_a[0] else 0
        if cands_a[2] > cands_a[bA]:
            bA = 2
        nA = cands_a[bA]
        bD = 0 if dH >= dA else 1
        if dD >= max(dH, dA):
            bD = 2
        nD = (dH, dA, dD)[bD]
        eH, eA, eD = logB[t]
        dH, dA, dD = nH + eH, nA + eA, nD + eD
        back[t] = (bH, bA, bD)
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax((dH, dA, dD)))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_v2(trk, bx, by, speed, dead, game, data_dir=None, verbose=True):
    """Stage 2 v2. Returns (labels H/A/D on graded frames, X elsewhere; diagnostics)."""
    db = v1._db
    graded = db.graded_mask(game)
    dead = dead & graded
    alive = graded & ~dead

    d_home, d_away = pe.distances(trk["P"], trk["slot_team"], bx, by)
    reg, team = pe.control_regimes(d_home, d_away, speed)
    llr = pe.control_llr(reg, team)
    stay, dribble, turnover = stay_probabilities(d_home, d_away, speed)
    priors = v1.restart_priors(alive, game, data_dir)

    # pass 1: no gap evidence -> who had the ball when it went missing
    path1 = viterbi_dynamic(emission_loglik(llr, dead, graded, priors), stay)
    team_before = np.where(path1 == S_H, 1, np.where(path1 == S_A, -1, 0))
    gap_llr, gaps = pe.intercept_gaps(trk["P"], trk["slot_team"], bx, by, speed,
                                      alive, team_before)
    # pass 2: + interception / hold evidence over the gaps
    path = viterbi_dynamic(emission_loglik(llr + gap_llr, dead, graded, priors), stay)

    labels = STATE_LABELS[path].astype("<U1")
    labels[~graded] = "X"
    assert np.all(labels[dead] == "D"), "dead frame decoded alive"
    assert np.all(np.isin(labels[alive], ("H", "A"))), "graded alive frame not H/A"
    assert set(np.unique(labels[graded])) <= {"H", "A", "D"}

    diag = dict(reg=reg, dribble=dribble, turnover=turnover, gaps=gaps)
    if verbose:
        al = alive
        print("  regimes on alive frames:")
        for i, name in enumerate(pe.REGIMES):
            k = int(((reg == i) & al).sum())
            print("    {:<12} {:>7}  ({:.1f}%)".format(name, k, 100.0 * k / max(1, al.sum())))
        print("  P(stay): dribble rows {} frames, turnover rows {} frames".format(
            int((dribble.any(1) & al).sum()), int((turnover.any(1) & al).sum())))
        ni = sum(g["intercepted"] for g in gaps)
        print("  missing-ball gaps (alive, bracketed): {}  intercepted: {}  held: {}".format(
            len(gaps), ni, len(gaps) - ni))
        sw = int(np.sum((labels[1:] != labels[:-1]) & al[1:] & al[:-1]))
        print("  restart priors: {}   in-play H<->A switches: {}".format(len(priors), sw))
    return labels, diag
