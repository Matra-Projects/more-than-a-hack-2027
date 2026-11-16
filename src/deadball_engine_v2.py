"""
src/deadball_engine_v2.py - Stage 1 dead mask, v2.

Runs module_1's rules unchanged (trap, ballistic out-of-play, goal, stoppage) and
replaces its out-of-bounds and set-piece rules with the v2 versions below. Each rule's
output is OR-ed; module_1's cleanup (alive slivers < 1 s -> D, dead blips < 1.5 s -> alive)
runs last and the 2nd-half pre-kickoff trap (110808-113741) is always 100% D.

v2 rules (ball speed from src/ball_speed.py; positions for the line test from the
reprojection-gated ATD track in module_1.load_tracking, because ball_speed.py's trust gate
rejects every ball outside the projected pitch quad - an out-of-bounds ball is never
"trusted" there, so the out rule fired on 0 frames when fed those positions):
  out      ball centre outside [0, 105.3] x [0, 68.0] by more than 1 m (line width +
           tracking noise) for >= 2 frames -> D from the crossing until the restart.
  setpiece Smart Tagging SET PIECES tag = candidate region only (noisy, rule 4):
           start: walk back from the tag start while the ball is slower than 1.0 m/s
                  (or unseen), at most 15 s;
           end:   first frame where the ball is faster than 2.2 m/s AND the mean player
                  kinetic energy spikes (the field reacts to the restart).
           A tag with no settled ball before it and no restart after it adds nothing.
  restart (shared by both rules) = ball > 2.2 m/s while players' mean KE over the next
           3 s reaches KE_SPIKE_RATIO x its level in the second before.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import module_1 as m1     # noqa: E402  (not modified)

FPS = m1.FPS
PITCH_L, PITCH_W = m1.PITCH_L, m1.PITCH_W
SECOND_HALF_GRADED_START = m1.SECOND_HALF_GRADED_START
SECOND_HALF_KICKOFF = m1.SECOND_HALF_KICKOFF

OOB_TOL_M = 1.0           # line width (12 cm) + ball radius + tracking noise
OOB_PERSIST = 2           # frames
SETTLED_SPEED = 1.0       # m/s: placed / held ball
SETPIECE_BACK_MAX_S = 15.0
SETPIECE_FWD_PAD_S = 10.0
RESTART_BALL_SPEED = 2.2  # m/s: the softest restart (short free kick, tap kick-off)
KE_SPIKE_RATIO = 2.0      # KE ~ v^2, so 2x KE = field speed up ~41%
KE_REACT_S = 3.0          # the field must react within this
KE_BASE_S = 1.0           # baseline: the second before the strike
KE_SMOOTH_S = 0.5
PLAYER_HALF_WIN = 6       # player speed from displacement over +-0.24 s
MAX_PLAYER_SPEED = 11.0   # m/s: faster = track swap
MAX_DEAD_S = 90.0


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def mean_kinetic_energy(P):
    """Mean 0.5*v^2 (J/kg) over tracked players per frame, lightly smoothed."""
    n = len(P)
    k = PLAYER_HALF_WIN
    v = np.full((n, P.shape[1]), np.nan, dtype=np.float32)
    d = P[2 * k:] - P[:-2 * k]
    v[k:n - k] = np.hypot(d[..., 0], d[..., 1]) * (FPS / (2.0 * k))
    v[v > MAX_PLAYER_SPEED] = np.nan
    cnt = np.sum(~np.isnan(v), axis=1)
    ke = np.full(n, np.nan)
    ok = cnt >= 6
    ke[ok] = np.nanmean(0.5 * v[ok].astype(float) ** 2, axis=1)
    w = int(KE_SMOOTH_S * FPS)
    return pd.Series(ke).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=float, copy=True)


class _Ctx2:
    def __init__(self, bx, by, speed, ke):
        self.n = len(bx)
        self.bx, self.by = bx, by
        self.vis = ~np.isnan(bx)
        x = np.where(self.vis, bx, PITCH_L / 2)
        y = np.where(self.vis, by, PITCH_W / 2)
        self.inside = self.vis & (x >= -OOB_TOL_M) & (x <= PITCH_L + OOB_TOL_M) & \
            (y >= -OOB_TOL_M) & (y <= PITCH_W + OOB_TOL_M)
        s = np.nan_to_num(speed, nan=0.0)
        self.slow = (~self.vis) | (s < SETTLED_SPEED)
        self.strike = self.inside & (s > RESTART_BALL_SPEED)
        self.ke = np.nan_to_num(ke, nan=0.0)

    def restart(self, f0, f_max):
        f_max = min(f_max, self.n)
        react = int(KE_REACT_S * FPS)
        base = int(KE_BASE_S * FPS)
        for c in np.where(self.strike[f0:f_max])[0]:
            g = f0 + c
            before = self.ke[max(0, g - base):g + 1].mean()
            after = self.ke[g:min(self.n, g + react)].max()
            if after >= KE_SPIKE_RATIO * max(before, 1e-3):
                return g
        return None

    def walk_back(self, f, f_min):
        while f - 1 >= f_min and self.slow[f - 1]:
            f -= 1
        return f


def rule_out_v2(c):
    m = np.zeros(c.n, dtype=bool)
    crossings = []
    cap = int(MAX_DEAD_S * FPS)
    for a, b in _runs(c.vis & ~c.inside):
        if b - a < OOB_PERSIST:
            continue
        crossings.append(int(a))
        r = c.restart(b, a + cap)
        m[a:(r if r is not None else b)] = True
    return m, crossings


def rule_setpiece_v2(c, restarts, crossings):
    m = np.zeros(c.n, dtype=bool)
    back = int(SETPIECE_BACK_MAX_S * FPS)
    pad = int(SETPIECE_FWD_PAD_S * FPS)
    for (s0, s1, _team, typ) in sorted(restarts):
        s0 = int(np.clip(s0, 0, c.n - 1)); s1 = int(np.clip(s1, s0, c.n - 1))
        start = c.walk_back(s0 + 1, max(0, s0 - back))
        if any(k in typ for k in m1.OUT_CAUSED):           # restart caused by the ball going out
            before = [x for x in crossings if s0 - back <= x <= s0]
            if before:
                start = min(start, before[-1])
        r = c.restart(max(start, s0), s1 + pad)
        if r is None or (start > s0 and not c.slow[s0]):
            continue                                     # no evidence of a stoppage here
        m[start:r] = True
    return m


def build_dead_mask_v2(trk, bx, by, speed, game=m1.GAME, data_dir=None, verbose=True,
                       return_reasons=False):
    n = len(bx)
    graded = m1.graded_mask(game)
    _, reasons = m1.build_dead_mask(trk, game, data_dir, verbose=False, return_reasons=True,
                                    rules=("trap", "ballistic", "goal", "stoppage"))
    restarts, _goals = m1.smart_tag_intervals(game, data_dir)
    # line test on the reprojection-gated track (see module docstring), speed from ball_speed
    c = _Ctx2(trk["ball_x"], trk["ball_y"], speed, mean_kinetic_energy(trk["P"]))
    out, crossings = rule_out_v2(c)
    reasons["out_v2"] = out
    reasons["setpiece_v2"] = rule_setpiece_v2(c, restarts, sorted(crossings))

    dead = np.zeros(n, dtype=bool)
    for k in reasons:
        reasons[k] = reasons[k] & graded
        dead |= reasons[k]
    trap = np.zeros(n, dtype=bool)
    trap[SECOND_HALF_GRADED_START:SECOND_HALF_KICKOFF] = True
    dead = m1._cleanup(dead, trap) & graded
    dead |= trap

    if verbose:
        g = int(graded.sum())
        print("  dead frames: {} of {} graded ({:.1f}%)".format(int(dead.sum()), g,
                                                                100.0 * dead.sum() / g))
        for k, v in reasons.items():
            print("    {:<12} {:>6} frames ({:.1f}% of graded)".format(k, int(v.sum()),
                                                                         100.0 * v.sum() / g))
    if return_reasons:
        return dead, reasons
    return dead
