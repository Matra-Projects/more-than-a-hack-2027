"""
Module 2 - POSSESSION ENGINE (tags H / A)

Runs AFTER the deadball engine and only on frames marked alive. Decides which team
has CONTROLLED possession using ball physics:

  * nearest-player distance in metres (anisotropic pitch scaling)
  * ball speed from the ATD (pass in flight -> possession is RETAINED, not contested)
  * dwell-time hysteresis: a team must hold the ball for DWELL frames before a
    switch is accepted (this is what implements "controlled" vs a simple touch,
    deflection or aerial duel - and what protects spellF1 from flicker)
  * restart prior: the Team tag on the Smart Tagging set piece tells us who
    restarts; throw-ins (Team = N/A) fall back to nearest player at the restart
  * missing ball: possession is HELD (last label). Ball-out extrapolation that
    crossed a boundary was already turned into D by Module 1.

Output: labels array with 'H' / 'A' / 'D' / 'X' for every one of the 199,319 frames.
"""

import numpy as np

from deadball_engine import (FPS, PITCH_L, PITCH_W, SECOND_HALF_KICKOFF,
                             smart_tag_intervals, graded_mask)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CONTROL_R = 4.5          # m: a player this close to a slow ball can control it
FLIGHT_SPEED = 9.0       # m/s: above this the ball is in flight; hold possession
DWELL = 16               # frames (~0.64 s) a challenger must hold the ball to take it
RESTART_DWELL = 6        # shorter confirmation right after a restart (cleaner signal)
MIN_RUN = 35             # alive runs shorter than this are flicker -> absorbed (~1.4 s)

HOME, AWAY, DEAD, UNGRADED = "H", "A", "D", "X"


def _nearest(px, py, bx, by):
    """Distance (m) and team (0/1) of the closest tracked player to the ball."""
    best_d, best_t = None, None
    for team, (xs, ys) in enumerate(((px[0], py[0]), (px[1], py[1]))):
        if xs is None or len(xs) == 0:
            continue
        d = np.hypot(xs - bx, ys - by)
        m = d.min()
        if best_d is None or m < best_d:
            best_d, best_t = float(m), team
    return best_d, best_t


def build_labels(ball_x, ball_y, ball_s, home_x, home_y, away_x, away_y,
                 dead, game="BAR-ATH", data_dir=None, verbose=True):
    n = len(ball_x)
    graded = graded_mask(game)
    labels = np.full(n, UNGRADED, dtype="<U1")
    labels[graded & dead] = DEAD

    restarts, _ = smart_tag_intervals(game, data_dir)
    # map restart frame -> team prior (None when the tag says N/A, e.g. throw-ins)
    restart_prior = {}
    for (s0, s1, team, typ) in restarts:
        if "barcelona" in team:
            restart_prior[s0] = HOME
        elif "atletic" in team or "athletic" in team:
            restart_prior[s0] = AWAY
        else:
            restart_prior[s0] = None    # N/A -> resolve by nearest player at restart

    alive = graded & ~dead
    current = None                  # current possession label
    cand, cand_since = None, None   # challenger team + frame it started challenging
    last_label_frame = None

    for f in range(n):
        if not alive[f]:
            current = None            # possession resets across a dead stretch
            cand = None
            continue

        bx, by, bs = ball_x[f], ball_y[f], ball_s[f]

        if np.isnan(bx):
            # ball unseen: hold possession. (Module 1 already turned true dead
            # stretches - out-of-bounds crossings, set pieces, long static gaps - into D.)
            if current is not None:
                labels[f] = current
                last_label_frame = f
            continue

        d, team = _nearest((home_x[f], away_x[f]), (home_y[f], away_y[f]), bx, by)

        if d is None:
            continue                    # no players tracked: hold previous

        in_flight = (not np.isnan(bs)) and bs > FLIGHT_SPEED and d > CONTROL_R
        if in_flight:
            # pass / shot / clearance travelling: possession stays where it was
            if current is not None:
                labels[f] = current
                last_label_frame = f
            continue

        controlled = d < CONTROL_R
        if not controlled:
            # loose ball between lines: possession stays with last team (per rules:
            # "in contested phases, possession stays with the last team that had it")
            if current is not None:
                labels[f] = current
                last_label_frame = f
            continue

        # a player is close to a playable ball
        if current is None:
            # right after a restart: use the smart-tag prior if we have one
            prior = None
            for s0 in restart_prior:
                if 0 <= f - s0 <= 3 * FPS:
                    prior = restart_prior[s0]
                    break
            if prior is not None:
                current = prior
                labels[f] = current
                last_label_frame = f
            else:
                need = RESTART_DWELL
                if cand == team and cand_since is not None and f - cand_since >= need:
                    current = HOME if team == 0 else AWAY
                    labels[f] = current
                    last_label_frame = f
                elif cand != team:
                    cand, cand_since = team, f
        elif (HOME if team == 0 else AWAY) == current:
            labels[f] = current
            last_label_frame = f
            cand = None
        else:
            # challenger team has a player on the ball; require dwell to switch
            need = DWELL
            if cand == team and cand_since is not None and f - cand_since >= need:
                current = HOME if team == 0 else AWAY
                labels[f] = current
                last_label_frame = f
                cand = None
            elif cand != team:
                cand, cand_since = team, f

    # --- spellF1 protection: absorb flicker runs -----------------------------
    labels = _absorb_short_runs(labels, alive)

    # 2nd-half safety: anything still unlabeled inside the graded halves that is
    # alive gets the last known possession (X inside a possession costs nothing on
    # spellF1, but a real label is better for macroF1)
    if verbose:
        graded_alive = alive.sum()
        lab = (labels == HOME) | (labels == AWAY)
        print("  alive frames: {}   labelled H/A: {} ({:.1f}%)".format(
            int(graded_alive), int((lab & alive).sum()), 100.0 * (lab & alive).sum() / graded_alive))
    return labels


def _absorb_short_runs(labels, alive):
    """Merge alive runs shorter than MIN_RUN into their longer neighbour.

    HHHHAHHHH with a 1-frame A (a deflection) becomes HHHHHHHHH - one true
    possession instead of three wrong ones. This single step is the difference
    between baseline_raw (spellF1 0.04) and baseline_smooth50 (0.24); doing it on
    *physics-grounded* runs instead of a blind window is why it does not also
    destroy macroF1.
    """
    n = len(labels)
    out = labels.copy()
    f = 0
    while f < n:
        if alive[f] and out[f] in (HOME, AWAY):
            g = f
            while g < n and alive[g] and out[g] == out[f]:
                g += 1
            run = g - f
            if run < MIN_RUN:
                prev_lab = out[f - 1] if f > 0 else None
                next_lab = out[g] if g < n else None
                prev_ok = prev_lab in (HOME, AWAY)
                next_ok = next_lab in (HOME, AWAY)
                if prev_ok and next_ok:
                    out[f:g] = prev_lab if (g - f) <= (run) else next_lab
                elif prev_ok:
                    out[f:g] = prev_lab
                elif next_ok:
                    out[f:g] = next_lab
            f = g
        else:
            f += 1
    return out

