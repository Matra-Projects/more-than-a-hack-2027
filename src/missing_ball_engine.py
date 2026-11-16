"""
Missing-ball engine - who has the ball while the tracker cannot see it.

For every visibility gap (ball unseen OR untrusted by ball_speed's homography
gate) that touches Stage-1-alive frames, the physics at the moment the ball
vanished and at the moment it reappears is turned into INTERNAL tags:

    CARRY_H / CARRY_A    ball moving at the same speed as the nearest player of
                         that team at the exit (dribble / shielding): continuous
                         control -> that team for the whole gap
    AIR_H / AIR_A        fast exit (> FLIGHT_SPEED) that is not accelerating: a
                         kicked ball in flight, possession stays with the kicker's
                         team until it lands
    LANDED_H / LANDED_A  the ball reappears much slower than the flight predicts
                         ("suddenly slowed down" = it landed / was received): from
                         the landing frame (closest point of the predicted path to
                         the reappearance spot) the team on the ball at
                         reappearance has it
    DEAD_UNSEEN          Stage-1 dead, or a static exit (ball at rest > 2 s) with
                         the whole field calm for the whole gap
    HOLD                 no physical signal: possession held by the HMM's P(stay)

The tags are INTERNAL ONLY. They are converted to emission log-likelihood terms
(llr_home / llr_away) and an extra dead mask, and the unchanged hmm_engine
Viterbi decodes them, so the output alphabet is exactly H / A / D on graded
frames (X only on ungraded frames, per the submission contract).

TACTICAL SOURCE ONLY. numpy + pandas + stdlib.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hmm_engine as he  # noqa: E402
import ball_speed  # noqa: E402

_db, _pe = he.import_engines()

# ----------------------------------------------------------------------------
# Parameters (physical meaning)
# ----------------------------------------------------------------------------
FPS = he.FPS
PITCH_L, PITCH_W = _db.PITCH_L, _db.PITCH_W
MAX_BALL_SPEED = _db.MAX_BALL_SPEED   # 42 m/s: faster frame-to-frame = tracker jump

EXIT_FRAMES = 4               # trusted frames inspected before the gap
CARRY_R = 1.5                 # m: ball this close to a player = at his feet
CARRY_DV = 1.5                # m/s: ball and player speeds this close = moving together
FLIGHT_SPEED = he.FLIGHT_SPEED    # 9 m/s: faster = struck ball in flight
ACCEL_TOL = 1.15              # "not accelerating": last exit speed <= first * this
KICK_LOOKBACK_S = 1.5         # walk back this far to find the strike
STATIC_SPEED = 1.5            # m/s: ball at rest
STATIC_S = 2.0                # ... for this long before vanishing
CALM_MOTION = _db.STOP_MOTION     # 0.9 m/s median player speed = whole field calm

RECEIVE_SPEED = 6.0           # m/s: a pass slower than this is controllable
LANDED_RATIO = 0.5            # reappearing slower than half the predicted flight speed = landed
REAPPEAR_FRAMES = 5           # frames used to read the reappearance context
ROLL_DECEL = _db.ROLL_DECEL   # 1.5 m/s^2 grass
AERIAL_SPEED = _db.AERIAL_SPEED   # 15 m/s
AIR_DRAG_K = _db.AIR_DRAG_K   # 0.013 /m
EXTRAP_MAX_S = 4.0            # longest flight considered (a 50 m diagonal)

# Tag -> emission strength, in units of the HMM's own scale so they stay
# consistent with its dwell logic (CONTROL_LLR = one frame of clear control).
CARRY_LLR = 1.0 * he.CONTROL_LLR      # as strong as seeing the player on the ball
AIR_LLR = 0.5 * he.CONTROL_LLR        # flight: kicker keeps it, but weaker
LANDED_LLR = 0.5 * he.CONTROL_LLR     # receiver after the landing frame ...
LANDING_BONUS = 0.6 * he.SWITCH_PENALTY   # ... plus a hand-over credit at the landing

TAGS = ("HOLD", "CARRY_H", "CARRY_A", "AIR_H", "AIR_A", "LANDED_H", "LANDED_A", "DEAD_UNSEEN")
_TEAM = {0: "H", 1: "A"}


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _player_speed(P, f, j, k=2):
    lo, hi = max(0, f - k), min(len(P) - 1, f + k)
    d = P[hi, j] - P[lo, j]
    if np.isnan(d).any() or hi == lo:
        return np.nan
    return float(np.hypot(d[0], d[1]) * FPS / (hi - lo))


def _nearest(P, team_of, f, x, y):
    """(distance, team, slot) of the nearest tracked player to (x, y) at frame f."""
    d = np.hypot(P[f, :, 0] - x, P[f, :, 1] - y)
    d = np.where(np.isnan(d), np.inf, d)
    j = int(np.argmin(d))
    return float(d[j]), int(team_of[j]), j


def _flight(x, y, vx, vy, frames):
    """Extrapolated (x, y, speed) per frame: air drag when fast, grass friction slow."""
    sp = float(np.hypot(vx, vy))
    ux, uy = vx / sp, vy / sp
    out = []
    dt = 1.0 / FPS
    for _ in range(frames):
        if sp > AERIAL_SPEED:
            sp -= AIR_DRAG_K * sp * sp * dt
        else:
            sp = max(0.0, sp - ROLL_DECEL * dt)
        x += ux * sp * dt
        y += uy * sp * dt
        out.append((x, y, sp))
    return out


def tag_gaps(trk, dead, bx, by, speed, trusted, game="BAR-ATH", verbose=True):
    """Per-frame internal tags (index into TAGS) + per-gap diagnostics."""
    n = len(bx)
    graded = _db.graded_mask(game)
    alive = graded & ~dead
    P, team_of = trk["P"], trk["slot_team"]
    mot = np.nan_to_num(trk["motion"] if trk.get("motion") is not None
                        else _db.field_motion(P), nan=np.inf)
    spd = np.where(speed > MAX_BALL_SPEED, np.nan, speed)

    tags = np.zeros(n, dtype=np.int8)                 # HOLD
    tags[graded & dead & ~trusted] = TAGS.index("DEAD_UNSEEN")
    landing = np.zeros(n, dtype=bool)
    stats = {"gaps": 0, "CARRY": 0, "AIR": 0, "AIR->LANDED": 0, "STATIC_DEAD": 0, "HOLD": 0}

    for g0, g1 in _runs(~trusted):
        if g0 < EXIT_FRAMES + 1 or not alive[g0:g1].any():
            continue
        stats["gaps"] += 1
        sl = slice(g0, g1)
        prof = spd[g0 - EXIT_FRAMES:g0]
        if not trusted[g0 - EXIT_FRAMES:g0].all() or np.isnan(prof).sum() > 1:
            stats["HOLD"] += 1
            continue
        prof = prof[~np.isnan(prof)]
        v_exit = float(np.median(prof))
        x1, y1 = bx[g0 - 1], by[g0 - 1]
        d_c, team_c, j = _nearest(P, team_of, g0 - 1, x1, y1)

        # --- STATIC exit + calm field -> dead --------------------------------
        need = int(STATIC_S * FPS)
        if g0 - need >= 0 and trusted[g0 - need:g0].all() and \
                np.all(np.nan_to_num(spd[g0 - need:g0], nan=0.0) < STATIC_SPEED) and \
                np.all(mot[g0 - need:g1] < CALM_MOTION):
            tags[sl] = TAGS.index("DEAD_UNSEEN")
            stats["STATIC_DEAD"] += 1
            continue

        # --- CARRY: ball moving with the nearest player -----------------------
        if v_exit < FLIGHT_SPEED and d_c <= CARRY_R:
            u = _player_speed(P, g0 - 1, j)
            if np.isfinite(u) and abs(v_exit - u) <= CARRY_DV:
                tags[sl] = TAGS.index("CARRY_" + _TEAM[team_c])
                stats["CARRY"] += 1
                continue

        # --- AIR: struck ball in flight --------------------------------------
        if v_exit > FLIGHT_SPEED and prof[-1] <= prof[0] * ACCEL_TOL:
            k = g0 - 1
            lim = max(1, g0 - int(KICK_LOOKBACK_S * FPS))
            while k - 1 >= lim and trusted[k - 1] and np.nan_to_num(spd[k - 1]) > FLIGHT_SPEED:
                k -= 1
            dk, kicker, _ = _nearest(P, team_of, k - 1, bx[k - 1], by[k - 1]) \
                if trusted[k - 1] else (np.inf, None, None)
            if kicker is None or dk > he.CONTROL_R:
                stats["HOLD"] += 1
                continue                          # nobody struck it: leave to the HMM
            stats["AIR"] += 1
            x3, y3 = bx[g0 - 3], by[g0 - 3]
            vx, vy = (x1 - x3) * FPS / 2.0, (y1 - y3) * FPS / 2.0
            if not np.isfinite(vx + vy) or np.hypot(vx, vy) == 0:
                tags[sl] = TAGS.index("AIR_" + _TEAM[kicker])
                continue
            path = _flight(x1, y1, vx, vy, min(g1 - g0, int(EXTRAP_MAX_S * FPS)))
            land_at = None
            if g1 < n and trusted[g1]:
                rs = spd[g1:min(n, g1 + REAPPEAR_FRAMES)]
                rs = rs[~np.isnan(rs)]
                k_re = min(len(path) - 1, g1 - g0 - 1)
                pred = path[k_re][2] if k_re >= 0 else v_exit
                if len(rs) and np.median(rs) < max(RECEIVE_SPEED, LANDED_RATIO * pred):
                    # landed / received during the gap: landing frame = closest point
                    # of the predicted path to where the ball reappeared
                    rx, ry = bx[g1], by[g1]
                    dists = [np.hypot(px - rx, py - ry) for px, py, _ in path]
                    land_at = g0 + int(np.argmin(dists))
            if land_at is None:
                tags[sl] = TAGS.index("AIR_" + _TEAM[kicker])
                continue
            d_r, receiver, _ = _nearest(P, team_of, g1, bx[g1], by[g1])
            tags[g0:land_at] = TAGS.index("AIR_" + _TEAM[kicker])
            if d_r <= he.CONTROL_R:
                tags[land_at:g1] = TAGS.index("LANDED_" + _TEAM[receiver])
                landing[land_at] = True
            else:
                tags[land_at:g1] = TAGS.index("HOLD")
            stats["AIR->LANDED"] += 1
            continue

        stats["HOLD"] += 1

    tags[~graded] = TAGS.index("HOLD")
    if verbose:
        counts = {t: int((tags[graded] == i).sum()) for i, t in enumerate(TAGS)}
        print("  gaps on alive play: {gaps}  CARRY {CARRY}  AIR {AIR} (landed {AIR->LANDED})  "
              "STATIC_DEAD {STATIC_DEAD}  HOLD {HOLD}".format(**stats))
        print("  internal tag frames (graded): " +
              "  ".join("{} {}".format(t, c) for t, c in counts.items() if c))
    return tags, landing, stats


def tags_to_evidence(tags, landing, dead_stage1):
    """Internal tags -> (llr_home, llr_away, dead). Only H/A/D come out of this."""
    n = len(tags)
    llr_h = np.zeros(n); llr_a = np.zeros(n)
    name = np.array(TAGS)[tags]
    llr_h[name == "CARRY_H"] += CARRY_LLR;   llr_a[name == "CARRY_A"] += CARRY_LLR
    llr_h[name == "AIR_H"] += AIR_LLR;       llr_a[name == "AIR_A"] += AIR_LLR
    llr_h[name == "LANDED_H"] += LANDED_LLR; llr_a[name == "LANDED_A"] += LANDED_LLR
    llr_h[landing & (name == "LANDED_H")] += LANDING_BONUS
    llr_a[landing & (name == "LANDED_A")] += LANDING_BONUS
    dead = dead_stage1 | (name == "DEAD_UNSEEN")
    return llr_h, llr_a, dead


def resolve_duel_by_pass_destination(trk, d_home, d_away, dynamic_team=None, min_players=3, max_flight_frames=40, verbose=True):
    """If a collision of MORE players happens (>= min_players total from both teams < 2.5m from the ball),
    look ahead to see which team cleanly receives the ball.
    The collision contact is attributed to the receiving team! Other scenarios (1v1 duels, etc.) are untouched."""
    bs = trk["ball_s"]
    n = len(bs)

    P = trk["P"]
    dx = P[..., 0] - trk["ball_x"][:, None]
    dy = P[..., 1] - trk["ball_y"][:, None]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)

    if dynamic_team is not None:
        home = (dynamic_team == 0)
        close_home = ((d < 2.5) & home).sum(axis=1)
        close_away = ((d < 2.5) & ~home).sum(axis=1)
    else:
        home = trk["slot_team"] == 0
        close_home = (d[:, home] < 2.5).sum(axis=1)
        close_away = (d[:, ~home] < 2.5).sum(axis=1)
    close_total = close_home + close_away

    # ONLY when collision of MORE players happens (>= min_players, with both teams present)
    multi_collision = (
        (close_home >= 1) &
        (close_away >= 1) &
        (close_total >= min_players) &
        ~np.isnan(trk["ball_x"]) &
        (bs < 15.0)
    )
    duel_frames = np.where(multi_collision)[0]

    episodes = []
    if len(duel_frames) > 0:
        cur_start = duel_frames[0]
        cur_end = duel_frames[0]
        for f in duel_frames[1:]:
            if f <= cur_end + 10:
                cur_end = f
            else:
                episodes.append((cur_start, cur_end))
                cur_start = f
                cur_end = f
        episodes.append((cur_start, cur_end))

    d_home_mod = np.array(d_home, copy=True)
    d_away_mod = np.array(d_away, copy=True)
    resolved = 0

    for ep_start, ep_end in episodes:
        recv_window = range(ep_end + 5, min(n, ep_end + max_flight_frames))
        clean_receivers = []
        for f in recv_window:
            dh, da = d_home[f], d_away[f]
            if dh < 1.8 and da > 2.8:
                clean_receivers.append((f, 0, dh))
                break
            elif da < 1.8 and dh > 2.8:
                clean_receivers.append((f, 1, da))
                break

        if clean_receivers:
            recv_f, recv_team, _ = clean_receivers[0]
            nom_team = 0 if d_home[ep_end] < d_away[ep_end] else 1
            if nom_team != recv_team:
                end_f = recv_f + 1 if (recv_f - ep_end) <= 25 else ep_end + 1
                if recv_team == 0:
                    d_home_mod[ep_start:end_f] = np.minimum(d_home_mod[ep_start:end_f], 0.8)
                    d_away_mod[ep_start:end_f] = np.maximum(d_away_mod[ep_start:end_f], 2.8)
                else:
                    d_away_mod[ep_start:end_f] = np.minimum(d_away_mod[ep_start:end_f], 0.8)
                    d_home_mod[ep_start:end_f] = np.maximum(d_home_mod[ep_start:end_f], 2.8)
                resolved += 1

    if verbose:
        print(f"  Multi-Player Collision Resolution (>= {min_players} players): resolved {resolved} multi-player collisions via receiver flow.")
    return d_home_mod, d_away_mod


def attribute_ball_in_motion_to_last_kicker(trk, d_home, d_away, dead, few_frames=12, min_speed=3.0, boost=0.2, verbose=True):
    """If no one holds the ball for a few frames while the ball is still in motion,
    possession remains with the team that kicked last."""
    bs = trk["ball_s"]
    n = len(bs)
    llr_k_h = np.zeros(n)
    llr_k_a = np.zeros(n)

    kicks_h = 0
    kicks_a = 0

    for f in range(1, n - few_frames):
        if dead[f]:
            continue
        # Check if Home kicked the ball into motion
        if d_home[f-1] < 1.8 and d_away[f-1] > 2.2 and bs[f] > min_speed and bs[f-1] < min_speed:
            kicks_h += 1
            for k in range(f, f + few_frames):
                if min(d_home[k], d_away[k]) > 1.8 and not dead[k]:
                    llr_k_h[k] = boost * he.CONTROL_LLR
        # Check if Away kicked the ball into motion
        elif d_away[f-1] < 1.8 and d_home[f-1] > 2.2 and bs[f] > min_speed and bs[f-1] < min_speed:
            kicks_a += 1
            for k in range(f, f + few_frames):
                if min(d_home[k], d_away[k]) > 1.8 and not dead[k]:
                    llr_k_a[k] = boost * he.CONTROL_LLR

    if verbose:
        print(f"  Ball-In-Motion Kick Retention: registered {kicks_h} Home kicks, {kicks_a} Away kicks into motion.")
    return llr_k_h, llr_k_a


def resolve_quick_pass_transits(trk, d_home, d_away, dead, max_flight_frames=40, verbose=True):
    """When a quick pass is completed between teammates of the same team,
    attribute all intermediate flight frames to that team, preventing false switches
    from opponents standing along the pass trajectory."""
    bs_arr = trk["ball_s"]
    bx, by = trk["ball_x"], trk["ball_y"]
    n = len(bs_arr)
    d_h_mod = np.array(d_home, copy=True)
    d_a_mod = np.array(d_away, copy=True)

    quick_h = 0
    quick_a = 0

    for f in range(1, n - 75):
        if dead[f]:
            continue
        # Home pass / knock-on (up to 75 frames when opponent never comes within 2.0m)
        if d_home[f-1] < 1.8 and d_away[f-1] > 2.2 and bs_arr[f] > 4.5 and (bs_arr[f-1] < 4.5 or d_home[f-1] <= 1.35):
            for k in range(f + 3, min(n, f + 75)):
                if dead[k]:
                    break
                opp_lim = 1.8 if (k - f) < max_flight_frames else 2.0
                if d_home[k] < 1.8 and d_away[k] > 2.2:
                    quick_h += 1
                    for t in range(f, k):
                        if min(d_home[t], d_away[t]) > 1.2:
                            d_h_mod[t] = min(d_h_mod[t], 1.2)
                            d_a_mod[t] = max(d_a_mod[t], 3.0)
                    break
                elif d_away[k] < opp_lim:
                    break
        # Away pass / knock-on (up to 75 frames when opponent never comes within 2.0m)
        elif d_away[f-1] < 1.8 and d_home[f-1] > 2.2 and bs_arr[f] > 4.5 and (bs_arr[f-1] < 4.5 or d_away[f-1] <= 1.35):
            for k in range(f + 3, min(n, f + 75)):
                if dead[k]:
                    break
                opp_lim = 1.8 if (k - f) < max_flight_frames else 2.0
                if d_away[k] < 1.8 and d_home[k] > 2.2:
                    quick_a += 1
                    for t in range(f, k):
                        if min(d_home[t], d_away[t]) > 1.2:
                            d_a_mod[t] = min(d_a_mod[t], 1.2)
                            d_h_mod[t] = max(d_h_mod[t], 3.0)
                    break
                elif d_home[k] < opp_lim:
                    break

        # One-touch / quick-redirect pass between teammates along an undeflected trajectory
        for is_home, d_own, d_opp, d_own_mod, d_opp_mod in (
            (True, d_home, d_away, d_h_mod, d_a_mod),
            (False, d_away, d_home, d_a_mod, d_h_mod),
        ):
            if d_own[f-1] <= 1.25 and d_opp[f-1] > d_own[f-1] + 0.35 and bs_arr[f] >= 5.0:
                for k in range(f + 2, min(n, f + 45)):
                    if dead[k]:
                        break
                    if d_own[k] <= 1.35 and d_opp[k] >= 2.0:
                        vec_tot = np.array([bx[k] - bx[f-1], by[k] - by[f-1]], dtype=np.float64)
                        norm_tot = float(np.linalg.norm(vec_tot))
                        ok = True
                        if norm_tot > 1.0:
                            u_tot = vec_tot / norm_tot
                            for t in range(f, k):
                                if d_opp[t] < 1.5:
                                    if not (bs_arr[t] >= 6.0):
                                        ok = False
                                        break
                                    vt = np.array([bx[min(n-1, t+1)] - bx[max(0, t-1)], by[min(n-1, t+1)] - by[max(0, t-1)]], dtype=np.float64)
                                    nv = float(np.linalg.norm(vt))
                                    if nv > 0.1 and float(np.dot(vt / nv, u_tot)) < 0.70:
                                        ok = False
                                        break
                        if ok:
                            if is_home:
                                quick_h += 1
                            else:
                                quick_a += 1
                            for t in range(f - 1, k + 1):
                                d_own_mod[t] = min(d_own_mod[t], 0.8)
                                d_opp_mod[t] = max(d_opp_mod[t], 3.2)
                        break
                    elif d_opp[k] <= 1.0 and bs_arr[k] < 6.0:
                        break
                break

    if verbose:
        print(f"  Quick Pass Transit Resolution: resolved {quick_h} Home, {quick_a} Away quick passes.")
    return d_h_mod, d_a_mod


def resolve_goalkeeper_inbound_outbound_passes(trk, d_home, d_away, dead, max_look_frames=75, game="BAR-ATH", verbose=True):
    """
    Goalkeeper Inbound & Outbound Pass & Homography Possession Flow.
    
    Uses pitch homography coordinates of the Goalkeeper Area (Penalty Box / Goal Area):
    - Left Box (Home def in H1, Away def in H2): x <= 18.0, 12.0 <= y <= 56.0
    - Right Box (Away def in H1, Home def in H2): x >= 87.3, 12.0 <= y <= 56.0
    
    1. Inbound pass/shot (e.g. H -> H -> A -> GK):
       Attacking team kicks/passes ball towards the GK box. The inbound frames
       from kick until GK establishes control are attributed to the attacking team.
    2. Goalkeeper Possession & Outfield Egress:
       When ball enters GK homography and outfield attacking players egress/retreat
       out of the box without the ball (or d_att > 3.5m and ball slows down),
       the defending goalkeeper has secured the ball.
    3. Outbound pass (GK -> Teammate):
       When GK passes out to a teammate, the transit from GK release to reception
       belongs to the goalkeeper's team.
    """
    P = trk["P"]
    n = len(P)
    bx, by = trk["ball_x"], trk["ball_y"]
    bs = trk["ball_s"]

    # Define Goalkeeper Homography Box geofence (allow -1.5m behind goal line for keeper saves/holds)
    in_lbox = (bx >= -1.5) & (bx <= 16.5) & (by >= 13.84) & (by <= 54.16) & ~np.isnan(bx)
    in_rbox = (bx >= 88.8) & (bx <= 106.8) & (by >= 13.84) & (by <= 54.16) & ~np.isnan(bx)

    box_def_team = np.full(n, -1, dtype=int)
    is_in_gk_box = np.zeros(n, dtype=bool)

    # Half 1 / Half 2 boundaries and defending sides inferred dynamically from tracking
    from games import half_video_blocks
    blocks = {h: (s, e) for h, s, e in half_video_blocks(game)}
    home_mask = (trk["slot_team"] == 0)
    for _h, (hs, he_f) in blocks.items():
        med_h = float(np.nanmedian(P[hs:he_f+1, home_mask, 0]))
        med_a = float(np.nanmedian(P[hs:he_f+1, ~home_mask, 0]))
        left_team = 0 if med_h < med_a else 1
        right_team = 1 - left_team
        for f in range(hs, min(n, he_f + 1)):
            if in_lbox[f]:
                is_in_gk_box[f] = True
                box_def_team[f] = left_team
            elif in_rbox[f]:
                is_in_gk_box[f] = True
                box_def_team[f] = right_team

    # Group contiguous box frames into episodes (allowing small tracking flickers <= 5 frames)
    box_frames = np.where(is_in_gk_box)[0]
    episodes = []
    if len(box_frames) > 0:
        cur_s = box_frames[0]
        cur_e = box_frames[0]
        cur_side = box_def_team[cur_s]
        for f in box_frames[1:]:
            if f <= cur_e + 5 and box_def_team[f] == cur_side:
                cur_e = f
            else:
                episodes.append((cur_s, cur_e, cur_side))
                cur_s = f
                cur_e = f
                cur_side = box_def_team[f]
        episodes.append((cur_s, cur_e, cur_side))

    d_h_mod = np.array(d_home, copy=True)
    d_a_mod = np.array(d_away, copy=True)
    gk_mask = np.zeros(n, dtype=bool)
    inbound_resolved = 0
    hold_resolved = 0
    outbound_resolved = 0

    # Raw player-ball distances so upstream transit overrides (e.g. quick_pass_transits)
    # do not mask who actually kicked/diverted the ball into the GK box.
    d_raw = np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])
    d_raw = np.where(np.isnan(d_raw), np.inf, d_raw)
    home_slots = (trk["slot_team"] == 0)
    dh_raw = d_raw[:, home_slots].min(axis=1)
    da_raw = d_raw[:, ~home_slots].min(axis=1)

    for ep_s, ep_e, def_team in episodes:
        att_team = 1 - def_team

        # 1. DETECT GK CLAIM / POSSESSION & OUTFIELD ATTACKER EGRESS
        claim_f = None
        for f in range(ep_s, ep_e + 1):
            dh, da = d_home[f], d_away[f]
            d_def = dh if def_team == 0 else da
            d_att = da if def_team == 0 else dh

            # Condition: attacking players retreat/egress (d_att > 3.5m) while defending team is close
            # or defending GK/defender secures ball (d_def < 1.5m and ball slow)
            if (d_att > 3.5 and d_def <= 1.5) or (d_att > 4.0 and bs[f] < 2.5 and d_def < 2.5):
                claim_f = f
                break

        # 2. INBOUND: Look backwards from just before GK claim (or box end) to find who
        #    last kicked or collided with the ball into the box. Only attribute from that
        #    collision/kick frame (kick_f) to GK, never before it.
        passer_team = None
        kick_f = None
        search_start = (claim_f - 1) if claim_f is not None else ep_e
        search_end = max(0, ep_s - max_look_frames)
        for f in range(search_start, search_end - 1, -1):
            if dead[f]:
                break
            dh, da = dh_raw[f], da_raw[f]
            d_min = min(dh, da)
            # True close collision/touch (even if opponent is nearby or inside box before claim)
            if d_min <= 0.65 and (not np.isfinite(bs[f]) or bs[f] < 15.0):
                best_f = f
                best_d = d_min
                k = f - 1
                while k >= max(search_end, f - 10) and not dead[k]:
                    dk = min(dh_raw[k], da_raw[k])
                    if dk > 0.85:
                        break
                    if dk < best_d:
                        best_d = dk
                        best_f = k
                    k -= 1
                if abs(dh_raw[best_f] - da_raw[best_f]) >= 0.3:
                    kick_f = best_f
                    passer_team = 0 if dh_raw[best_f] < da_raw[best_f] else 1
                    break
            if f >= ep_s:
                continue
            # Skip fast-flying ball passing >1.8m from a non-touching player
            if d_min > 1.8 and np.isfinite(bs[f]) and bs[f] > 10.0:
                continue
            if dh < 3.0 and dh < da - 1.0:
                passer_team = 0
                kick_f = f
                break
            elif da < 3.0 and da < dh - 1.0:
                passer_team = 1
                kick_f = f
                break

        # Attribute inbound pass/backpass transit to the passing team (A->GK = A, H->GK = H).
        # If an attacking inbound pass disappears (NaN) inside the GK box before any GK claim,
        # hold the attacking team's inbound possession through the disappearance gap (up to 50 frames).
        if passer_team is not None and kick_f is not None:
            transit_end = claim_f if claim_f is not None else ep_e + 1
            nan_hold = False
            if claim_f is None and passer_team == att_team and ep_e + 1 < n and np.isnan(bx[ep_e + 1]):
                k = ep_e + 1
                while k < min(n, ep_e + 50) and np.isnan(bx[k]) and not dead[k]:
                    k += 1
                transit_end = k
                nan_hold = True
            d_win = 0.4 if nan_hold else 1.2
            d_lose = 3.5 if nan_hold else 3.0
            for f in range(kick_f, transit_end):
                if not dead[f]:
                    if passer_team == 0:
                        d_h_mod[f] = d_win if not np.isfinite(d_h_mod[f]) else min(d_h_mod[f], d_win)
                        d_a_mod[f] = d_lose if not np.isfinite(d_a_mod[f]) else max(d_a_mod[f], d_lose)
                    else:
                        d_a_mod[f] = d_win if not np.isfinite(d_a_mod[f]) else min(d_a_mod[f], d_win)
                        d_h_mod[f] = d_lose if not np.isfinite(d_h_mod[f]) else max(d_h_mod[f], d_lose)
                    gk_mask[f] = True
            inbound_resolved += 1

        # Goalkeeper hold period: defending team in possession
        if claim_f is not None:
            for f in range(claim_f, ep_e + 1):
                if not dead[f]:
                    if def_team == 0:
                        d_h_mod[f] = min(d_h_mod[f], 1.2)
                        d_a_mod[f] = max(d_a_mod[f], 3.0)
                    else:
                        d_a_mod[f] = min(d_a_mod[f], 1.2)
                        d_h_mod[f] = max(d_h_mod[f], 3.0)
                    gk_mask[f] = True
            hold_resolved += 1

        # 3. OUTBOUND: Look forward from ep_e to see who receives the ball
        recv_team = None
        recv_f = None
        for f in range(ep_e + 1, min(n, ep_e + max_look_frames + 15)):
            if dead[f]:
                break
            dh, da = d_home[f], d_away[f]
            if dh < 1.8 and da > 2.5:
                recv_team = 0
                recv_f = f
                break
            elif da < 1.8 and dh > 2.5:
                recv_team = 1
                recv_f = f
                break

        if recv_team is not None and recv_f is not None:
            if recv_team == def_team:
                for f in range(ep_e, recv_f):
                    if not dead[f] and min(d_home[f], d_away[f]) > 1.2:
                        if def_team == 0:
                            d_h_mod[f] = min(d_h_mod[f], 1.2)
                            d_a_mod[f] = max(d_a_mod[f], 3.0)
                        else:
                            d_a_mod[f] = min(d_a_mod[f], 1.2)
                            d_h_mod[f] = max(d_h_mod[f], 3.0)
                        gk_mask[f] = True
                outbound_resolved += 1

    if verbose:
        print(f"  Goalkeeper Homography Pass Flow: resolved {inbound_resolved} inbound passes TO GK, {hold_resolved} GK holds, {outbound_resolved} outbound passes FROM GK.")
    return d_h_mod, d_a_mod, gk_mask


def resolve_noncolliding_ground_transits(labels, trk, d_home, d_away, dead, touch_r=0.70, lookahead=45, verbose=True):
    """
    Non-Colliding Ground Transit Protection:
    When a ball is kicked along the pitch and rolls near an opponent player without
    physical touch contact (distance reaches a closest-approach minimum > touch_r and separates),
    possession remains with the kicking team rather than prematurely flipping to the opponent.
    Contested multi-player tackles/collisions (collide mode) are excluded and handled by duel logic.
    """
    mod_labels = np.array(labels, copy=True)
    bs_arr = trk["ball_s"]
    n = len(labels)
    delayed_count = 0

    for f in range(1, n - lookahead):
        prev = mod_labels[f - 1]
        curr = mod_labels[f]

        if (prev == "H" and curr == "A") or (prev == "A" and curr == "H"):
            kicking_team = prev
            receiving_team = curr
            d_kick = d_home if kicking_team == "H" else d_away
            d_recv = d_away if kicking_team == "H" else d_home

            # 1. Kicked recently by kicking_team into motion
            had_kick = np.any(d_kick[max(0, f - 12):f] <= touch_r) and (bs_arr[f] >= 2.0)
            if not had_kick:
                continue

            # 2. Receiving team does NOT have physical touch at transition
            if d_recv[f] <= touch_r:
                continue

            # 3. Not in collide mode (kicking team is not tackling in the duel)
            if d_kick[f] <= 1.4:
                continue

            # 4. Check if ball passes near receiving player without touch (closest approach > touch_r and separates)
            d_window = d_recv[f:min(n, f + 15)]
            min_idx = np.argmin(d_window)
            min_dist = d_window[min_idx]

            is_pass_by = (min_dist > touch_r) and (d_window[-1] > min_dist + 0.3)
            if is_pass_by:
                touch_frame = None
                for k in range(f + min_idx, min(n, f + lookahead)):
                    if dead[k] or mod_labels[k] != receiving_team:
                        break
                    # Receiving team executes kick:
                    if bs_arr[k] >= 10.0 and d_recv[k] <= 2.0:
                        touch_frame = k
                        break
                    # Or establishes settled close control:
                    if bs_arr[k] <= 2.5 and d_recv[k] <= 0.50:
                        touch_frame = k
                        break
                if touch_frame is not None and touch_frame > f:
                    for t in range(f, touch_frame):
                        if not dead[t]:
                            mod_labels[t] = kicking_team
                    delayed_count += 1

    if verbose:
        print(f"  Non-Colliding Ground Transit Protection: resolved {delayed_count} premature opponent proximity switches.")
    return mod_labels


def resolve_one_touch_flicks(labels, trk, dynamic_team, touch_r=1.2, fast=15.0, leave=3.0,
                             win=10, look=40, ctrl_r=2.0, verbose=True):
    """
    One-Touch Flick != Possession:
    When the new team's first contact is a single flick/deflection (ball leaves that
    player fast, >= `fast` m/s for `win` frames, and ends >= `leave` m from him), the
    switch is delayed until ANOTHER player of the new team is next near the ball
    (<= ctrl_r), i.e. when that team actually controls it.
    """
    out = np.array(labels, copy=True)
    bs_arr = np.nan_to_num(trk["ball_s"])
    P = trk["P"]
    D = np.hypot(P[..., 0] - trk["ball_x"][:, None], P[..., 1] - trk["ball_y"][:, None])
    D = np.where(np.isnan(D), np.inf, D)
    n = len(out)
    count = 0

    for f in range(1, n - look - win - 6):
        p, c = out[f - 1], out[f]
        if p not in ("H", "A") or c not in ("H", "A") or p == c:
            continue
        team = 1 if c == "A" else 0

        # 1. new-team player touches the ball around the switch
        best = None
        for t in range(f, f + 6):
            dd = np.where(dynamic_team[t] == team, D[t], np.inf)
            s = int(np.argmin(dd))
            if dd[s] <= touch_r and (best is None or dd[s] < best[2]):
                best = (t, s, dd[s])
        if best is None:
            continue
        t, s_touch, _ = best

        # 2. ball leaves that toucher fast (flick / one-touch), no control
        if not all(bs_arr[k] >= fast for k in range(t + 1, t + 1 + win)):
            continue
        if D[t + win, s_touch] < leave:
            continue

        # 3. switch only when ANOTHER new-team player is next near the ball
        k_ctrl = None
        for k in range(t + 3, min(n, f + look)):
            if out[k] != c:
                break
            dd = np.where(dynamic_team[k] == team, D[k], np.inf)
            dd[s_touch] = np.inf
            if dd.min() <= ctrl_r:
                k_ctrl = k
                break
        if k_ctrl is None or k_ctrl <= f:
            continue
        out[f:k_ctrl] = p
        count += 1

    if verbose:
        print(f"  One-Touch Flick Resolution: delayed {count} switches to the new team's controlled receipt.")
    return out


def resolve_gk_restart_passes(labels, dead, trk, dynamic_team, gk_slots=None, strike_r=1.0,
                              strike_v=4.0, recv_r=1.5, min_travel=5.0, look=100, cap=75, verbose=True):
    """
    Goalkeeper Restart Pass -> Own Teammate:
    A dead stretch where the keeper strikes the ball (<= strike_r, ball > strike_v m/s) to a
    teammate who receives it (<= recv_r) with no opponent near, is given to the keeper's team
    from the frame the ball is closer to the receiver than to the keeper (the equal-distance
    point of the pass) until the dead stretch ends (at most `cap` frames after receipt).
    """
    if gk_slots is None:
        gk_slots = trk.get("gk_slots", ())
    out = np.array(labels, copy=True)
    dead = np.array(dead, copy=True)
    bs_arr = np.nan_to_num(trk["ball_s"])
    bx, by = trk["ball_x"], trk["ball_y"]
    P = trk["P"]
    D = np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])
    D = np.where(np.isnan(D), np.inf, D)
    n = len(out)
    count = 0

    for g in gk_slots:
        f = 0
        while f < n - look:
            if not (out[f] == "D" and D[f, g] <= strike_r and bs_arr[f + 3] > strike_v):
                f += 1
                continue
            team = dynamic_team[f, g]
            lab = "H" if team == 0 else "A"
            mid = recv = None
            for t in range(f + 1, f + look):
                if out[t] != "D":
                    break
                mates = np.where(dynamic_team[t] == team, D[t], np.inf)
                mates[g] = np.inf
                opp = np.where(dynamic_team[t] != team, D[t], np.inf)
                if opp.min() <= recv_r:
                    break                                   # opponent intercepts -> leave it
                if mid is None and mates.min() < D[t, g]:
                    mid = t
                if mates.min() <= recv_r:
                    recv = t
                    break
            travel = np.hypot(bx[recv] - bx[f], by[recv] - by[f]) if recv is not None else 0.0
            if mid is not None and recv is not None and mid > f and travel >= min_travel:
                e = mid
                while e < n and out[e] == "D" and e < recv + cap:
                    e += 1
                out[mid:e] = lab
                dead[mid:e] = False
                count += 1
                f = e
            else:
                f += 1

    if verbose:
        print(f"  Goalkeeper Restart Pass: gave {count} keeper-to-teammate restart passes to the keeper's team.")
    return out, dead


def resolve_kickoff_hold(labels, dead, score, d_home, d_away, graded, game="BAR-ATH", data_dir=None,
                         opp_control=0.3, max_hold=750, verbose=True):
    """Kickoff: the team that kicks off keeps the ball until the opponent clearly controls it.

    Kicker = team with the nearest player (within CONTROL_R) on the first frame inside a
    Smart Tagging 'kick off' window where the ball is seen. From that frame the kicker's
    label is held until the opponent's control score reaches `opp_control`, both teams are
    within CONTROL_R of the ball (a challenge), the ball goes dead, or `max_hold` frames pass.
    """
    out = labels.copy()
    restarts, _ = _db.smart_tag_intervals(game, data_dir)
    n = len(out)
    applied = 0
    for (s0, s1, _team, typ) in restarts:
        if "kick off" not in str(typ).lower():
            continue
        kick = None
        for f in range(max(0, s0), min(n, s1 + 1)):
            if dead[f] or not graded[f]:
                continue
            dh, da = d_home[f], d_away[f]
            if np.isnan(dh) and np.isnan(da):
                continue
            dh = np.inf if np.isnan(dh) else dh
            da = np.inf if np.isnan(da) else da
            if min(dh, da) < he.CONTROL_R:
                kick = (f, "H" if dh < da else "A")
                break
        if kick is None:
            continue
        f0, lab = kick
        sign = 1.0 if lab == "H" else -1.0
        f = f0
        changed = 0
        while f < n and f < f0 + max_hold:
            if dead[f] or not graded[f]:
                break
            if -sign * score[f] >= opp_control:     # opponent clearly in control
                break
            if d_home[f] < he.CONTROL_R and d_away[f] < he.CONTROL_R and f >= f0 + 10:
                break                                # opponent challenges -> normal model decides
            if out[f] != lab:
                out[f] = lab
                changed += 1
            f += 1
        if changed:
            applied += 1
            if verbose:
                print(f"  Kickoff Hold: kick at {f0} by {lab}, held {lab} to {f - 1} ({changed} frames relabelled).")
    if verbose:
        print(f"  Kickoff Hold: applied to {applied} kickoff(s).")
    return out


def resolve_same_team_pass_midpoints(labels, trk, d_home, d_away, dead,
                                     touch_r=1.8, leave_r=2.0, lookahead=60,
                                     verbose=True):
    """On an H<->A turnover (outside GK boxes), if the new team makes an immediate
    same-team pass (first touch t1 -> ball leaves -> teammate receipt t2),
    delay the possession switch to the halfway point (t1 + t2) // 2."""
    out = np.array(labels, copy=True)
    P = trk["P"]
    bx, by = trk["ball_x"], trk["ball_y"]
    n = len(out)
    d_raw = np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])
    d_raw = np.where(np.isnan(d_raw), np.inf, d_raw)
    home_slots = (trk["slot_team"] == 0)
    dh_raw = d_raw[:, home_slots].min(axis=1)
    da_raw = d_raw[:, ~home_slots].min(axis=1)
    in_gk = (((bx >= -1.5) & (bx <= 16.5)) | ((bx >= 88.8) & (bx <= 106.8))) & (by >= 13.84) & (by <= 54.16) & np.isfinite(bx)
    adjusted = 0

    for f in range(1, n - lookahead):
        prev_t = out[f - 1]
        new_t = out[f]
        if prev_t not in ("H", "A") or new_t not in ("H", "A") or prev_t == new_t:
            continue
        if np.any(in_gk[max(0, f - 10):min(n, f + 25)]):
            continue
        dn = dh_raw if new_t == "H" else da_raw
        t1 = None
        for t in range(f, min(n, f + lookahead)):
            if out[t] != new_t or dead[t]:
                break
            if dn[t] <= touch_r:
                t1 = t
                break
        if t1 is None:
            continue
        t_leave = None
        t2 = None
        for t in range(t1 + 1, min(n, t1 + 45)):
            if out[t] != new_t or dead[t]:
                break
            if t_leave is None and dn[t] > leave_r:
                t_leave = t
            elif t_leave is not None and dn[t] <= touch_r:
                t2 = t
                break
        if t2 is not None:
            mid = (t1 + t2) // 2
            if mid > f:
                out[f:mid] = prev_t
                adjusted += 1

    if verbose:
        print(f"  Same-Team Pass Midpoint: adjusted {adjusted} turnover(s) to pass midpoint.")
    return out


def decode_missing(trk, dead_stage1, game="BAR-ATH", data_dir=None, dynamic_team=None, pass_destination_duels=False, quick_pass_transits=False, gk_pass_flow=False, kicker_in_motion=False, ground_transit_touch=False, one_touch_flick=False, gk_restart_pass=False, verbose=True, ball_override=None, kickoff_hold=False, same_team_pass_midpoint=False):
    """Stage-1 dead mask + missing-ball tags -> labels in {H, A, D, X}."""
    graded = _db.graded_mask(game)
    bx, by, speed, trusted = ball_speed.build_speed_and_trust(game, data_dir, verbose=verbose, ball_override=ball_override)
    tags, landing, stats = tag_gaps(trk, dead_stage1, bx, by, speed, trusted, game, verbose)
    llr_h, llr_a, dead = tags_to_evidence(tags, landing, dead_stage1)
    dead &= graded
    alive = graded & ~dead

    d_home, d_away = he.evidence_from_tracking(trk, dynamic_team=dynamic_team)
    if pass_destination_duels:
        d_home, d_away = resolve_duel_by_pass_destination(trk, d_home, d_away, dynamic_team=dynamic_team, verbose=verbose)
    if quick_pass_transits:
        d_home, d_away = resolve_quick_pass_transits(trk, d_home, d_away, dead, verbose=verbose)
    gk_mask = np.zeros(len(dead), dtype=bool)
    if gk_pass_flow:
        d_home, d_away, gk_mask = resolve_goalkeeper_inbound_outbound_passes(trk, d_home, d_away, dead, game=game, verbose=verbose)
    score = he.control_score(d_home, d_away, trk["ball_s"])
    score = np.where(trusted | gk_mask, score, 0.0)          # untrusted ball = missing ball (except GK box holds)
    priors = he.restart_priors(alive, game, data_dir)
    logB = he.emission_loglik(score, dead, graded, priors)
    logB[alive, he.S_H] += llr_h[alive]
    logB[alive, he.S_A] += llr_a[alive]

    if kicker_in_motion:
        llr_k_h, llr_k_a = attribute_ball_in_motion_to_last_kicker(trk, d_home, d_away, dead, verbose=verbose)
        logB[alive, he.S_H] += llr_k_h[alive]
        logB[alive, he.S_A] += llr_k_a[alive]

    assert not np.isnan(logB).any(), "NaN leaked into the Viterbi log-likelihoods"
    assert np.isfinite(logB[alive][:, :2]).all()

    path = he.viterbi(logB, he.transition_logmatrix())
    labels = he.STATE_LABELS[path].astype("<U1")
    labels[~graded] = he.UNGRADED
    assert set(np.unique(labels[graded])) <= {"H", "A", "D"}, "graded label outside H/A/D"
    assert np.all(labels[dead] == "D")

    if ground_transit_touch:
        labels = resolve_noncolliding_ground_transits(labels, trk, d_home, d_away, dead, verbose=verbose)

    if one_touch_flick:
        dyn_t = dynamic_team if dynamic_team is not None else np.tile(trk["slot_team"], (len(labels), 1))
        labels = resolve_one_touch_flicks(labels, trk, dyn_t, verbose=verbose)
        labels[dead] = "D"
        labels[~graded] = he.UNGRADED

    if gk_restart_pass:
        dyn_t = dynamic_team if dynamic_team is not None else np.tile(trk["slot_team"], (len(labels), 1))
        labels, dead = resolve_gk_restart_passes(labels, dead, trk, dyn_t, verbose=verbose)
        labels[~graded] = he.UNGRADED

    if kickoff_hold:
        labels = resolve_kickoff_hold(labels, dead, score, d_home, d_away, graded, game, data_dir, verbose=verbose)
        labels[dead] = "D"
        labels[~graded] = he.UNGRADED

    if same_team_pass_midpoint:
        labels = resolve_same_team_pass_midpoints(labels, trk, d_home, d_away, dead, verbose=verbose)
        labels[dead] = "D"
        labels[~graded] = he.UNGRADED

    return labels, dead, tags, stats
