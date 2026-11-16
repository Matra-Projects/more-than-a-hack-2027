"""
Orchestrator - strict two-stage pipeline -> 199,319-row submission CSV.

  Stage 1  module_1 (deadball engine)  owns D, runs first, is final
  Stage 2  possession physics evidence on Stage-1-alive frames only
           -> hmm_engine Viterbi decode (H/A), dead frames forced to D

    python src/build_submission.py            # -> evaluation/results/team_engine_hmm.csv
    python src/build_submission.py --out my.csv

Then ALWAYS:
    python evaluation/validate_submission.py evaluation/results/team_engine_hmm.csv
    python evaluation/run_evaluation.py

Tactical source only. A Python audit hook records every file opened under the data
directory and aborts the run if anything under data/broadcast/ is touched.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
SRC_DIR = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC_DIR)

from games import GAME, DATA_DIR  # noqa: E402

# ----------------------------------------------------------------------------
# Data-access audit: tactical only
# ----------------------------------------------------------------------------
_OPENED = set()


def _install_audit(data_dir):
    data_root = os.path.realpath(data_dir)
    broadcast_root = os.path.join(data_root, "broadcast")

    def hook(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(args[0])
        real = os.path.realpath(path)
        if real.startswith(broadcast_root):
            raise PermissionError("broadcast data is off-limits: " + path)
        if real.startswith(data_root):
            _OPENED.add(os.path.relpath(real, data_root))

    sys.addaudithook(hook)


def apply_passflow_player_reattribution(trk, game=None, verbose=True):
    """
    Forward and Backward Pass-Flow Context Analyzer.
    
    Rather than naively deleting tracklets or hardcoding slot assignments, this inspects
    the surrounding pass and possession flow when an anomalous tracklet occurs near a goal:
    - If a tracklet assigned to Team B is deep in Team A's defensive third/goalmouth,
      check who had the ball BEFORE (backward) and who has the ball AFTER (forward).
    - If Team A passed to this player/location and Team A receives the ball after,
      this player is acting for Team A (e.g. goalkeeper / defender misidentified in ATD).
    - Dynamically attributes that touch sequence to Team A without touching raw files.
    """
    import games

    P = trk["P"]
    bx, by = trk["ball_x"], trk["ball_y"]
    bs = trk["ball_s"]
    n = len(bx)
    slot_team = trk["slot_team"]
    gk_slots = set(trk.get("gk_slots", ()))
    PITCH_L = 105.3

    try:
        half_blocks = [(h, s, min(e, n - 1)) for h, s, e, _ in games.half_video_blocks(game or GAME)]
    except Exception:
        half_blocks = [(1, 0, n // 2 - 1), (2, n // 2, n - 1)]

    half_sides = []
    home_mask_slots = (slot_team == 0)
    for _h, hs, he_f in half_blocks:
        if hs > he_f:
            continue
        med_h = float(np.nanmedian(P[hs:he_f + 1, home_mask_slots, 0]))
        med_a = float(np.nanmedian(P[hs:he_f + 1, ~home_mask_slots, 0]))
        left_def_team = 0 if med_h < med_a else 1
        right_def_team = 1 - left_def_team
        half_sides.append((hs, he_f, left_def_team, right_def_team))

    dx = P[..., 0] - bx[:, None]
    dy = P[..., 1] - by[:, None]
    d = np.hypot(dx, dy)
    d = np.where(np.isnan(d), np.inf, d)

    min_s = np.argmin(d, axis=1)
    min_d = np.min(d, axis=1)

    is_touch = (min_d < 3.0) & (bs < 15.0) & ~np.isnan(bx)
    touch_frames = np.where(is_touch)[0]

    events = []
    if len(touch_frames) > 0:
        cur_slot = min_s[touch_frames[0]]
        cur_start = touch_frames[0]
        cur_end = touch_frames[0]
        for f in touch_frames[1:]:
            s = min_s[f]
            if s == cur_slot and f <= cur_end + 5:
                cur_end = f
            else:
                events.append({
                    "start": cur_start, "end": cur_end,
                    "slot": cur_slot, "team": slot_team[cur_slot],
                    "x": float(np.nanmean(P[cur_start:cur_end+1, cur_slot, 0])),
                    "y": float(np.nanmean(P[cur_start:cur_end+1, cur_slot, 1])),
                })
                cur_slot = s
                cur_start = f
                cur_end = f
        events.append({
            "start": cur_start, "end": cur_end,
            "slot": cur_slot, "team": slot_team[cur_slot],
            "x": float(np.nanmean(P[cur_start:cur_end+1, cur_slot, 0])),
            "y": float(np.nanmean(P[cur_start:cur_end+1, cur_slot, 1])),
        })

    dynamic_team = np.tile(slot_team, (n, 1))
    reattributed = 0

    for i, ev in enumerate(events):
        prev_team = events[i-1]["team"] if i > 0 else None
        next_team = events[i+1]["team"] if i < len(events) - 1 else None
        time_gap_prev = (ev["start"] - events[i-1]["end"]) if i > 0 else 999
        time_gap_next = (events[i+1]["start"] - ev["end"]) if i < len(events) - 1 else 999

        is_gk_slot = ev["slot"] in gk_slots
        if not is_gk_slot and not (20.0 <= ev["y"] <= 48.0):
            continue
        x_left_lim = 22.0 if is_gk_slot else 16.5
        x_right_lim = (PITCH_L - 22.3) if is_gk_slot else (PITCH_L - 16.5)

        left_def_team = None
        right_def_team = None
        hs_cur, he_cur = 0, n - 1
        for hs, he_f, l_def, r_def in half_sides:
            if hs <= ev["start"] <= he_f:
                left_def_team, right_def_team = l_def, r_def
                hs_cur, he_cur = hs, he_f
                break
        if left_def_team is None:
            continue

        s = ev["slot"]
        if ev["team"] != left_def_team and ev["x"] < x_left_lim:
            if prev_team == left_def_team and next_team == left_def_team and time_gap_prev < 250 and time_gap_next < 250:
                dynamic_team[ev["start"]:ev["end"]+1, s] = left_def_team
                # Extend across contiguous stay in the left goal area
                f_lo = ev["start"]
                while f_lo > hs_cur and P[f_lo - 1, s, 0] < 20.0 and 20.0 <= P[f_lo - 1, s, 1] <= 48.0:
                    f_lo -= 1
                f_hi = ev["end"]
                while f_hi < he_cur and P[f_hi + 1, s, 0] < 20.0 and 20.0 <= P[f_hi + 1, s, 1] <= 48.0:
                    f_hi += 1
                dynamic_team[f_lo:f_hi+1, s] = left_def_team
                reattributed += 1
        elif ev["team"] != right_def_team and ev["x"] > x_right_lim:
            if prev_team == right_def_team and next_team == right_def_team and time_gap_prev < 250 and time_gap_next < 250:
                dynamic_team[ev["start"]:ev["end"]+1, s] = right_def_team
                # Extend across contiguous stay in the right goal area
                f_lo = ev["start"]
                while f_lo > hs_cur and P[f_lo - 1, s, 0] > (PITCH_L - 20.3) and 20.0 <= P[f_lo - 1, s, 1] <= 48.0:
                    f_lo -= 1
                f_hi = ev["end"]
                while f_hi < he_cur and P[f_hi + 1, s, 0] > (PITCH_L - 20.3) and 20.0 <= P[f_hi + 1, s, 1] <= 48.0:
                    f_hi += 1
                dynamic_team[f_lo:f_hi+1, s] = right_def_team
                reattributed += 1

    # Enforce goalkeeper zone assignment for dynamically detected GK slots
    frame_idx = np.arange(n)
    for s in gk_slots:
        for hs, he_f, left_def_team, right_def_team in half_sides:
            in_half = (frame_idx >= hs) & (frame_idx <= he_f)
            m_left = in_half & (P[:, s, 0] < 20.0)
            m_right = in_half & (P[:, s, 0] > (PITCH_L - 20.3))
            dynamic_team[m_left, s] = left_def_team
            dynamic_team[m_right, s] = right_def_team

    if verbose:
        print(f"  Forward & Backward Pass-Flow Analysis: dynamically reattributed {reattributed} goal-area touches to the defending team.")
    return dynamic_team


def _populate_ball_from_json(trk, ball_json_path, game, data_dir, deadball_engine, only_missing=False):
    """Populate trk['ball_x']/['ball_y'] from cleaned ball JSON (projecting centroid_pixel via homography if needed)."""
    import json
    if not os.path.isfile(ball_json_path):
        return 0

    with open(ball_json_path, mode="r", encoding="utf-8") as f:
        clean_data = json.load(f)
    frames_dict = clean_data.get("frames", {})
    PITCH_L, PITCH_W = deadball_engine.PITCH_L, deadball_engine.PITCH_W
    n_trk = len(trk["ball_x"])
    H = valid = None
    loaded = 0

    for f_str, dets in frames_dict.items():
        if not dets:
            continue
        fid = int(f_str)
        if not (0 <= fid < n_trk):
            continue
        if only_missing and not np.isnan(trk["ball_x"][fid]):
            continue

        det = dets[0]
        pn = det.get("pitch_norm")
        if pn is not None:
            nx, ny = float(pn["x"]), float(pn["y"])
        elif det.get("centroid_pixel") is not None:
            if H is None:
                import load_homography as lh
                H, _, valid = lh.load(game, data_dir)
            if fid >= len(H) or not valid[fid]:
                continue
            mat = H[fid]
            if np.isnan(mat).any() or abs(np.linalg.det(mat)) <= 1e-9:
                continue
            pt = np.linalg.inv(mat) @ np.array(
                [float(det["centroid_pixel"]["x"]), float(det["centroid_pixel"]["y"]), 1.0],
                dtype=np.float64,
            )
            if abs(pt[2]) < 1e-7 or np.isnan(pt[2]):
                continue
            nx, ny = float(pt[0] / pt[2]), float(pt[1] / pt[2])
        else:
            continue

        trk["ball_x"][fid] = nx * PITCH_L
        trk["ball_y"][fid] = ny * PITCH_W
        loaded += 1
    return loaded


def _sync_fps(fps_val, deadball_engine, possession_engine, hmm_engine, mb_engine=None):
    """Synchronize FPS and derived rate constants across all pipeline engines."""
    fps_val = float(fps_val)
    if abs(fps_val - float(deadball_engine.FPS)) < 1e-6:
        return fps_val
    deadball_engine.FPS = fps_val
    possession_engine.FPS = fps_val
    hmm_engine.FPS = fps_val
    hmm_engine.P_SWITCH = 1.0 / (hmm_engine.MIN_POSSESSION_S * fps_val)
    hmm_engine.P_STAY = 1.0 - hmm_engine.P_SWITCH
    hmm_engine.SWITCH_PENALTY = float(np.log(hmm_engine.P_STAY / hmm_engine.P_SWITCH))
    hmm_engine.CHALLENGE_DWELL_S = possession_engine.DWELL / fps_val
    hmm_engine.CONTROL_LLR = hmm_engine.SWITCH_PENALTY / (hmm_engine.CHALLENGE_DWELL_S * fps_val)
    if mb_engine is not None:
        mb_engine.FPS = fps_val
    return fps_val


def track_near_player_gaps(trk, max_gap=None, near_r=3.0, verbose=True, fps=25.0):
    """
    Track and fill ball gaps ONLY when the ball disappears near a player (carry/occlusion)
    and reappears within `max_gap` frames. Leaves all other gaps NaN.
    Returns boolean mask of frames filled by near-player tracking.
    """
    fps = float(fps)
    if max_gap is None:
        max_gap = int(round(3.0 * fps))
    gap_pass = int(round(1.8 * fps))
    bx, by = trk["ball_x"], trk["ball_y"]
    P = trk["P"]
    n = len(bx)
    filled = np.zeros(n, dtype=bool)
    seen = np.where(~np.isnan(bx))[0]
    if len(seen) < 2:
        return filled

    gaps_filled = 0
    # Reverse tracking before segment starts (e.g. kickoff / restart where initial frames miss ball)
    seg_starts = [int(seen[0])] + [
        int(seen[i + 1]) for i in range(len(seen) - 1) if (int(seen[i + 1]) - int(seen[i]) - 1) > max_gap
    ]
    rev_filled = 0
    for f_s in seg_starts:
        # Try backward velocity extrapolation up to 15 frames if ball moves right after f_s
        has_vel = (f_s + 3 < n) and not np.isnan(bx[f_s + 3])
        vx = (bx[f_s + 3] - bx[f_s]) / 3.0 if has_vel else 0.0
        vy = (by[f_s + 3] - by[f_s]) / 3.0 if has_vel else 0.0
        spd = float(np.hypot(vx, vy))
        d_s = np.hypot(P[f_s, :, 0] - bx[f_s], P[f_s, :, 1] - by[f_s])
        s_near = int(np.nanargmin(d_s)) if not np.all(np.isnan(d_s)) else -1
        for k in range(1, 21):
            f = f_s - k
            if f < 0 or not np.isnan(bx[f]):
                break
            if s_near >= 0 and d_s[s_near] <= near_r and not np.isnan(P[f, s_near, 0]):
                bx[f] = P[f, s_near, 0] + (bx[f_s] - P[f_s, s_near, 0])
                by[f] = P[f, s_near, 1] + (by[f_s] - P[f_s, s_near, 1])
                filled[f] = True
                rev_filled += 1
            elif has_vel and (0.15 * 25.0) <= (spd * fps) <= (1.6 * 25.0) and k <= max(3, int(round(0.48 * fps))):
                bx[f] = bx[f_s] - vx * k
                by[f] = by[f_s] - vy * k
                filled[f] = True
                rev_filled += 1
            else:
                break

    for idx in range(len(seen) - 1):
        f0 = int(seen[idx])
        f1 = int(seen[idx + 1])
        gap = f1 - f0 - 1
        if gap <= 0 or gap > max_gap:
            continue

        # Find nearest player to ball at f0
        d0 = np.hypot(P[f0, :, 0] - bx[f0], P[f0, :, 1] - by[f0])
        if np.all(np.isnan(d0)):
            continue
        s = int(np.nanargmin(d0))
        if d0[s] > near_r:
            continue

        # Ensure player s is tracked across the gap
        if np.isnan(P[f0:f1 + 1, s, 0]).any():
            continue

        off0_x = bx[f0] - P[f0, s, 0]
        off0_y = by[f0] - P[f0, s, 1]
        d1_s = float(np.hypot(P[f1, s, 0] - bx[f1], P[f1, s, 1] - by[f1]))

        if d1_s <= near_r + 0.5:
            # Same player still has the ball at f1 -> pure carry/dribble tracking
            off1_x = bx[f1] - P[f1, s, 0]
            off1_y = by[f1] - P[f1, s, 1]
            for f in range(f0 + 1, f1):
                t = (f - f0) / float(f1 - f0)
                bx[f] = P[f, s, 0] + (1.0 - t) * off0_x + t * off1_x
                by[f] = P[f, s, 1] + (1.0 - t) * off0_y + t * off1_y
                filled[f] = True
            gaps_filled += 1
        elif gap <= gap_pass:
            # Check if ball at f0 was already in fast collinear pass flight across the gap
            in_free_flight = False
            if f0 >= 3 and not np.isnan(bx[f0 - 3]):
                v_pre = np.array([bx[f0] - bx[f0 - 3], by[f0] - by[f0 - 3]], dtype=np.float64) / 3.0
                v_gap = np.array([bx[f1] - bx[f0], by[f1] - by[f0]], dtype=np.float64) / float(f1 - f0)
                s_pre = float(np.linalg.norm(v_pre)) * fps
                s_gap = float(np.linalg.norm(v_gap)) * fps
                if s_pre >= 6.0 and s_gap >= 5.0:
                    cos_p = float(np.dot(v_pre, v_gap) / (np.linalg.norm(v_pre) * np.linalg.norm(v_gap) + 1e-9))
                    if cos_p >= 0.70:
                        in_free_flight = True

            if in_free_flight:
                for f in range(f0 + 1, f1):
                    t = (f - f0) / float(f1 - f0)
                    bx[f] = (1.0 - t) * bx[f0] + t * bx[f1]
                    by[f] = (1.0 - t) * by[f0] + t * by[f1]
                    filled[f] = True
                gaps_filled += 1
            else:
                # Player s carried the ball, then passed it before f1
                pass_len = min(12, max(2, (gap + 1) // 2))
                f_split = f1 - pass_len
                for f in range(f0 + 1, f_split + 1):
                    bx[f] = P[f, s, 0] + off0_x
                    by[f] = P[f, s, 1] + off0_y
                    filled[f] = True
                sx, sy = bx[f_split], by[f_split]
                for f in range(f_split + 1, f1):
                    t = (f - f_split) / float(f1 - f_split)
                    bx[f] = (1.0 - t) * sx + t * bx[f1]
                    by[f] = (1.0 - t) * sy + t * by[f1]
                    filled[f] = True
                gaps_filled += 1

    if verbose:
        print(f"  Near-Player Gap Tracker: filled {int(filled.sum()):,} frames ({gaps_filled:,} carry/pass gaps, {rev_filled:,} reverse-start frames).")
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ball-json", default=None,
                    help="Path to cleaned ball JSON to fill missing ball frames (default: <data-dir>/ball_detections_cleaned.json)")
    ap.add_argument("--use-ball-json", action="store_true", default=True,
                    help="Whether to inject cleaned ball JSON into missing ATD frames (default: True)")
    ap.add_argument("--clean-ball-only", action="store_true", default=True,
                    help="Remove all ATD ball positions and use ONLY the cleaned ball JSON detections (default: True)")
    ap.add_argument("--correct-gk-side", action="store_true", default=True,
                    help="Enforce that goalkeepers cannot be on the wrong half/side of the pitch (default: True)")
    ap.add_argument("--pass-destination-duels", action="store_true", default=True,
                    help="Attribute multi-player contested collisions to the team that cleanly receives the emerging pass (default: True)")
    ap.add_argument("--quick-pass-transits", action="store_true", default=True,
                    help="Attribute completed quick pass transits between teammates to the passing team (default: True)")
    ap.add_argument("--gk-pass-flow", action="store_true", default=True,
                    help="Attribute passes to goalkeeper to passing team, and from goalkeeper to receiving team (default: True)")
    ap.add_argument("--kicker-in-motion", action="store_true", default=False,
                    help="Attribute ball-in-motion frames without clear control to the team that kicked last")
    ap.add_argument("--ground-transit-touch", action="store_true", default=True,
                    help="Protect ground ball transits from premature switching to opponents without physical touch contact (default: True)")
    ap.add_argument("--one-touch-flick", action="store_true", default=True,
                    help="A single flick/deflection by the new team is not possession; switch on their controlled receipt (default: True)")
    ap.add_argument("--gk-restart-pass", action="store_true", default=True,
                    help="Keeper restart pass to a teammate = keeper's team from the equal-distance point of the pass (default: True)")
    ap.add_argument("--kickoff-hold", action="store_true", default=True,
                    help="The team that kicks off keeps possession until the opponent clearly controls the ball (default: True)")
    ap.add_argument("--same-team-pass-midpoint", action="store_true", default=True,
                    help="On H<->A turnovers (outside GK boxes), switch at midpoint of first same-team pass (default: True)")
    ap.add_argument("--bbox-touch", action="store_true", default=False,
                    help="Hard-coded possession rules from ball-bbox / player-bbox contact (see rules.txt)")
    ap.add_argument("--atd-fill-missing", action="store_true", default=True,
                    help="With --clean-ball-only: use the ATD ball only on frames where the cleaned ball is missing (default: True)")
    ap.add_argument("--track-near-player-gaps", action="store_true", default=True,
                    help="Track/fill ball gaps only when the ball disappears near a player (carry/pass occlusion) (default: True)")
    ap.add_argument("--static-abandoned-dead", action="store_true", default=True,
                    help="Mark >=6.5s static-ball windows where all players are >=1.5m away as dead ball (default: True)")
    ap.add_argument("--unseen-blackout-dead", action="store_true", default=True,
                    help="Mark >=10s unseen-ball blackout spans with low player motion (<2.2 m/s) as dead ball (default: True)")
    ap.add_argument("--unconfirmed-aerial-live", action="store_true", default=True,
                    help="Unmark ballistic-only aerial gaps when ball reappears inside pitch and players stay moving (default: True)")
    ap.add_argument("--slow-out-roll-dead", action="store_true", default=True,
                    help="Mark slow roll-outs >=0.35m beyond pitch boundary for >=3f followed by >=10f unseen as dead ball (default: True)")
    ap.add_argument("--diversion-kicks", action="store_true", default=True,
                    help="Detect (dTheta, dV) ball diversion kicks with near player and switch if not reclaimed (default: True)")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override video FPS (defaults to games.GAMES[game]['fps'])")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "Matra.csv"))
    args = ap.parse_args()
    if args.ball_json is None:
        data_ball_json = os.path.join(args.data_dir or DATA_DIR, "ball_detections_cleaned.json")
        src_ball_json = os.path.join(SRC_DIR, "ball_detections_cleaned.json")
        args.ball_json = data_ball_json if os.path.isfile(data_ball_json) else src_ball_json

    _install_audit(args.data_dir or DATA_DIR)

    import hmm_engine
    import missing_ball_engine as mb
    import json
    deadball_engine, possession_engine = hmm_engine.import_engines()
    import games as _games
    fps_run = _sync_fps(
        args.fps if args.fps is not None else _games.GAMES.get(args.game, {}).get("fps", deadball_engine.FPS),
        deadball_engine, possession_engine, hmm_engine, mb,
    )

    t0 = time.time()
    print("[1/4] loading tactical ATD + homography (one pass) ...")
    trk = deadball_engine.load_tracking(args.game, args.data_dir)
    ball_override = None
    atd_fill = None

    if args.clean_ball_only:
        atd_x, atd_y = trk["ball_x"].copy(), trk["ball_y"].copy()
        print("  [CLEAN-BALL-ONLY MODE] Removing all ball positions from ATD txt file...")
        trk["ball_x"].fill(np.nan)
        trk["ball_y"].fill(np.nan)
        trk["ball_s"].fill(np.nan)
        if os.path.isfile(args.ball_json):
            print(f"  Populating ball positions exclusively from {args.ball_json} ...")
            loaded_clean = _populate_ball_from_json(
                trk, args.ball_json, args.game, args.data_dir, deadball_engine, only_missing=False
            )
            print(f"  Successfully loaded {loaded_clean:,} ball frames exclusively from cleaned JSON!")
            if args.track_near_player_gaps:
                tracked = track_near_player_gaps(trk, fps=fps_run)
                atd_fill = tracked if atd_fill is None else (atd_fill | tracked)
            if args.atd_fill_missing:
                P_all = trk["P"]
                dx_a = P_all[..., 0] - atd_x[:, None]
                dy_a = P_all[..., 1] - atd_y[:, None]
                with np.errstate(invalid="ignore"):
                    d_near_atd = np.min(np.where(np.isnan(dx_a), np.inf, np.hypot(dx_a, dy_a)), axis=1)
                in_pitch_atd = (
                    ~np.isnan(atd_x)
                    & (atd_x >= 1.0) & (atd_x <= deadball_engine.PITCH_L - 1.0)
                    & (atd_y >= 1.0) & (atd_y <= deadball_engine.PITCH_W - 1.0)
                    & (d_near_atd <= 4.0)
                )
                # Only fill ATD ball inside missing-ball gaps where the ATD track
                # connects smoothly (<= 2.5m) to our cleaned/tracked ball at both ends
                bx_cur, by_cur = trk["ball_x"], trk["ball_y"]
                n_frames = len(bx_cur)
                consistent_gap_mask = np.zeros(n_frames, dtype=bool)
                max_atd_gap = int(round(3.0 * fps_run))
                for g0, g1 in deadball_engine._runs(np.isnan(bx_cur)):
                    if (g1 - g0) <= max_atd_gap or g0 <= 0 or g1 >= n_frames:
                        continue
                    if np.isnan(atd_x[g0]) or np.isnan(atd_x[g1 - 1]):
                        continue
                    d_start = float(np.hypot(atd_x[g0] - bx_cur[g0 - 1], atd_y[g0] - by_cur[g0 - 1]))
                    d_end = float(np.hypot(atd_x[g1 - 1] - bx_cur[g1], atd_y[g1 - 1] - by_cur[g1]))
                    if d_start <= 2.5 and d_end <= 2.5:
                        consistent_gap_mask[g0:g1] = True
                miss = consistent_gap_mask & in_pitch_atd
                trk["ball_x"][miss] = atd_x[miss]
                trk["ball_y"][miss] = atd_y[miss]
                atd_fill = miss if atd_fill is None else (atd_fill | miss)
                print(f"  ATD fill: {int(miss.sum()):,} endpoint-consistent in-pitch frames taken from ATD ball")
            # Keep the raw (JSON [+ tracked/ATD fill]) ball so later stages never re-read the ATD ball file
            ball_override = (trk["ball_x"].copy(), trk["ball_y"].copy())
            trk["ball_x"], trk["ball_y"], trk["ball_s"] = deadball_engine._clean_ball(trk["ball_x"], trk["ball_y"])

    print("[2/4] STAGE 1 deadball engine (module_1) ...")
    if args.unconfirmed_aerial_live:
        dead, reasons = deadball_engine.build_dead_mask(trk, args.game, args.data_dir, return_reasons=True)
        bx_a, by_a = trk["ball_x"], trk["ball_y"]
        P_a = trk["P"]
        n_a = len(bx_a)
        Ps_a = np.hypot(P_a[2:, :, 0] - P_a[:-2, :, 0], P_a[2:, :, 1] - P_a[:-2, :, 1]) * (fps_run / 2.0)
        mot_a = np.nanmedian(Ps_a, axis=1)
        other_dead = reasons["trap"] | reasons["out"] | reasons["setpiece"] | reasons["goal"] | reasons["stoppage"]
        pad_b = int(round(1.0 * fps_run))
        reapp_w = int(round(2.0 * fps_run))
        removed_bal = 0
        for a_b, b_b in deadball_engine._runs(reasons["ballistic"] & ~other_dead):
            if np.any(other_dead[max(0, a_b - pad_b):min(n_a, b_b + pad_b)]):
                continue
            reapp = np.where(~np.isnan(bx_a[b_b:min(n_a, b_b + reapp_w)]))[0]
            if len(reapp) > 0:
                rf = b_b + int(reapp[0])
                if (2.5 <= bx_a[rf] <= 102.8) and (3.0 <= by_a[rf] <= 65.0):
                    if float(np.nanmin(mot_a[max(0, a_b - 1):min(len(mot_a), b_b)])) >= 1.1:
                        removed_bal += (b_b - a_b)
                        dead[a_b:b_b] = False
        print(f"  Unconfirmed Aerial Live Guard: unmarked {removed_bal:,} false ballistic-only dead frames.")
    else:
        dead = deadball_engine.build_dead_mask(trk, args.game, args.data_dir)
    if args.slow_out_roll_dead:
        bx_s, by_s = trk["ball_x"], trk["ball_y"]
        n_s = len(bx_s)
        vis_s = ~np.isnan(bx_s)
        m_out = 0.35
        beyond_s = vis_s & (
            (bx_s <= -m_out) | (bx_s >= deadball_engine.PITCH_L + m_out)
            | (by_s <= -m_out) | (by_s >= deadball_engine.PITCH_W + m_out)
        )
        ctx_s = deadball_engine._Ctx(trk)
        min_vis_out = max(2, int(round(0.12 * fps_run)))
        min_unseen_after = max(4, int(round(0.4 * fps_run)))
        added_slow = 0
        for a_s, b_s in deadball_engine._runs(beyond_s):
            if (b_s - a_s) >= min_vis_out and (b_s + min_unseen_after) <= n_s and np.all(~vis_s[b_s:b_s + min_unseen_after]):
                r_s = ctx_s.restart(b_s, a_s + int(deadball_engine.MAX_DEAD_S * fps_run), need_settle=False)
                end_s = r_s if r_s is not None else b_s
                added_slow += int(np.sum(~dead[a_s:end_s]))
                dead[a_s:end_s] = True
        print(f"  Slow Outward Roll Dead-Ball: marked {added_slow:,} additional dead frames.")
    if args.static_abandoned_dead:
        bx_d, by_d = trk["ball_x"], trk["ball_y"]
        P_d = trk["P"]
        n_d = len(bx_d)
        d_all = np.hypot(P_d[..., 0] - bx_d[:, None], P_d[..., 1] - by_d[:, None])
        d_near = np.where(np.all(np.isnan(d_all), axis=1), np.inf, np.nanmin(d_all, axis=1))
        W_stat = int(6.5 * fps_run)  # 162 frames at 25fps
        step_w = max(1, int(round(0.4 * fps_run)))  # 10 frames at 25fps
        min_vis_w = max(5, int(round(1.6 * fps_run)))  # 40 frames at 25fps
        added_stat = 0
        for s_w in range(0, n_d - W_stat, step_w):
            e_w = s_w + W_stat
            vis_w = ~np.isnan(bx_d[s_w:e_w])
            if int(np.sum(vis_w)) < min_vis_w:
                continue
            sx = float(np.nanmax(bx_d[s_w:e_w]) - np.nanmin(bx_d[s_w:e_w]))
            sy = float(np.nanmax(by_d[s_w:e_w]) - np.nanmin(by_d[s_w:e_w]))
            if np.hypot(sx, sy) <= 1.8 and float(np.median(d_near[s_w:e_w][vis_w])) >= 1.5:
                added_stat += int(np.sum(~dead[s_w:e_w]))
                dead[s_w:e_w] = True
        print(f"  Static Abandoned Ball Dead-Ball: marked {added_stat:,} additional dead frames.")
    if args.unseen_blackout_dead:
        bx_u = trk["ball_x"]
        P_u = trk["P"]
        Ps_u = np.hypot(P_u[2:, :, 0] - P_u[:-2, :, 0], P_u[2:, :, 1] - P_u[:-2, :, 1]) * (fps_run / 2.0)
        mot_u = np.nanmedian(Ps_u, axis=1)
        min_blk = int(round(10.0 * fps_run))  # 250 frames at 25fps
        added_blk = 0
        for a_u, b_u in deadball_engine._runs(np.isnan(bx_u)):
            if (b_u - a_u) >= min_blk:
                mm_u = float(np.nanmedian(mot_u[max(0, a_u - 1):min(len(mot_u), b_u - 1)]))
                if mm_u < 2.2:
                    added_blk += int(np.sum(~dead[a_u:b_u]))
                    dead[a_u:b_u] = True
        print(f"  Unseen Ball Blackout (>=10s): marked {added_blk:,} additional dead frames.")

    print("[3/4] STAGE 2 possession evidence (ball-player distances, ball speed) ...")
    # Inject cleaned ball detections into missing frames for possession evidence on alive frames
    if not args.clean_ball_only and args.use_ball_json and os.path.isfile(args.ball_json):
        print(f"  Injecting cleaned ball positions from {args.ball_json} ...")
        injected = _populate_ball_from_json(
            trk, args.ball_json, args.game, args.data_dir, deadball_engine, only_missing=True
        )
        print(f"  Successfully injected {injected:,} missing ball frames from cleaned JSON!")
        # Recompute cleaned ball velocities with filled coordinates
        trk["ball_x"], trk["ball_y"], trk["ball_s"] = deadball_engine._clean_ball(trk["ball_x"], trk["ball_y"])

    dynamic_team = None
    if args.correct_gk_side:
        dynamic_team = apply_passflow_player_reattribution(trk, args.game)

    print("[4/4] STAGE 2 missing-ball physics + HMM Viterbi decode ...")
    import missing_ball_engine as mb
    labels, dead, _tags, _stats = mb.decode_missing(
        trk, dead, args.game, args.data_dir,
        dynamic_team=dynamic_team,
        pass_destination_duels=args.pass_destination_duels,
        quick_pass_transits=args.quick_pass_transits,
        gk_pass_flow=args.gk_pass_flow,
        kicker_in_motion=args.kicker_in_motion,
        ground_transit_touch=args.ground_transit_touch,
        one_touch_flick=args.one_touch_flick,
        gk_restart_pass=args.gk_restart_pass,
        ball_override=ball_override,
        kickoff_hold=args.kickoff_hold,
        same_team_pass_midpoint=args.same_team_pass_midpoint
    )

    if args.diversion_kicks:
        bx_k, by_k = trk["ball_x"], trk["ball_y"]
        P_k = trk["P"]
        n_k = len(bx_k)
        slot_team_k = dynamic_team if dynamic_team is not None else np.broadcast_to(trk["slot_team"], (n_k, P_k.shape[1]))
        d_k = np.hypot(P_k[..., 0] - bx_k[:, None], P_k[..., 1] - by_k[:, None])
        dh_k = np.full(n_k, np.inf)
        da_k = np.full(n_k, np.inf)
        for f_i in range(n_k):
            if np.isnan(bx_k[f_i]):
                continue
            hm = (slot_team_k[f_i] == 0)
            if np.any(~np.isnan(d_k[f_i, hm])):
                dh_k[f_i] = float(np.nanmin(d_k[f_i, hm]))
            if np.any(~np.isnan(d_k[f_i, ~hm])):
                da_k[f_i] = float(np.nanmin(d_k[f_i, ~hm]))
        w_k = max(2, int(round(0.2 * fps_run)))       # 5 frames at 25fps
        w_disp = max(3, int(round(0.4 * fps_run)))    # 10 frames at 25fps
        reclaim_hi = int(round(1.6 * fps_run))        # 40 frames at 25fps
        max_hold_k = int(round(3.6 * fps_run))        # 90 frames at 25fps
        vel_scale = fps_run / float(w_k)              # 5.0 at 25fps
        div_applied = 0
        for f_i in range(w_k + 1, n_k - (w_disp + 2)):
            if dead[f_i] or np.isnan(bx_k[f_i]) or np.isnan(bx_k[f_i - w_k]) or np.isnan(bx_k[f_i + w_k]) or np.isnan(bx_k[f_i + w_disp]):
                continue
            vx1, vy1 = bx_k[f_i] - bx_k[f_i - w_k], by_k[f_i] - by_k[f_i - w_k]
            vx2, vy2 = bx_k[f_i + w_k] - bx_k[f_i], by_k[f_i + w_k] - by_k[f_i]
            n1, n2 = float(np.hypot(vx1, vy1)), float(np.hypot(vx2, vy2))
            s1, s2 = n1 * vel_scale, n2 * vel_scale
            post_disp = float(np.hypot(bx_k[f_i + w_disp] - bx_k[f_i], by_k[f_i + w_disp] - by_k[f_i]))
            if post_disp < 3.5 or s2 < 8.0:
                continue
            cos_a = float(np.clip((vx1 * vx2 + vy1 * vy2) / (n1 * n2), -1.0, 1.0)) if (n1 > 0.3 and n2 > 0.3) else 1.0
            dth = float(np.degrees(np.arccos(cos_a)))
            if not (dth >= 40.0 or (s2 - s1) >= 8.0):
                continue
            team_k = None
            if dh_k[f_i] <= 1.25 and da_k[f_i] >= 2.0 and labels[f_i] != "H":
                team_k = "H"
            elif da_k[f_i] <= 1.25 and dh_k[f_i] >= 2.0 and labels[f_i] != "A":
                team_k = "A"
            if team_k is None:
                continue
            opp_d = da_k if team_k == "H" else dh_k
            own_d = dh_k if team_k == "H" else da_k
            opp_reclaim = False
            end_k = min(n_k, f_i + max_hold_k)
            for k_i in range(f_i + w_k, min(n_k, f_i + reclaim_hi)):
                if dead[k_i]:
                    end_k = k_i
                    break
                if opp_d[k_i] <= 1.35 and own_d[k_i] >= 2.0:
                    opp_reclaim = True
                    break
            if opp_reclaim:
                continue
            div_applied += 1
            for k_i in range(f_i, end_k):
                if dead[k_i] or (opp_d[k_i] <= 1.35 and own_d[k_i] >= 2.0):
                    break
                labels[k_i] = team_k
        print(f"  Diversion Kicks (dTheta/dV): applied {div_applied} non-reclaimed kick switches.")

    if args.bbox_touch:
        import load_homography as lh
        import bbox_touch_engine as bte
        H_all, _, valid_all = lh.load(args.game, args.data_dir)
        ball_m = ball_override if ball_override is not None else (trk["ball_x"], trk["ball_y"])
        graded_m = deadball_engine.graded_mask(args.game)
        labels = bte.apply_bbox_rules(
            labels, dead, graded_m, trk, args.ball_json,
            np.asarray(H_all, dtype=float), np.asarray(valid_all, dtype=bool), ball_m,
            deadball_engine.PITCH_L, deadball_engine.PITCH_W,
            atd_fill=atd_fill,
        )

    graded = deadball_engine.graded_mask(args.game)
    labels[~graded] = "X"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    # --- sanity report ------------------------------------------------------
    n = len(labels)
    alive = graded & ~dead
    print("\nwrote {}  ({} rows)  in {:.0f}s".format(args.out, n, time.time() - t0))
    for c in "HADX":
        k = int((labels == c).sum())
        print("  {} {:>7}  ({:.1f}%)".format(c, k, 100.0 * k / n))
    print("sanity:")
    if n > deadball_engine.SECOND_HALF_KICKOFF:
        trap = labels[deadball_engine.SECOND_HALF_GRADED_START:deadball_engine.SECOND_HALF_KICKOFF]
        print("  2nd-half trap {}-{} all D : {}".format(
            deadball_engine.SECOND_HALF_GRADED_START, deadball_engine.SECOND_HALF_KICKOFF - 1, bool(np.all(trap == "D"))))
    print("  graded frames labelled X          : {}".format(int((labels[graded] == "X").sum())))
    print("  dead-mask frames not D            : {}".format(int((labels[dead] != "D").sum())))
    print("  data files opened (tactical only) :")
    for p in sorted(_OPENED):
        print("    data/" + p)
    # Non-tactical data check: only broadcast/ is prohibited
    assert not any(p.startswith("broadcast" + os.sep) for p in _OPENED), "broadcast data is prohibited"
    print("\nnext: python evaluation/validate_submission.py {}".format(args.out))
    print("      python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
