#!/usr/bin/env python3
"""
Render Match Half / Full Match Tactical Video with Bounding Boxes & Possession Overlay.

Features:
- Full support for First Half (11,185 .. 90,651) and Second Half (110,808 .. 190,197).
- Draws Bounding Boxes for all 22 players and the ball across every frame:
    * Home (Barcelona): Royal Blue box with slot ID (H1..H24)
    * Home GK (Slot 24): Gold/Yellow box
    * Away (Athletic Club): Crimson Red box with slot ID (A25..A46)
    * Away GK (Slot 46): Orange/Amber box
    * Dynamically reattributed defensive players: Magenta box with tag
    * Ball: Yellow box & centroid dot with detection confidence badge
- Top Notification HUD:
    * For first 5 minutes (Frames 11,185 .. 18,684):
        - Compares Ground Truth vs Prediction
        - Green/Cyan "[OK] MATCH" or Red "[!] MISMATCH DETECTED"
    * Outside first 5 minutes (Remainder of Half 1 & Full Half 2):
        - Writes PREDICTION ALONE (no false mismatch alerts):
          * Blue banner for HOME possession
          * Red banner for AWAY possession
          * Slate/Gray banner for DEAD BALL / stoppage
- Streams ATD tracking data in 1:1 lockstep with video frames (ultra-fast, zero-memory).
"""

import os
import sys
import csv
import json
import argparse
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import starter.load_homography as lh  # noqa: E402
import src.hmm_engine as he  # noqa: E402
import build_submission as bs  # noqa: E402

# Half video frame boundaries from games.py
HALF1_START = 11185
HALF1_END = 90651
HALF2_START = 110808
HALF2_END = 190197

# 5-minute ground truth benchmark window
GT_START = 11185
GT_END = 18684

VIDEO_PATH = "data/tactical/BAR-ATH_tactical-002.mp4"
ATD_PATH = "data/tactical/BAR-ATH_tactical_atd.txt"
GT_PATH = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
PRED_PATH = "evaluation/results/team_engine_v4_myball_tracked.csv"
BALL_JSON_PATH = "data/ball_detections_cleaned.json"

# Color Palette (RGB for Pillow)
COLOR_HOME = (37, 99, 235)         # Royal Blue
COLOR_HOME_GK = (234, 179, 8)       # Gold / Yellow
COLOR_AWAY = (220, 38, 38)         # Crimson Red
COLOR_AWAY_GK = (249, 115, 22)      # Amber / Orange
COLOR_BALL = (255, 255, 0)         # Bright Yellow
COLOR_CORRECTED = (236, 72, 153)   # Vibrant Magenta
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Banner Backgrounds (RGBA)
BG_MATCH = (15, 23, 42, 225)       # Dark Slate Cyan
BG_MISMATCH = (185, 28, 28, 240)    # Red Alert
BG_PRED_H = (30, 58, 138, 225)      # Deep Navy Blue
BG_PRED_A = (153, 27, 27, 225)      # Deep Crimson Red
BG_PRED_D = (31, 41, 55, 225)       # Charcoal Dark Gray


def load_gt_and_pred(gt_path, pred_path, total_frames=199319):
    """Load Ground Truth and Prediction lists up to total_frames."""
    gt_all = ["X"] * total_frames
    if os.path.isfile(gt_path):
        with open(gt_path, mode="r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for i, row in enumerate(r):
                if i < total_frames and len(row) >= 2:
                    gt_all[i] = row[1].strip()

    pred_all = ["X"] * total_frames
    if os.path.isfile(pred_path):
        with open(pred_path, mode="r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for i, row in enumerate(r):
                if i < total_frames and len(row) >= 2:
                    pred_all[i] = row[1].strip()

    return gt_all, pred_all


def render_video_segment(
    start_frame,
    end_frame,
    output_mp4,
    video_source=None,
    atd_path=None,
    gt_path=GT_PATH,
    pred_path=PRED_PATH,
    ball_json_path=None,
    stride=1,
    no_atd_ball=False,
    track_near_player=False,
    data_dir=None,
    game=None,
):
    import games as _games
    g_name = game or _games.GAME
    ddir = data_dir or _games.DATA_DIR
    t_paths = _games.asset_paths(game=g_name, data_dir=ddir, source="tactical")
    video_source = video_source or t_paths["video"]
    atd_path = atd_path or t_paths["atd_raw"]
    ball_json_path = ball_json_path or os.path.join(ddir, "ball_detections_cleaned.json")

    print("=" * 75)
    print("Match Video Possession & Bounding Box Renderer")
    print(f"  Frame Range: {start_frame:,} .. {end_frame:,} ({(end_frame - start_frame + 1):,} frames, stride={stride})")
    print(f"  Input Video: {video_source}")
    print(f"  Output MP4:  {output_mp4}")
    print(f"  Predictions: {pred_path}")
    print("=" * 75)

    gt_all, pred_all = load_gt_and_pred(gt_path, pred_path, total_frames=_games.GAMES[g_name]["n_video_frames"])

    # Load Homography matrices
    print("  Loading tactical homography matrices...")
    matrices, _, valid = lh.load(g_name, ddir)

    # Load Tracking and Dynamic GK reattributions
    print("  Loading tracking and goalkeeper dynamic reattributions...")
    deadball_engine, _ = he.import_engines()
    trk = deadball_engine.load_tracking(g_name, ddir)
    if no_atd_ball or track_near_player:
        trk["ball_x"][:] = np.nan
        trk["ball_y"][:] = np.nan
        trk["ball_s"][:] = np.nan
        bs._populate_ball_from_json(trk, ball_json_path, g_name, ddir, deadball_engine, only_missing=False)
    if track_near_player:
        bs.track_near_player_gaps(trk, verbose=True)
    dyn = bs.apply_passflow_player_reattribution(trk, g_name, verbose=False)

    # Load Cleaned Ball Detections
    ball_json = {}
    if os.path.isfile(ball_json_path):
        print(f"  Loading cleaned ball detections from {ball_json_path}...")
        with open(ball_json_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            ball_json = data.get("frames", {})

    # Start ffmpeg input pipe (demuxing rawvideo rgb24)
    start_sec = start_frame / 25.0
    seek_str = f"{int(start_sec//3600):02d}:{int((start_sec%3600)//60):02d}:{start_sec%60:06.3f}"
    num_frames = end_frame - start_frame + 1

    in_cmd = [
        "ffmpeg", "-hwaccel", "cuda", "-ss", seek_str,
        "-i", video_source,
        "-vframes", str(num_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1"
    ]
    p_in = subprocess.Popen(in_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    # Start ffmpeg output pipe (encoding H.264 MP4 on GPU via h264_nvenc)
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    out_fps = 25 // stride
    out_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", "1920x1080",
        "-r", str(out_fps),
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-preset", "p4",
        "-cq", "20",
        output_mp4
    ]
    p_out = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    # TrueType Fonts
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.isfile(font_path):
        font_large = ImageFont.truetype(font_path, 20)
        font_mid = ImageFont.truetype(font_path, 16)
        font_badge = ImageFont.truetype(font_path, 11)
    else:
        font_large = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_badge = ImageFont.load_default()

    frame_bytes = 1920 * 1080 * 3
    processed = 0

    print("Rendering video with player bounding boxes, ball, and possession overlay...")

    # Open ATD file and fast-forward to start_frame
    with open(atd_path, mode="r", encoding="utf-8") as f_atd:
        for _ in range(start_frame):
            f_atd.readline()

        for f_idx in range(num_frames):
            current_vf = start_frame + f_idx
            raw = p_in.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            atd_line = f_atd.readline()

            # If stride > 1, skip intermediate frames
            if stride > 1 and (f_idx % stride != 0):
                continue

            img = Image.frombytes("RGB", (1920, 1080), raw)
            draw = ImageDraw.Draw(img, "RGBA")

            H = matrices[current_vf] if (current_vf < len(valid) and valid[current_vf]) else None
            gt_label = gt_all[current_vf] if current_vf < len(gt_all) else "X"
            pred_label = pred_all[current_vf] if current_vf < len(pred_all) else "X"
            has_gt = (gt_label in ("H", "A", "D"))

            # ---------------------------------------------------------
            # 1. PARSE & DRAW PLAYER BOUNDING BOXES
            # ---------------------------------------------------------
            players_data = None
            ball_data = None
            if atd_line:
                parts = atd_line.strip().split(":")
                if len(parts) == 3:
                    players_data = parts[1].split(";")
                    ball_data = parts[2].split(",")

            if players_data is not None and H is not None:
                for s_idx, s_str in enumerate(players_data):
                    if not s_str or s_str[0] == "N":
                        continue
                    vals = s_str.split(",")
                    if len(vals) < 2 or vals[0] == "NaN" or vals[1] == "NaN":
                        continue
                    slot_num = s_idx + 1
                    norm_x = float(vals[0])
                    norm_y = float(vals[1])

                    denom = H[2, 0] * norm_x + H[2, 1] * norm_y + H[2, 2]
                    if abs(denom) < 1e-7:
                        continue
                    u = int(round((H[0, 0] * norm_x + H[0, 1] * norm_y + H[0, 2]) / denom))
                    v = int(round((H[1, 0] * norm_x + H[1, 1] * norm_y + H[1, 2]) / denom))

                    if not (0 <= u < 1920 and 0 <= v < 1080):
                        continue

                    is_home = (slot_num <= 24)
                    is_gk = (slot_num in (24, 46))
                    is_corrected = (dyn is not None and dyn[current_vf, s_idx] != (0 if slot_num <= 24 else 1))

                    if is_corrected:
                        color = COLOR_CORRECTED
                    elif is_home:
                        color = COLOR_HOME_GK if is_gk else COLOR_HOME
                    else:
                        color = COLOR_AWAY_GK if is_gk else COLOR_AWAY

                    # Bounding Box: ~52px tall, 32px wide
                    x1, y1 = u - 16, v - 52
                    x2, y2 = u + 16, v + 4
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3 if is_corrected else 2)
                    draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color)

                    role_str = "H" if is_home else "A"
                    if is_corrected:
                        corr_team = "H" if dyn[current_vf, s_idx] == 0 else "A"
                        tag = f"{role_str}{slot_num}*[GK->{corr_team}]"
                        badge_w = 66
                    else:
                        tag = f"{role_str}{slot_num}" + ("[GK]" if is_gk else "")
                        badge_w = 44
                    draw.rectangle([x1 - 2, y1 - 18, x1 + badge_w, y1 - 1], fill=(*color, 230))
                    draw.text((x1 + 2, y1 - 17), tag, fill=COLOR_WHITE, font=font_badge)

                # -----------------------------------------------------
                # 2. DRAW BALL BOUNDING BOX & CENTROID
                # -----------------------------------------------------
                bx_u, by_v = None, None
                ball_bbox = None
                ball_source = "ATD"

                if ball_json and str(current_vf) in ball_json and ball_json[str(current_vf)]:
                    dets = ball_json[str(current_vf)][0]
                    ball_bbox = dets.get("bbox_xyxy")
                    cp = dets.get("centroid_pixel")
                    if cp:
                        bx_u, by_v = int(round(cp["x"])), int(round(cp["y"]))
                    if "interpolated_gap" in dets:
                        ball_source = f"TRAJ GAP {dets['interpolated_gap']}f"
                    else:
                        ball_source = f"CLEAN {dets.get('confidence', 1.0):.2f}"
                elif track_near_player and np.isfinite(trk["ball_x"][current_vf]) and np.isfinite(trk["ball_y"][current_vf]):
                    bnx, bny = float(trk["ball_x"][current_vf]), float(trk["ball_y"][current_vf])
                    denom = H[2, 0] * bnx + H[2, 1] * bny + H[2, 2]
                    if abs(denom) >= 1e-7:
                        bx_u = int(round((H[0, 0] * bnx + H[0, 1] * bny + H[0, 2]) / denom))
                        by_v = int(round((H[1, 0] * bnx + H[1, 1] * bny + H[1, 2]) / denom))
                    ball_source = "TRACKED"
                elif (not no_atd_ball) and ball_data is not None and len(ball_data) >= 2 and ball_data[0] != "NaN" and ball_data[1] != "NaN":
                    bnx, bny = float(ball_data[0]), float(ball_data[1])
                    denom = H[2, 0] * bnx + H[2, 1] * bny + H[2, 2]
                    if abs(denom) >= 1e-7:
                        bx_u = int(round((H[0, 0] * bnx + H[0, 1] * bny + H[0, 2]) / denom))
                        by_v = int(round((H[1, 0] * bnx + H[1, 1] * bny + H[1, 2]) / denom))

                if ball_source == "TRACKED":
                    ball_color = (0, 229, 255)
                elif ball_source.startswith("TRAJ GAP"):
                    ball_color = (0, 255, 128)
                else:
                    ball_color = COLOR_BALL
                if ball_bbox is not None:
                    bx1, by1, bx2, by2 = [int(round(c)) for c in ball_bbox]
                    draw.rectangle([bx1, by1, bx2, by2], outline=ball_color, width=2)
                    draw.text((bx2 + 4, by1 - 4), f"BALL [{ball_source}]", fill=ball_color, font=font_badge)
                    if bx_u is not None and by_v is not None and (0 <= bx_u < 1920 and 0 <= by_v < 1080):
                        draw.ellipse([bx_u - 3, by_v - 3, bx_u + 3, by_v + 3], fill=ball_color)
                elif bx_u is not None and by_v is not None and (0 <= bx_u < 1920 and 0 <= by_v < 1080):
                    draw.ellipse([bx_u - 7, by_v - 7, bx_u + 7, by_v + 7], fill=ball_color, outline=COLOR_BLACK, width=1)
                    draw.rectangle([bx_u - 10, by_v - 10, bx_u + 10, by_v + 10], outline=ball_color, width=2)
                    draw.text((bx_u + 14, by_v - 8), f"BALL [{ball_source}]", fill=ball_color, font=font_badge)

            # ---------------------------------------------------------
            # 3. TOP NOTIFICATION HUD (GT Window vs Prediction Alone)
            # ---------------------------------------------------------
            hud_w, hud_h = 920, 84
            hud_x = (1920 - hud_w) // 2
            hud_y = 20

            # Match clock calculation
            if current_vf < HALF2_START:
                elapsed_sec = max(0.0, (current_vf - HALF1_START) / 25.0)
                half_prefix = "1H"
            else:
                elapsed_sec = 45.0 * 60.0 + max(0.0, (current_vf - 113742) / 25.0)
                half_prefix = "2H"

            time_str = f"{half_prefix} {int(elapsed_sec // 60):02d}:{elapsed_sec % 60:04.1f}"

            if has_gt:
                # -----------------------------------------------------
                # MODE A: 5-MIN BENCHMARK WINDOW (GT VS PREDICTION)
                # -----------------------------------------------------
                if gt_label != pred_label:
                    draw.rectangle([hud_x, hud_y, hud_x + hud_w, hud_y + hud_h], fill=BG_MISMATCH, outline=(255, 120, 120), width=3)
                    draw.text((hud_x + 20, hud_y + 12),
                              f"[!] POSSESSION MISMATCH DETECTED  |  Match Time: {time_str}  |  Frame: {current_vf:,}",
                              fill=(255, 255, 200), font=font_large)
                    draw.text((hud_x + 20, hud_y + 44),
                              f"GROUND TRUTH: [ {gt_label} ]  vs  PREDICTION: [ {pred_label} ]   (H = Barcelona, A = Athletic Club, D = Dead)",
                              fill=COLOR_WHITE, font=font_mid)
                else:
                    draw.rectangle([hud_x, hud_y, hud_x + hud_w, hud_y + hud_h], fill=BG_MATCH, outline=(56, 189, 248), width=2)
                    draw.text((hud_x + 20, hud_y + 14),
                              f"[OK] GROUND TRUTH & PREDICTION MATCH  |  Match Time: {time_str}  |  Frame: {current_vf:,}",
                              fill=(180, 230, 255), font=font_large)
                    draw.text((hud_x + 20, hud_y + 44),
                              f"Possession: {gt_label} (HOME = Barcelona, AWAY = Athletic Club, D = Dead Ball)",
                              fill=(200, 220, 240), font=font_mid)
            else:
                # -----------------------------------------------------
                # MODE B: REST OF MATCH (PREDICTION ALONE - NO GT)
                # -----------------------------------------------------
                if pred_label == "H":
                    bg_col = BG_PRED_H
                    border_col = (96, 165, 250)
                    status_title = "[HOME POSSESSION] FC Barcelona Control"
                    detail_str = "Possession: HOME (Barcelona)  |  HMM & Ball-Physics Tracking"
                elif pred_label == "A":
                    bg_col = BG_PRED_A
                    border_col = (248, 113, 113)
                    status_title = "[AWAY POSSESSION] Athletic Club Control"
                    detail_str = "Possession: AWAY (Athletic Club)  |  HMM & Ball-Physics Tracking"
                elif pred_label == "D":
                    bg_col = BG_PRED_D
                    border_col = (156, 163, 175)
                    status_title = "[DEAD BALL / STOPPAGE] Out of Play / Set Piece"
                    detail_str = "Status: DEAD BALL (Stage-1 Physics & Smart Tagging Mask)"
                else:
                    bg_col = (20, 20, 20, 220)
                    border_col = (100, 100, 100)
                    status_title = f"[UNGRADED PLAY] State: {pred_label}"
                    detail_str = "Status: Ungraded match interval"

                draw.rectangle([hud_x, hud_y, hud_x + hud_w, hud_y + hud_h], fill=bg_col, outline=border_col, width=2)
                draw.text((hud_x + 20, hud_y + 14),
                          f"{status_title}  |  {time_str}  |  Frame: {current_vf:,}",
                          fill=COLOR_WHITE, font=font_large)
                draw.text((hud_x + 20, hud_y + 44), detail_str, fill=(220, 225, 235), font=font_mid)

            # Dynamic GK badge indicator
            if dyn is not None:
                active_gk_corr = None
                for s in range(46):
                    nom_t = 0 if s < 24 else 1
                    if dyn[current_vf, s] != nom_t:
                        active_gk_corr = (s + 1, "Home" if dyn[current_vf, s] == 0 else "Away")
                        break
                if active_gk_corr:
                    draw.rectangle([hud_x, hud_y + hud_h - 22, hud_x + hud_w, hud_y + hud_h], fill=(236, 72, 153, 230))
                    draw.text((hud_x + 20, hud_y + hud_h - 18),
                              f"[*] GK PASS-FLOW CORRECTION: Slot {active_gk_corr[0]} attributed to {active_gk_corr[1]}",
                              fill=COLOR_WHITE, font=font_badge)

            p_out.stdin.write(img.tobytes())
            processed += 1

            if processed % 500 == 0:
                pct = (processed / (num_frames // stride)) * 100.0
                print(f"  Rendered {processed:,}/{(num_frames // stride):,} frames ({pct:.1f}%) ...")

    p_in.stdout.close()
    p_in.wait()
    p_out.stdin.close()
    p_out.wait()

    print("=" * 75)
    print(f"SUCCESS! Rendered video written to: {output_mp4}")
    print("=" * 75)


def main():
    ap = argparse.ArgumentParser(description="Render Match Half / Full Match Video with Possession Overlay")
    ap.add_argument("--half", type=int, choices=[1, 2], default=None,
                    help="Render complete half: 1 (11,185..90,651) or 2 (110,808..190,197)")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="Custom start frame")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="Custom end frame")
    ap.add_argument("--stride", type=int, default=1,
                    help="Render every Nth frame (default=1 for full 25fps video; use 2 or 5 for fast preview)")
    ap.add_argument("--data-dir", default=None,
                    help="Custom data directory (default: MTAH_DATA_DIR or ./data)")
    ap.add_argument("--game", default=None,
                    help="Game identifier in games.py (default: games.GAME)")
    ap.add_argument("--pred", default=PRED_PATH,
                    help="Path to prediction CSV")
    ap.add_argument("--ball-json", default=None,
                    help="Path to cleaned ball JSON (default: <data-dir>/ball_detections_cleaned.json)")
    ap.add_argument("--no-atd-ball", action="store_true",
                    help="Draw only the cleaned JSON ball (never fall back to ATD ball)")
    ap.add_argument("--track-near-player-gaps", action="store_true",
                    help="Draw near-player tracked ball in cyan on gaps")
    ap.add_argument("--out", default=None,
                    help="Output MP4 video path")
    args = ap.parse_args()

    import games as _games
    g_name = args.game or _games.GAME
    ddir = args.data_dir or _games.DATA_DIR
    blocks = {h: (s, e) for h, s, e in _games.half_video_blocks(g_name)}

    if args.half == 1:
        s_frame, e_frame = blocks.get(1, (HALF1_START, HALF1_END))
        out_path = args.out or os.path.join(ddir, "tactical", f"{g_name}_tactical_half1_possession.mp4")
    elif args.half == 2:
        s_frame, e_frame = blocks.get(2, (HALF2_START, HALF2_END))
        out_path = args.out or os.path.join(ddir, "tactical", f"{g_name}_tactical_half2_possession.mp4")
    else:
        if args.start_frame is None or args.end_frame is None:
            print("Please specify either --half [1|2] or both --start-frame and --end-frame.")
            sys.exit(1)
        s_frame = args.start_frame
        e_frame = args.end_frame
        out_path = args.out or os.path.join(ddir, "tactical", f"{g_name}_custom_{s_frame}_{e_frame}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    render_video_segment(
        start_frame=s_frame,
        end_frame=e_frame,
        output_mp4=out_path,
        pred_path=args.pred,
        ball_json_path=args.ball_json,
        stride=args.stride,
        no_atd_ball=args.no_atd_ball,
        track_near_player=args.track_near_player_gaps,
        data_dir=ddir,
        game=g_name,
    )


if __name__ == "__main__":
    main()
