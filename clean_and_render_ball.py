#!/usr/bin/env python3
"""
Clean ball detections from a detection CSV using continuous trajectory linking,
write the cleaned ball CSV, and render an annotated video over the CSV frame span.

Filtering rules:
1. Size & confidence filtering: rejects player-sized boxes (e.g., height > 28px)
   and non-spherical boxes, keeping only plausible ball candidates.
2. Continuous trajectory linking: links detections across frames into tracklets
   using kinematic gating (max speed per frame + velocity prediction) so that
   multi-detection frames (2-6 boxes from shoes/lines) are disambiguated and
   brief aerial/occlusion dropouts (up to --max-gap frames) are bridged smoothly.
3. Tracklet validation: drops isolated/rare false positives (tracklets shorter
   than --min-track-len detections) that do not form a continuous trajectory.
4. Video rendering: renders the source video from the CSV's first frame to its
   last frame with only the cleaned ball box, trajectory trail, and frame HUD.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    import games as _games
    DEFAULT_GAME = _games.GAME
    DEFAULT_DATA_DIR = _games.DATA_DIR
    _TACTICAL_DEFAULTS = _games.asset_paths(game=DEFAULT_GAME, data_dir=DEFAULT_DATA_DIR, source="tactical")
    DEFAULT_VIDEO = _TACTICAL_DEFAULTS["video"]
    DEFAULT_HOMOGRAPHY = _TACTICAL_DEFAULTS["homography"]
except Exception:
    _games = None
    DEFAULT_GAME = "BAR-ATH"
    DEFAULT_DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO_ROOT, "data")
    DEFAULT_VIDEO = os.path.join(DEFAULT_DATA_DIR, "tactical", f"{DEFAULT_GAME}_tactical.mp4")
    DEFAULT_HOMOGRAPHY = os.path.join(DEFAULT_DATA_DIR, "tactical", f"{DEFAULT_GAME}_tactical_homography.csv")


def load_homographies(homography_csv_path):
    """Load homography CSV into frame_idx -> H_inv (3x3) for valid frames."""
    if not homography_csv_path or not os.path.isfile(homography_csv_path):
        raise FileNotFoundError(f"Homography CSV not found: {homography_csv_path}")

    homographies = {}
    with open(homography_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fn = int(row["frame"])
                valid = int(float(row.get("valid", 1)))
                if valid != 1:
                    continue
                H = np.array([
                    float(row["h00"]), float(row["h01"]), float(row["h02"]),
                    float(row["h10"]), float(row["h11"]), float(row["h12"]),
                    float(row["h20"]), float(row["h21"]), float(row["h22"]),
                ], dtype=np.float64).reshape((3, 3))
                if np.isnan(H).any() or abs(np.linalg.det(H)) <= 1e-9:
                    continue
                homographies[fn] = np.linalg.inv(H)
            except (ValueError, KeyError, np.linalg.LinAlgError):
                continue
    return homographies


def is_inside_pitch(H_inv, u, v, margin=0.02):
    """Return True if pixel (u, v) projects inside [0-margin, 1+margin]^2 pitch bounds."""
    if H_inv is None:
        return True
    pt = H_inv @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    w = pt[2]
    if abs(w) < 1e-7 or np.isnan(w):
        return False
    nx = pt[0] / w
    ny = pt[1] / w
    if np.isnan(nx) or np.isnan(ny):
        return False
    return (-margin <= nx <= 1.0 + margin) and (-margin <= ny <= 1.0 + margin)


def is_valid_ball_size(w, h, min_size=3.0, max_size=28.0, min_aspect=0.4, max_aspect=2.2):
    """Return True if bounding box dimensions match a tactical-view football."""
    if w < min_size or h < min_size:
        return False
    if w > max_size or h > max_size:
        return False
    aspect = w / max(1e-6, h)
    if aspect < min_aspect or aspect > max_aspect:
        return False
    return True


def load_ball_candidates(
    csv_path,
    min_conf=0.35,
    min_size=3.0,
    max_size=28.0,
    homographies=None,
    pitch_margin=0.02,
):
    """
    Load raw detections from CSV and filter to plausible ball-sized candidates
    inside the pitch homography boundaries.
    Returns (candidates_by_frame, min_frame, max_frame, stats).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    candidates = defaultdict(list)
    min_frame = None
    max_frame = None
    total_rows = 0
    detected_rows = 0
    rejected_size = 0
    rejected_conf = 0
    rejected_outside_pitch = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                fn = int(row.get("frame_number") or row.get("frame"))
            except (TypeError, ValueError):
                continue

            if min_frame is None or fn < min_frame:
                min_frame = fn
            if max_frame is None or fn > max_frame:
                max_frame = fn

            if str(row.get("ball_detected", "1")).strip() != "1":
                continue
            if not row.get("x_center") or not row.get("y_center"):
                continue

            detected_rows += 1
            try:
                conf = float(row.get("confidence") or 0.0)
                cx = float(row["x_center"])
                cy = float(row["y_center"])
                w = float(row["width"])
                h = float(row["height"])
                x1 = float(row.get("x1") or (cx - w / 2.0))
                y1 = float(row.get("y1") or (cy - h / 2.0))
                x2 = float(row.get("x2") or (cx + w / 2.0))
                y2 = float(row.get("y2") or (cy + h / 2.0))
            except (TypeError, ValueError, KeyError):
                continue

            if conf < min_conf:
                rejected_conf += 1
                continue

            if not is_valid_ball_size(w, h, min_size=min_size, max_size=max_size):
                rejected_size += 1
                continue

            if homographies is not None:
                H_inv = homographies.get(fn)
                if H_inv is not None and not is_inside_pitch(H_inv, cx, cy, margin=pitch_margin):
                    rejected_outside_pitch += 1
                    continue

            candidates[fn].append({
                "frame": fn,
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
            })

    stats = {
        "total_rows": total_rows,
        "detected_rows": detected_rows,
        "rejected_conf": rejected_conf,
        "rejected_size": rejected_size,
        "rejected_outside_pitch": rejected_outside_pitch,
        "candidate_frames": len(candidates),
        "candidate_boxes": sum(len(v) for v in candidates.values()),
    }
    return candidates, min_frame, max_frame, stats


def predict_track_position(points, target_frame, max_speed_px=45.0):
    """
    Predict (px, py) at target_frame from the tail of a tracklet's points.
    Uses recent velocity over up to last 5 detections.
    """
    last = points[-1]
    dt = target_frame - last["frame"]
    if len(points) < 2:
        return last["cx"], last["cy"]

    recent = points[-min(5, len(points)):]
    f0 = recent[0]["frame"]
    f1 = recent[-1]["frame"]
    span = f1 - f0
    if span <= 0:
        return last["cx"], last["cy"]

    vx = (recent[-1]["cx"] - recent[0]["cx"]) / span
    vy = (recent[-1]["cy"] - recent[0]["cy"]) / span
    speed = math.hypot(vx, vy)
    if speed > max_speed_px:
        scale = max_speed_px / speed
        vx *= scale
        vy *= scale

    # Dampen velocity slightly across longer aerial gaps
    damp = 0.85 ** max(0, dt - 1)
    return last["cx"] + vx * dt * damp, last["cy"] + vy * dt * damp


def link_and_filter_trajectories(
    candidates_by_frame,
    max_gap=8,
    max_speed_px=40.0,
    base_radius=16.0,
    min_track_len=5,
    window_size=45,
):
    """
    Link ball candidates using a tight sliding-window trajectory gate and
    remove isolated/outlier detections so unwanted boxes are never merged.
    Returns (cleaned_by_frame, track_stats).
    """
    active_tracks = []
    finished_tracks = []
    next_track_id = 1

    sorted_frames = sorted(candidates_by_frame.keys())
    for fn in sorted_frames:
        # Expire stale active tracklets outside the sliding window
        still_active = []
        for tr in active_tracks:
            if fn - tr["last_frame"] <= max_gap:
                still_active.append(tr)
            else:
                finished_tracks.append(tr)
        active_tracks = still_active

        dets = candidates_by_frame[fn]
        pairs = []
        for ti, tr in enumerate(active_tracks):
            last = tr["points"][-1]
            dt = fn - last["frame"]
            if dt <= 0 or dt > max_gap:
                continue

            # Tighter gating across gaps so unrelated detections are never merged
            if dt > 1 and len(tr["points"]) < 3:
                continue

            px, py = predict_track_position(tr["points"], fn, max_speed_px=max_speed_px)
            max_dist_last = base_radius + max_speed_px * dt
            max_dist_pred = 22.0 + 14.0 * dt

            for di, det in enumerate(dets):
                dist_last = math.hypot(det["cx"] - last["cx"], det["cy"] - last["cy"])
                if dist_last > max_dist_last:
                    continue
                dist_pred = math.hypot(det["cx"] - px, det["cy"] - py)
                if len(tr["points"]) >= 3 and dist_pred > max_dist_pred:
                    continue
                cost = (0.7 * dist_pred + 0.3 * dist_last) if len(tr["points"]) >= 2 else dist_last
                cost -= 5.0 * det["conf"]
                pairs.append((cost, ti, di))

        pairs.sort(key=lambda x: x[0])
        matched_tracks = set()
        matched_dets = set()

        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            active_tracks[ti]["points"].append(dets[di])
            active_tracks[ti]["last_frame"] = fn
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Unmatched detections start separate candidate tracklets
        for di, det in enumerate(dets):
            if di not in matched_dets:
                active_tracks.append({
                    "id": next_track_id,
                    "last_frame": fn,
                    "points": [det],
                })
                next_track_id += 1

    all_tracks = finished_tracks + active_tracks

    # Keep only continuous trajectories (length >= min_track_len)
    valid_tracks = []
    dropped_short = 0
    for tr in all_tracks:
        if len(tr["points"]) < min_track_len:
            dropped_short += 1
            continue
        valid_tracks.append(tr)

    # Resolve any frame with >1 valid tracklet by preferring longer tracklet, then confidence
    cleaned_by_frame = {}
    for tr in valid_tracks:
        tr_len = len(tr["points"])
        for det in tr["points"]:
            fn = det["frame"]
            entry = {
                **det,
                "track_id": tr["id"],
                "track_len": tr_len,
            }
            prev = cleaned_by_frame.get(fn)
            if prev is None or (tr_len, det["conf"]) > (prev["track_len"], prev["conf"]):
                cleaned_by_frame[fn] = entry

    # Sliding-window check (window_size > 30):
    # 1) There is never a second ball in play: if a short tracklet (< 15 dets) appears far
    #    (> 120 px) from a longer dominant tracklet inside the same sliding window, drop it.
    # 2) Reject any remaining detections inconsistent (> 90 px) with window neighbors.
    half_w = max(2, window_size // 2)
    kept_frames_sorted = sorted(cleaned_by_frame.keys())
    outliers_removed = 0

    # Pass 1: remove short stray tracklets that coexist with a dominant tracklet in +-half_w
    track_pts = {}
    for fn in kept_frames_sorted:
        e = cleaned_by_frame[fn]
        track_pts.setdefault(e["track_id"], []).append((fn, e["cx"], e["cy"], e["track_len"]))

    stray_track_ids = set()
    for tid, pts in track_pts.items():
        tr_len = pts[0][3]
        if tr_len >= 15:
            continue
        t_min, t_max = pts[0][0] - half_w, pts[-1][0] + half_w
        dom_pts = []
        for fn in kept_frames_sorted:
            if fn < t_min:
                continue
            if fn > t_max:
                break
            e = cleaned_by_frame[fn]
            if e["track_id"] != tid and e["track_len"] >= max(15, 2 * tr_len):
                dom_pts.append((e["cx"], e["cy"]))
        if len(dom_pts) >= 3:
            dom_x = float(np.median([p[0] for p in dom_pts]))
            dom_y = float(np.median([p[1] for p in dom_pts]))
            cur_x = float(np.median([p[1] for p in pts]))
            cur_y = float(np.median([p[2] for p in pts]))
            if math.hypot(cur_x - dom_x, cur_y - dom_y) > 120.0:
                stray_track_ids.add(tid)

    if stray_track_ids:
        for fn in list(cleaned_by_frame.keys()):
            if cleaned_by_frame[fn]["track_id"] in stray_track_ids:
                del cleaned_by_frame[fn]
                outliers_removed += 1
        kept_frames_sorted = sorted(cleaned_by_frame.keys())

    # Pass 2: full sliding-window spike check across [fn - half_w, fn + half_w].
    # Keep frames that smoothly continue along a fast trajectory (connected to
    # an immediate neighbor within <= 4 frames at <= 45 px/frame), and only
    # drop isolated spikes that jump > 90 px away from the window median.
    to_remove = []
    n_kept = len(kept_frames_sorted)
    for idx, fn in enumerate(kept_frames_sorted):
        cur = cleaned_by_frame[fn]
        smooth_link = False
        if idx > 0:
            prev_fn = kept_frames_sorted[idx - 1]
            dt = fn - prev_fn
            if dt <= 4:
                prev_d = cleaned_by_frame[prev_fn]
                if math.hypot(cur["cx"] - prev_d["cx"], cur["cy"] - prev_d["cy"]) <= 45.0 * dt:
                    smooth_link = True
        if not smooth_link and idx + 1 < n_kept:
            next_fn = kept_frames_sorted[idx + 1]
            dt = next_fn - fn
            if dt <= 4:
                next_d = cleaned_by_frame[next_fn]
                if math.hypot(cur["cx"] - next_d["cx"], cur["cy"] - next_d["cy"]) <= 45.0 * dt:
                    smooth_link = True
        if smooth_link:
            continue

        neighbors = []
        j = idx - 1
        while j >= 0 and (fn - kept_frames_sorted[j]) <= half_w:
            neighbors.append(cleaned_by_frame[kept_frames_sorted[j]])
            j -= 1
        j = idx + 1
        while j < n_kept and (kept_frames_sorted[j] - fn) <= half_w:
            neighbors.append(cleaned_by_frame[kept_frames_sorted[j]])
            j += 1
        if len(neighbors) >= 3:
            med_x = float(np.median([nb["cx"] for nb in neighbors]))
            med_y = float(np.median([nb["cy"] for nb in neighbors]))
            if math.hypot(cur["cx"] - med_x, cur["cy"] - med_y) > 90.0:
                to_remove.append(fn)
    for fn in to_remove:
        del cleaned_by_frame[fn]
    outliers_removed += len(to_remove)

    track_stats = {
        "total_tracklets": len(all_tracks),
        "valid_tracklets": len(valid_tracks),
        "dropped_short_tracklets": dropped_short,
        "outliers_removed": outliers_removed,
        "kept_frames": len(cleaned_by_frame),
    }
    return cleaned_by_frame, track_stats


def write_cleaned_csv(out_csv, cleaned_by_frame, start_frame, end_frame, img_w=1920, img_h=1080, write_empty=True):
    """Write cleaned ball detections to CSV with the standard schema."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    header = [
        "frame_number", "ball_detected", "confidence",
        "x_center", "y_center", "x1", "y1", "x2", "y2",
        "width", "height", "x_center_norm", "y_center_norm",
    ]
    rows_written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fn in range(start_frame, end_frame + 1):
            det = cleaned_by_frame.get(fn)
            if det is not None:
                cx_norm = min(1.0, max(0.0, det["cx"] / img_w))
                cy_norm = min(1.0, max(0.0, det["cy"] / img_h))
                writer.writerow([
                    fn, 1, f"{det['conf']:.4f}",
                    f"{det['cx']:.2f}", f"{det['cy']:.2f}",
                    f"{det['x1']:.2f}", f"{det['y1']:.2f}",
                    f"{det['x2']:.2f}", f"{det['y2']:.2f}",
                    f"{det['w']:.2f}", f"{det['h']:.2f}",
                    f"{cx_norm:.6f}", f"{cy_norm:.6f}",
                ])
                rows_written += 1
            elif write_empty:
                writer.writerow([fn, 0, "", "", "", "", "", "", "", "", "", "", ""])
                rows_written += 1
    return rows_written


def render_cleaned_video(video_path, out_video, cleaned_by_frame, start_frame, end_frame, fps=25, width=1920, height=1080):
    """Render [start_frame..end_frame] from video_path with cleaned ball overlay to out_video."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)

    num_frames = end_frame - start_frame + 1
    start_sec = start_frame / float(fps)
    hh = int(start_sec // 3600)
    mm = int((start_sec % 3600) // 60)
    ss = start_sec % 60.0
    seek_str = f"{hh:02d}:{mm:02d}:{ss:06.3f}"

    decode_cmd = [
        "ffmpeg", "-y",
        "-ss", seek_str,
        "-i", video_path,
        "-vframes", str(num_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "20",
        out_video,
    ]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    dec_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = width * height * 3
    trail = deque(maxlen=15)
    rendered = 0

    try:
        for offset in range(num_frames):
            raw = dec_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            fn = start_frame + offset

            img = Image.frombuffer("RGB", (width, height), raw, "raw", "RGB", 0, 1).copy()
            draw = ImageDraw.Draw(img)

            det = cleaned_by_frame.get(fn)
            if det is not None:
                trail.append((fn, det["cx"], det["cy"]))

            # Draw recent trajectory trail (only points within 15 frames)
            active_trail = [(f, x, y) for (f, x, y) in trail if fn - f <= 15]
            for idx in range(1, len(active_trail)):
                _, x0, y0 = active_trail[idx - 1]
                _, x1, y1 = active_trail[idx]
                draw.line([(x0, y0), (x1, y1)], fill=(0, 255, 255), width=2)

            if det is not None:
                pad = 4
                draw.rectangle(
                    [det["x1"] - pad, det["y1"] - pad, det["x2"] + pad, det["y2"] + pad],
                    outline=(255, 230, 0),
                    width=3,
                )
                r = 3
                draw.ellipse(
                    [det["cx"] - r, det["cy"] - r, det["cx"] + r, det["cy"] + r],
                    fill=(255, 60, 60),
                )
                status = f"Frame: {fn} | Ball: CLEANED | Track #{det['track_id']} (len={det['track_len']}) | Conf: {det['conf']:.2f}"
                hud_color = (255, 230, 0)
            else:
                status = f"Frame: {fn} | Ball: SKIPPED (aerial / no trajectory)"
                hud_color = (180, 180, 180)

            draw.rectangle([16, 16, 860, 56], fill=(15, 15, 15))
            draw.text((28, 24), status, fill=hud_color, font=font)

            enc_proc.stdin.write(img.tobytes())
            rendered += 1
    finally:
        if dec_proc.stdout:
            dec_proc.stdout.close()
        if enc_proc.stdin:
            enc_proc.stdin.close()
        dec_proc.wait()
        enc_proc.wait()

    if enc_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoder failed with exit code {enc_proc.returncode}")
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Clean ball CSV via trajectory continuity and render video.")
    parser.add_argument("--csv", required=True, help="Input ball detections CSV path.")
    parser.add_argument("--out-csv", required=True, help="Output cleaned ball CSV path.")
    parser.add_argument("--data-dir", default=None, help="Custom data directory (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--game", default=DEFAULT_GAME, help=f"Game identifier in games.py (default: {DEFAULT_GAME})")
    parser.add_argument("--video", default=None, help="Source tactical video path.")
    parser.add_argument("--homography", default=None, help="Homography CSV path (pass '' to disable).")
    parser.add_argument("--pitch-margin", type=float, default=0.02, help="Normalized pitch margin outside [0, 1].")
    parser.add_argument("--out-video", default=None, help="Output annotated MP4 path (omit to skip video).")
    parser.add_argument("--start-frame", type=int, default=None, help="Override start frame (default: CSV min frame).")
    parser.add_argument("--end-frame", type=int, default=None, help="Override end frame (default: CSV max frame).")
    parser.add_argument("--min-conf", type=float, default=0.35, help="Minimum ball detection confidence.")
    parser.add_argument("--min-size", type=float, default=3.0, help="Minimum ball box width/height (px).")
    parser.add_argument("--max-size", type=float, default=28.0, help="Maximum ball box width/height (px).")
    parser.add_argument("--max-gap", type=int, default=8, help="Max missing frames to bridge in a trajectory.")
    parser.add_argument("--max-speed", type=float, default=40.0, help="Max ball speed in px/frame.")
    parser.add_argument("--min-track-len", type=int, default=5, help="Minimum detections to keep a trajectory.")
    parser.add_argument("--window-size", type=int, default=45, help="Sliding window size (frames) for outlier rejection.")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS.")
    args = parser.parse_args()

    if args.video is None or args.homography is None:
        if _games is not None:
            t_paths = _games.asset_paths(game=args.game, data_dir=args.data_dir, source="tactical")
            if args.video is None:
                args.video = t_paths["video"]
            if args.homography is None:
                args.homography = t_paths["homography"]
        else:
            ddir = args.data_dir or DEFAULT_DATA_DIR
            if args.video is None:
                args.video = os.path.join(ddir, "tactical", f"{args.game}_tactical.mp4")
            if args.homography is None:
                args.homography = os.path.join(ddir, "tactical", f"{args.game}_tactical_homography.csv")

    homographies = load_homographies(args.homography) if args.homography else None
    if homographies is not None:
        print(f"Loaded homographies for {len(homographies)} valid frames from {args.homography} (margin={args.pitch_margin})")

    candidates, csv_min_f, csv_max_f, stats = load_ball_candidates(
        args.csv,
        min_conf=args.min_conf,
        min_size=args.min_size,
        max_size=args.max_size,
        homographies=homographies,
        pitch_margin=args.pitch_margin,
    )
    if csv_min_f is None or csv_max_f is None:
        print("Error: No valid frames found in CSV.", file=sys.stderr)
        sys.exit(1)

    start_f = args.start_frame if args.start_frame is not None else csv_min_f
    end_f = args.end_frame if args.end_frame is not None else csv_max_f

    print(f"Input CSV: {args.csv}")
    print(f"  CSV frame range: {csv_min_f} .. {csv_max_f} (render range: {start_f} .. {end_f})")
    print(f"  Total rows: {stats['total_rows']} | Detected rows: {stats['detected_rows']}")
    print(
        f"  Rejected -> conf < {args.min_conf}: {stats['rejected_conf']} | "
        f"size not in [{args.min_size}, {args.max_size}]: {stats['rejected_size']} | "
        f"outside pitch homography: {stats['rejected_outside_pitch']}"
    )
    print(f"  Candidate ball frames: {stats['candidate_frames']} | Candidate boxes: {stats['candidate_boxes']}")

    if stats["candidate_boxes"] == 0:
        print(
            "Warning: 0 ball-sized candidates remained after size/confidence filtering! "
            "Check that your CSV actually contains ball detections (not only player boxes).",
            file=sys.stderr,
        )

    cleaned_by_frame, track_stats = link_and_filter_trajectories(
        candidates,
        max_gap=args.max_gap,
        max_speed_px=args.max_speed,
        min_track_len=args.min_track_len,
        window_size=args.window_size,
    )
    print(f"Trajectory cleaning:")
    print(
        f"  Total tracklets: {track_stats['total_tracklets']} | "
        f"Valid (>={args.min_track_len} dets): {track_stats['valid_tracklets']} | "
        f"Dropped short: {track_stats['dropped_short_tracklets']} | "
        f"Window outliers removed: {track_stats['outliers_removed']}"
    )
    print(f"  Cleaned ball frames kept: {track_stats['kept_frames']}")

    rows_written = write_cleaned_csv(args.out_csv, cleaned_by_frame, start_f, end_f)
    print(f"Wrote cleaned CSV: {args.out_csv} ({rows_written} rows)")

    if args.out_video:
        print(f"Rendering video {start_f} .. {end_f} -> {args.out_video} ...")
        rendered = render_cleaned_video(
            args.video, args.out_video, cleaned_by_frame, start_f, end_f, fps=args.fps
        )
        print(f"Rendered {rendered} frames to {args.out_video}")


if __name__ == "__main__":
    main()
