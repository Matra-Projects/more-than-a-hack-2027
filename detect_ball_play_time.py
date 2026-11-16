#!/usr/bin/env python3
"""
Detect ball in tactical football match video using YOLO and export detections to a single JSON.

Key Features:
- Reads play time spans dynamically from the match XML:
  * Supports Smart Tagging XML (data/tactical/BAR-ATH_tactical_smart_tagging.xml)
  * Supports games.py official half blocks as an option or fallback
  * Allows selecting specific half ('1st Half', '2nd Half', 'all')
- Processes only frames within play time ranges, skipping pre-match, halftime, and post-match
- Computes ball detections:
  * Bounding box in absolute pixel coordinates: x1, y1, x2, y2, width, height
  * Centroid in absolute pixel coordinates: (cx, cy)
  * Centroid normalized to [0.0, 1.0]: (cx_norm, cy_norm)
  * Confidence score
- Saves all detections into a single JSON file structured by frame number
- Video output generation is disabled by default (WRITE_VIDEO = False)
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET

# Lazy imports for heavy ML / CV dependencies so utilities can be tested without GPU/cv2
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    torch = None
    np = None
    YOLO = None

# ================= DEFAULT CONFIGURATION =================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    import games as _games
    DEFAULT_GAME = _games.GAME
    DEFAULT_DATA_DIR = _games.DATA_DIR
except Exception:
    _games = None
    DEFAULT_GAME = "BAR-ATH"
    DEFAULT_DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO_ROOT, "data")

_LOCAL_MODEL = os.path.join(REPO_ROOT, "model", "best.pt")
_REPO_MODEL_FALLBACK = os.path.join(REPO_ROOT, "yolo-football-ball-detection.pt")
DEFAULT_MODEL_PATH = _LOCAL_MODEL if os.path.isfile(_LOCAL_MODEL) else _REPO_MODEL_FALLBACK


def resolve_dynamic_paths(data_dir=None, game=DEFAULT_GAME):
    """Resolve tactical video, XML, homography, and output JSON/video paths dynamically."""
    ddir = data_dir or DEFAULT_DATA_DIR
    if _games is not None:
        paths = _games.asset_paths(game=game, data_dir=ddir, source="tactical")
        video_path = paths["video"]
        xml_path = paths["smart_tagging"]
        homography_csv = paths["homography"]
    else:
        tactical_dir = os.path.join(ddir, "tactical")
        video_path = os.path.join(tactical_dir, f"{game}_tactical.mp4")
        xml_path = os.path.join(tactical_dir, f"{game}_tactical_smart_tagging.xml")
        homography_csv = os.path.join(tactical_dir, f"{game}_tactical_homography.csv")
    vstem, _ = os.path.splitext(os.path.basename(video_path))
    return {
        "data_dir": ddir,
        "video": video_path,
        "xml": xml_path,
        "homography": homography_csv,
        "output_json": os.path.join(ddir, "ball_detections.json"),
        "output_cleaned_json": os.path.join(ddir, "ball_detections_cleaned.json"),
        "output_video": os.path.join(os.path.dirname(video_path), f"{vstem}_ball_detected.mp4"),
    }


_DYN_DEFAULTS = resolve_dynamic_paths()
DEFAULT_VIDEO_SOURCE = _DYN_DEFAULTS["video"]
DEFAULT_XML_PATH = _DYN_DEFAULTS["xml"]
DEFAULT_OUTPUT_JSON = _DYN_DEFAULTS["output_json"]
DEFAULT_OUTPUT_CLEANED_JSON = _DYN_DEFAULTS["output_cleaned_json"]
DEFAULT_HOMOGRAPHY_CSV = _DYN_DEFAULTS["homography"]
DEFAULT_OUTPUT_VIDEO = _DYN_DEFAULTS["output_video"]

# Set WRITE_VIDEO to False by default as requested
WRITE_VIDEO_DEFAULT = False

BATCH_SIZE_DEFAULT = 32
FRAME_STRIDE_DEFAULT = 1

# Classes available in model: 0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"
CLASS_NAMES = ['player', 'goalkeeper', 'referee', 'ball']
BALL_CLS_ID = 0

# Distinct BGR colors for drawing if video output is enabled
CLASS_COLORS = {
    0: (0, 255, 255),   # ball: Yellow
    1: (0, 165, 255),   # goalkeeper: Orange
    2: (0, 255, 0),     # player: Green
    3: (0, 0, 255),     # referee: Red
}

# Per-class confidence thresholds
CLASS_CONF_THRESH = {
    0: 0.50,  # ball threshold
    1: 0.65,  # goalkeeper
    2: 0.65,  # player
    3: 0.45,  # ball
}

# Duplicate Box Merging Settings
MERGE_CENTROID_DIST = 18.0  # Max pixel distance between two box centers to merge
MERGE_IOU_THRESH = 0.45     # Or if IoU overlap is greater than this, merge them
WEIGHTED_MERGE = True       # True = Fuse overlapping boxes by confidence | False = Keep highest conf box
# =========================================================


def parse_play_spans_from_xml(xml_path, fps=25.0, mode="all"):
    """
    Parses start and end play frames directly from the Smart Tagging XML file.
    
    Modes:
      - '1st Half': Returns only 1st Half range
      - '2nd Half': Returns only 2nd Half range
      - 'all': Returns both 1st and 2nd Half ranges
      - 'official': Uses games.py / challenge ground-truth blocks (11185..90651, 110808..190197)
    """
    if mode == "official":
        if _games is not None:
            return [
                {"period": "1st Half" if h == 1 else "2nd Half", "start_frame": s, "end_frame": e}
                for h, s, e in _games.half_video_blocks(DEFAULT_GAME)
            ]
        return [
            {"period": "1st Half", "start_frame": 11185, "end_frame": 90651},
            {"period": "2nd Half", "start_frame": 110808, "end_frame": 190197}
        ]

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Smart tagging XML file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    spans_by_half = {"1st Half": [], "2nd Half": []}
    for inst in root.findall(".//instance"):
        st = float(inst.findtext("start", "0"))
        en = float(inst.findtext("end", "0"))
        half = None
        for label in inst.findall("label"):
            if label.findtext("group") == "Half":
                half = label.findtext("text")
        if half in spans_by_half:
            spans_by_half[half].append((st, en))

    ranges = []
    for half_name in ["1st Half", "2nd Half"]:
        if mode != "all" and mode != half_name:
            continue
        half_spans = spans_by_half.get(half_name, [])
        if half_spans:
            min_st = min(s[0] for s in half_spans)
            max_en = max(s[1] for s in half_spans)
            start_frame = int(round(min_st * fps))
            end_frame = int(round(max_en * fps))
            ranges.append({
                "period": half_name,
                "start_time_sec": min_st,
                "end_time_sec": max_en,
                "start_frame": start_frame,
                "end_frame": end_frame
            })

    return ranges


def compute_iou(box1, box2):
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(area1 + area2 - inter_area + 1e-6)


def filter_and_merge_detections(boxes_obj, target_cls_id=BALL_CLS_ID, min_conf=0.50):
    """
    1. Filters boxes by target class and confidence threshold.
    2. Merges duplicate boxes on the same object using centroid distance & IoU.
    Returns list of dicts with box coordinates and confidence.
    """
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    xyxy = boxes_obj.xyxy.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    classes = boxes_obj.cls.cpu().numpy().astype(int)

    # Filter by target class (ball) and confidence
    valid_indices = [
        i for i, (cls_id, conf) in enumerate(zip(classes, confs))
        if cls_id == target_cls_id and conf >= min_conf
    ]

    if not valid_indices:
        return []

    xyxy = xyxy[valid_indices]
    confs = confs[valid_indices]

    # Sort detections from highest confidence to lowest
    order = np.argsort(-confs)
    xyxy = xyxy[order]
    confs = confs[order]

    used = np.zeros(len(xyxy), dtype=bool)
    merged_detections = []

    for i in range(len(xyxy)):
        if used[i]:
            continue

        used[i] = True
        base_box = xyxy[i]
        base_conf = float(confs[i])

        cx1 = (base_box[0] + base_box[2]) / 2.0
        cy1 = (base_box[1] + base_box[3]) / 2.0

        cluster_boxes = [base_box]
        cluster_confs = [base_conf]

        for j in range(i + 1, len(xyxy)):
            if used[j]:
                continue

            cand_box = xyxy[j]
            cx2 = (cand_box[0] + cand_box[2]) / 2.0
            cy2 = (cand_box[1] + cand_box[3]) / 2.0

            centroid_dist = np.hypot(cx1 - cx2, cy1 - cy2)
            iou = compute_iou(base_box, cand_box)

            if centroid_dist <= MERGE_CENTROID_DIST or iou >= MERGE_IOU_THRESH:
                used[j] = True
                cluster_boxes.append(cand_box)
                cluster_confs.append(float(confs[j]))

        if WEIGHTED_MERGE and len(cluster_boxes) > 1:
            weights = np.array(cluster_confs)
            weights /= weights.sum()
            final_box = np.average(np.array(cluster_boxes), axis=0, weights=weights)
            final_conf = max(cluster_confs)
        else:
            final_box = base_box
            final_conf = base_conf

        x1, y1, x2, y2 = final_box
        merged_detections.append({
            "confidence": round(float(final_conf), 4),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        })

    return merged_detections


def draw_ball_on_frame(frame, ball_dets):
    """Draws ball detections on video frame if video writer is enabled."""
    for det in ball_dets:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        conf = det["confidence"]
        label = f"ball {conf:.2f}"

        # Draw yellow box and centroid circle
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 255), 2)
        cx, cy = int(round(det["centroid_pixel"]["x"])), int(round(det["centroid_pixel"]["y"]))
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Draw label badge
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(iy1, th + 4)
        cv2.rectangle(frame, (ix1, text_y - th - 4), (ix1 + tw + 4, text_y), (0, 255, 255), -1)
        cv2.putText(frame, label, (ix1 + 2, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def clean_ball_results_json(
    results_json,
    output_cleaned_json,
    homography_csv=DEFAULT_HOMOGRAPHY_CSV,
    pitch_margin=0.02,
    min_conf=0.35,
    min_size=3.0,
    max_size=28.0,
    max_gap=8,
    max_speed=40.0,
    min_track_len=5,
    window_size=45,
):
    """
    Clean raw ball detections in results_json using homography pitch filtering
    and sliding-window trajectory linking, then write output_cleaned_json.
    """
    from clean_and_render_ball import (
        is_inside_pitch,
        is_valid_ball_size,
        link_and_filter_trajectories,
        load_homographies,
    )

    homographies = load_homographies(homography_csv) if homography_csv else None
    img_w = results_json.get("metadata", {}).get("image_size", {}).get("width", 1920)
    img_h = results_json.get("metadata", {}).get("image_size", {}).get("height", 1080)

    candidates = {}
    rejected_conf = 0
    rejected_size = 0
    rejected_outside = 0

    for f_str, dets in results_json.get("frames", {}).items():
        fn = int(f_str)
        H_inv = homographies.get(fn) if homographies is not None else None
        kept_list = []
        for det in dets:
            conf = float(det.get("confidence", 0.0))
            if conf < min_conf:
                rejected_conf += 1
                continue
            w = float(det.get("bbox_size", {}).get("width", 0.0))
            h = float(det.get("bbox_size", {}).get("height", 0.0))
            if not is_valid_ball_size(w, h, min_size=min_size, max_size=max_size):
                rejected_size += 1
                continue
            cx = float(det.get("centroid_pixel", {}).get("x", 0.0))
            cy = float(det.get("centroid_pixel", {}).get("y", 0.0))
            if H_inv is not None and not is_inside_pitch(H_inv, cx, cy, margin=pitch_margin):
                rejected_outside += 1
                continue
            x1, y1, x2, y2 = det.get("bbox_xyxy", [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
            kept_list.append({
                "frame": fn,
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "w": w,
                "h": h,
            })
        if kept_list:
            candidates[fn] = kept_list

    cleaned_by_frame, track_stats = link_and_filter_trajectories(
        candidates,
        max_gap=max_gap,
        max_speed_px=max_speed,
        min_track_len=min_track_len,
        window_size=window_size,
    )

    cleaned_frames = {}
    for f_str in results_json.get("frames", {}):
        fn = int(f_str)
        det = cleaned_by_frame.get(fn)
        if det is None:
            cleaned_frames[f_str] = []
        else:
            cleaned_frames[f_str] = [{
                "confidence": round(det["conf"], 4),
                "bbox_xyxy": [round(det["x1"], 2), round(det["y1"], 2), round(det["x2"], 2), round(det["y2"], 2)],
                "bbox_size": {"width": round(det["w"], 2), "height": round(det["h"], 2)},
                "centroid_pixel": {"x": round(det["cx"], 2), "y": round(det["cy"], 2)},
                "centroid_norm": {
                    "x": round(min(1.0, max(0.0, det["cx"] / img_w)), 6),
                    "y": round(min(1.0, max(0.0, det["cy"] / img_h)), 6),
                },
            }]

    cleaned_json = {
        "metadata": {
            **results_json.get("metadata", {}),
            "cleaned": True,
            "homography_source": homography_csv,
            "pitch_margin": pitch_margin,
            "window_size": window_size,
            "cleaning_stats": {
                "rejected_conf": rejected_conf,
                "rejected_size": rejected_size,
                "rejected_outside_pitch": rejected_outside,
                **track_stats,
            },
        },
        "play_spans": results_json.get("play_spans", []),
        "frames": cleaned_frames,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_cleaned_json)), exist_ok=True)
    with open(output_cleaned_json, "w", encoding="utf-8") as cf:
        json.dump(cleaned_json, cf, indent=2)

    print("\n" + "=" * 70)
    print(f"CLEANED BALL JSON SAVED: {output_cleaned_json}")
    print(
        f"  Rejected -> conf: {rejected_conf:,} | size: {rejected_size:,} | "
        f"outside pitch: {rejected_outside:,}"
    )
    print(
        f"  Tracklets: {track_stats['total_tracklets']:,} | Valid: {track_stats['valid_tracklets']:,} | "
        f"Window outliers removed: {track_stats['outliers_removed']:,} | "
        f"Kept frames: {track_stats['kept_frames']:,}"
    )
    print("=" * 70)
    return cleaned_json


def process_play_time_ranges(
    model_path,
    video_source,
    xml_path,
    output_json,
    output_cleaned_json=None,
    homography_csv=DEFAULT_HOMOGRAPHY_CSV,
    pitch_margin=0.02,
    window_size=45,
    output_video=None,
    write_video=False,
    period_mode="all",
    batch_size=32,
    frame_stride=1,
    conf_thresh=0.50,
    device=0
):
    print("=" * 70)
    print("Football Ball Detection — Play Time Processor")
    print(f"  Video Source:   {video_source}")
    print(f"  Smart XML:      {xml_path}")
    print(f"  Model Path:     {model_path}")
    print(f"  Output JSON:    {output_json}")
    print(f"  Write Video:    {write_video} (Output: {output_video if write_video else 'Disabled'})")
    print(f"  Period Mode:    {period_mode}")
    print(f"  Batch Size:     {batch_size} | Stride: {frame_stride} | Conf Thresh: {conf_thresh}")
    print("=" * 70)

    # 1. Open Video Capture first so play-span XML timestamps use the video's actual FPS
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_source}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Properties: {img_w}x{img_h} @ {fps:.2f} fps | Total Frames: {total_video_frames:,}")

    # 2. Parse play time ranges from XML at the video's FPS
    play_ranges = parse_play_spans_from_xml(xml_path, fps=fps, mode=period_mode)
    print(f"Extracted {len(play_ranges)} play time span(s) from XML:")
    for r in play_ranges:
        print(f"  - {r['period']}: Frames {r['start_frame']:,} -> {r['end_frame']:,} "
              f"({r['end_frame'] - r['start_frame'] + 1:,} frames)")

    # 3. Setup Video Writer if enabled
    writer = None
    if write_video and output_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        out_fps = fps / frame_stride if frame_stride > 1 else fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, out_fps, (img_w, img_h))

    # 4. Load YOLO Model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    ball_cls_id = BALL_CLS_ID
    if hasattr(model, "names") and isinstance(model.names, dict):
        for idx, name in model.names.items():
            if str(name).strip().lower() == "ball":
                ball_cls_id = int(idx)
                break
    print(f"Using ball class ID: {ball_cls_id}")
    actual_device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {actual_device} (CUDA Available: {torch.cuda.is_available()})")

    # Output storage
    # Structure:
    # {
    #   "metadata": { ... },
    #   "play_spans": [ ... ],
    #   "frames": {
    #       "<frame_num>": [ { "confidence": ..., "bbox_xyxy": [...], "centroid_pixel": {...}, "centroid_norm": {...} } ]
    #   }
    # }
    results_json = {
        "metadata": {
            "video_source": video_source,
            "xml_source": xml_path,
            "model_path": model_path,
            "image_size": {"width": img_w, "height": img_h},
            "fps": fps,
            "frame_stride": frame_stride,
            "confidence_threshold": conf_thresh,
            "period_mode": period_mode,
        },
        "play_spans": play_ranges,
        "frames": {}
    }

    total_frames_to_process = sum(
        ((min(total_video_frames - 1, r["end_frame"]) - r["start_frame"]) // frame_stride) + 1
        for r in play_ranges
    )
    processed_count = 0
    ball_frames_count = 0

    # 5. Process each play time range
    for r_idx, span in enumerate(play_ranges):
        start_frame = max(0, span["start_frame"])
        end_frame = min(total_video_frames - 1, span["end_frame"])
        period_name = span["period"]

        print(f"\nProcessing {period_name}: Frames {start_frame:,} to {end_frame:,}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        frames_batch = []
        frame_nums_batch = []

        while cap.isOpened() and current_frame <= end_frame:
            if (current_frame - start_frame) % frame_stride == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)
                frame_nums_batch.append(current_frame)
            else:
                ret = cap.grab()
                if not ret:
                    break

            current_frame += 1

            if len(frames_batch) > 0 and (len(frames_batch) == batch_size or current_frame > end_frame):
                results = model.predict(
                    source=frames_batch,
                    batch=len(frames_batch),
                    conf=conf_thresh,
                    imgsz=1280,
                    device=actual_device,
                    half=(actual_device != "cpu"),
                    verbose=False
                )

                for frame_img, f_num, result in zip(frames_batch, frame_nums_batch, results):
                    raw_dets = filter_and_merge_detections(result.boxes, target_cls_id=ball_cls_id, min_conf=conf_thresh)

                    frame_ball_list = []
                    for d in raw_dets:
                        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        cx_norm = float(np.clip(cx / img_w, 0.0, 1.0))
                        cy_norm = float(np.clip(cy / img_h, 0.0, 1.0))

                        frame_ball_list.append({
                            "confidence": d["confidence"],
                            "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                            "bbox_size": {"width": round(w, 2), "height": round(h, 2)},
                            "centroid_pixel": {"x": round(cx, 2), "y": round(cy, 2)},
                            "centroid_norm": {"x": round(cx_norm, 6), "y": round(cy_norm, 6)}
                        })

                    # Save to JSON dict (use string key for JSON standard)
                    results_json["frames"][str(f_num)] = frame_ball_list

                    if len(frame_ball_list) > 0:
                        ball_frames_count += 1

                    if writer is not None:
                        annotated = draw_ball_on_frame(frame_img, frame_ball_list)
                        writer.write(annotated)

                    processed_count += 1

                frames_batch.clear()
                frame_nums_batch.clear()

                if processed_count % 500 == 0 or current_frame > end_frame:
                    pct = (processed_count / max(1, total_frames_to_process)) * 100
                    print(f"  [Progress] Processed {processed_count:,}/{total_frames_to_process:,} frames ({pct:.1f}%) "
                          f"| Ball Detected in {ball_frames_count:,} frames")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nAnnotated video saved to: {output_video}")

    # 6. Save JSON output
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2)

    file_size_mb = os.path.getsize(output_json) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"SUCCESS! Saved Ball Detections to JSON: {output_json}")
    print(f"  Total Frames Processed: {processed_count:,}")
    print(f"  Frames with Ball:       {ball_frames_count:,} ({(ball_frames_count / max(1, processed_count))*100:.1f}%)")
    print(f"  JSON File Size:         {file_size_mb:.2f} MB")
    print("=" * 70)

    # 7. Clean ball detections using homography + sliding-window trajectory linking
    if output_cleaned_json is None:
        base, ext = os.path.splitext(output_json)
        output_cleaned_json = f"{base}_cleaned{ext or '.json'}"
    clean_ball_results_json(
        results_json=results_json,
        output_cleaned_json=output_cleaned_json,
        homography_csv=homography_csv,
        pitch_margin=pitch_margin,
        min_conf=min(0.35, conf_thresh),
        window_size=window_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Process match video ball detections within play time ranges defined in XML.")
    parser.add_argument("--data-dir", default=None, help="Custom data directory (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--game", default=DEFAULT_GAME, help=f"Game identifier in games.py (default: {DEFAULT_GAME})")
    parser.add_argument("--video", default=None, help="Path to video file (.mp4)")
    parser.add_argument("--xml", default=None, help="Path to smart tagging XML file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained YOLO model weights (.pt)")
    parser.add_argument("--output-json", default=None, help="Path to output JSON file")
    parser.add_argument("--output-cleaned-json", default=None, help="Path to output cleaned ball JSON file")
    parser.add_argument("--homography", default=None, help="Path to homography CSV for pitch filtering")
    parser.add_argument("--pitch-margin", type=float, default=0.02, help="Normalized pitch margin outside [0, 1]")
    parser.add_argument("--window-size", type=int, default=45, help="Sliding window size (frames) for outlier rejection")
    parser.add_argument("--output-video", default=None, help="Path to output video file")
    parser.add_argument("--write-video", action="store_true", default=WRITE_VIDEO_DEFAULT,
                        help="Enable annotated video writer (disabled by default)")
    parser.add_argument("--period", default="all", choices=["all", "1st Half", "2nd Half", "official"],
                        help="Which play period to process from XML ('all', '1st Half', '2nd Half', 'official')")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Inference batch size")
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE_DEFAULT, help="Frame stride (1 = every frame)")
    parser.add_argument("--conf", type=float, default=0.50, help="Ball confidence threshold")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")

    args = parser.parse_args()
    dyn = resolve_dynamic_paths(data_dir=args.data_dir, game=args.game)
    video_path = args.video or dyn["video"]
    xml_path = args.xml or dyn["xml"]
    homography_csv = args.homography or dyn["homography"]
    output_json = args.output_json or dyn["output_json"]
    output_cleaned_json = args.output_cleaned_json or dyn["output_cleaned_json"]
    output_video = args.output_video or dyn["output_video"]

    process_play_time_ranges(
        model_path=args.model,
        video_source=video_path,
        xml_path=xml_path,
        output_json=output_json,
        output_cleaned_json=output_cleaned_json,
        homography_csv=homography_csv,
        pitch_margin=args.pitch_margin,
        window_size=args.window_size,
        output_video=output_video,
        write_video=args.write_video,
        period_mode=args.period,
        batch_size=args.batch_size,
        frame_stride=args.stride,
        conf_thresh=args.conf,
        device=args.device
    )


if __name__ == "__main__":
    main()
