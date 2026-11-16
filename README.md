# Football Possession Tracking (`Matra`)

This repository predicts frame-by-frame ball possession (**H** = Home, **A** = Away, **D** = Dead Ball, **X** = Ungraded) from tactical-camera match data.

For full methodology, training data, rules, and visual tables, see [Matra.md](Matra.md) and [Matra.pdf](Matra.pdf).

### Latest Score (`evaluation/results/Matra.csv`)

| Metric | Matra (Final) | Ground Truth |
|---|---:|---:|
| **Macro F1** | **0.9558** | 1.0000 |
| **Spell F1** | **0.8485** | 1.0000 |
| **Frame Accuracy** | **95.80%** | 100.0% |
| **Transition F1** | **0.8205** | 1.0000 |
| **Possessions / Mean Duration** | **16** (`15.4 s`) | 17 (`14.2 s`) |
| **Mean Latency / Share Error** | **`-1.4` frames** / `0.003` | `+0.0` / `0.000` |

---

## Step 1: Create Environment & Install Dependencies

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install all required packages (includes ultralytics, torch, opencv-python, numpy, pandas, pillow)
pip install -r requirements.txt
```

---

## Step 2: Add Match Data

Put your tactical match folder inside `./data/tactical/` (or keep it anywhere and pass `--data-dir /path/to/your/data` to the commands below):

```text
data/
└── tactical/
    ├── <MATCH>_tactical.mp4
    ├── <MATCH>_tactical_atd.xml
    ├── <MATCH>_tactical_atd.txt
    ├── <MATCH>_tactical_homography.csv
    └── <MATCH>_tactical_smart_tagging.xml
```

*(You can also set `export MTAH_DATA_DIR=/path/to/your/data` instead of passing `--data-dir`.)*

---

## Step 3: Ball Detection (`detect_ball_play_time.py`)

### Case A — Using the Provided Tactical Match (Already Done)
We already ran ball detection on the provided tactical video and included the cleaned file inside the repo at **`src/ball_detections_cleaned.json`**.  
If you are running on that match, you can **skip Step 3** and go straight to **Step 4**.

### Case B — Running on a Fresh / New Video or Replaced `data/` Folder
Run ball detection + cleaning on your new video (uses the weights in `model/best.pt` automatically):

```bash
# Recommended: Run on GPU (--device 0) and adjust --batch-size (e.g., 16, 32, or 64) based on your GPU VRAM size
python3 detect_ball_play_time.py --device 0 --batch-size 32

# If using a custom data folder path:
python3 detect_ball_play_time.py --data-dir /path/to/your/data --device 0 --batch-size 32
```

> **Note:** Running on **GPU** and choosing a larger `--batch-size` that fits your **GPU VRAM** (e.g., `16`, `32`, or `64`) runs much faster. Running on **CPU (`--device cpu`) will take a long time** on full-match video.

This automatically saves `<data-dir>/ball_detections_cleaned.json`.

---

## Step 4: Generate Final Possession Predictions

```bash
# Default (reads ./data/ and saves evaluation/results/Matra.csv):
python3 build_submission.py

# Custom data folder (and custom FPS if video is not 25 FPS, e.g., --fps 30):
python3 build_submission.py --data-dir /path/to/your/data
```

### How `build_submission.py` Automatically Finds the Clean Ball JSON
1. If you ran Step 3 on a new `data/` folder, it automatically picks up **`<data-dir>/ball_detections_cleaned.json`**.
2. If `<data-dir>/ball_detections_cleaned.json` is not present (for example, when skipping Step 3 on the provided match), it automatically falls back to **`src/ball_detections_cleaned.json`** inside the repo.
3. You can also override it manually anytime with `--ball-json /path/to/ball_detections_cleaned.json`.

Output is saved to **`evaluation/results/Matra.csv`**.

---

## Step 5: Validate & Get Scores

```bash
# 1. Check CSV format and label counts
python3 evaluation/validate_submission.py evaluation/results/Matra.csv

# 2. Score against public 5-minute ground truth
python3 evaluation/run_evaluation.py --no-detail

# (Organizers scoring full match private GT):
python3 evaluation/run_evaluation.py --gt private --no-detail
```

---

## Step 6 (Optional): Render Annotated Match Videos

```bash
python3 render_match_half_video.py --half 1 --pred evaluation/results/Matra.csv --no-atd-ball --track-near-player-gaps
python3 render_match_half_video.py --half 2 --pred evaluation/results/Matra.csv --no-atd-ball --track-near-player-gaps
```
