# Technical Document — Tactical Football Possession (`Matra`)

## 1. Approach

We built a **two-stage physics + sequence-decoding pipeline** (`Matra`) that separates **Dead Ball (`D`)** from **Live Play (`H` / `A`)** and runs exclusively on tactical-camera assets:

1. **Step 0 — Custom Ball Reconstruction**: Because the provided optical ATD ball track is missing on **41.73%** of graded frames and disagrees with true ball coordinates by **`> 2.0 m` on 26.00%** of shared frames, we run a trained **YOLO ball detector** across play-time spans, clean false detections via **homography pitch projection** and a **45-frame sliding-window trajectory filter** (`0.35%` jump outliers, `0` multi-ball frames), and fill short player occlusions (`<= 3.0 s`) via **near-player foot-offset tracking** (**66.14%** graded ball availability).
2. **Stage 1 — Dead-Ball Engine (`src/module_1.py` + Physical Boundary & Motion Guards)**: Locks `D` frames using physical boundary crossings, a **Slow Outward-Roll Guard**, **Ballistic Exit & Aerial Live-Play Guard**, Smart Tagging set pieces/goals, whole-field speed collapses (with open-play passing veto), a **Static Abandoned-Ball Guard**, and a **Long Unseen-Blackout Guard**.
3. **Stage 2 — Possession Physics & 3-State Viterbi HMM (`src/missing_ball_engine.py`, `src/hmm_engine.py`)**: Resolves loose balls, contested duels, flicks, GK box plays, and restarts on alive frames, then decodes globally coherent `H` / `A` spells with a **3.0 s minimum-spell Viterbi HMM** plus a **guarded $(\Delta\theta, \Delta v)$ diversion-kick rule**. All temporal windows scale dynamically with video **FPS**.

**Benchmark (`evaluation/results/Matra.csv`)**: `macroF1 = 0.9558`, `spellF1 = 0.8485`, `transF1 = 0.8205`, `lat_f = -1.4`, `poss = 16` (`15.4 s` mean), `shareE = 0.003`.

---

## 2. Assets Used

A `sys.addaudithook` guard in `build_submission.py` verifies **only tactical assets** are opened and aborts if `data/broadcast/` is accessed.

| Asset | Status | How Used (or Why Dropped) |
|---|---|---|
| **Tactical Video (`*_tactical*.mp4`)** | **Used** | Input to YOLO ball inference (`detect_ball_play_time.py`) and optional GPU verification rendering. |
| **Homography (`*_homography.csv`)** | **Used** | Projects YOLO boxes into normalized pitch coords `[0, 1]^2` (`2%` gate) and filters unreliable frames (`reproj > 0.8 / 1.0`). |
| **ATD Player Tracking (`*_atd.txt`, `*_atd.xml`)** | **Used** | Provides 46 player slots, pitch `(x, y)` positions, team IDs, and GK flags (with pass-flow goal-area reattribution). |
| **Smart Tagging (`*_smart_tagging.xml`)** | **Used** | Defines half play-time spans for YOLO inference and `SETPIECE` / `GOAL` anchors for Stage 1. |
| **ATD Optical Ball (in `*_atd.txt`)** | **Dropped (Primary)** | Only **53.10%** total / **58.27%** graded availability; deviates `> 2.0 m` on **26.00%** (`> 5.0 m` on **11.87%**) of shared frames. Retained **only** as an endpoint-consistent (`<= 2.5 m` at both ends) fallback for long gaps `> 3.0 s`. |
| **ATD `speed_kph` (3rd ball column)** | **Dropped** | Unusable (`r = -0.08` vs. displacement speed). Speed is derived directly from cleaned coordinates. |
| **Broadcast Source (`data/broadcast/`)** | **Dropped** | Never accessed or used at any stage. |

---

## 3. Training Data (Ball Detection Model)

- **Dataset Size**: **25,299 images** total.
- **Composition**: **90% simulated football match data** + **10% real-world broadcast/tactical footage** scraped from **FanCode** and **YouTube**.
- **Labelling Methodology & Annotators**: **98% of labels were generated automatically** from the simulation pipeline to minimize manual effort. Human annotators did not label every frame from scratch; instead, they performed **Quality Control (QC) and ball-box correction** on real-world and edge frames.
- **Zero Challenge Leakage**: **None** of the training or validation images/labels came from the provided challenge match (`BAR-ATH`).

---

## 4. Possession Rules (Loose Balls, Challenges, Restarts & Transitions)

Frames outside official half bounds are `X`; Stage-1 dead frames are locked to `D`; all remaining frames decode to `H` or `A`.

```
  [Alive H/A] --(Out >=0.35m/1.0m, Slow Roll, Ballistic, Setpiece, Goal, Stoppage, Static >=6.5s, Blackout >=10s)--> [Dead: D]
  [Dead:  D ] --(Restart strike > 4.0 m/s + field acceleration / Kickoff Hold / GK Restart)-----------------------> [Alive H/A]
```

### Loose Balls & Passes
- **Ground Transits & Quick Passes (`resolve_quick_pass_transits`, `resolve_ground_transit_touches`)**: Fast teammate passes (`> 5.0 m/s`, up to `75` frames) and pressed undeflected passes (`<= 45` frames, `cos >= 0.70`, `>= 6.0 m/s`) keep possession with the passing team even if the ball rolls near an opponent without stopping.
- **Aerial / Occluded Flight (`AIR_H/A -> LANDED_H/A`)**: Fast ball exits (`> 9.0 m/s`) simulate aerodynamic drag (`k = 0.013 /m`) and grass friction (`1.5 m/s^2`), holding the kicking team across flight until landing and receiver control.
- **Flight Dampening**: When `ball_s > 9.0 m/s`, player-proximity evidence is weighted by `0.15x` so defenders underneath an aerial pass do not trigger false turnovers.
- **Static Abandoned Balls (`--static-abandoned-dead`)**: Visible ball moving `<= 1.8 m` over `6.5 s` with `median(d_near) >= 1.5 m` marks `D`.
- **Slow Outward Roll (`--slow-out-roll-dead`)**: Visible `>= 0.35 m` outside `[0, 105.3] x [0, 68.0]` for `>= 0.12 s` followed by `>= 0.4 s` unseen marks `D` until restart.

### Challenges, Duels & Turnovers
- **Contested Duels (`resolve_duel_by_pass_destination`)**: When `>= 3` players from both teams crowd within `2.5 m`, per-frame noise is ignored and possession goes to the team that wins the first clean uncontested control afterward (`<= 2.0 m`, opponent `> 3.5 m`).
- **One-Touch Flicks (`resolve_one_touch_flicks`)**: Brief 1–2 frame deflections return to the original owner unless followed by controlled teammate receipt.
- **Interception Midpoint (`resolve_same_team_pass_midpoints`)**: Clean turnover followed immediately by a same-team pass switches at the midpoint `(t1 + t2) // 2`.
- **Diversion Kicks (`--diversion-kicks`)**: Sudden kick (`dTheta >= 40 deg` or `dV >= 8.0 m/s`, `s2 >= 8.0 m/s`, `disp >= 3.5 m`) by an uncontested player (`<= 1.25 m`, opp `>= 2.0 m`) switches possession if not reclaimed within `0.2–1.6 s`.
- **Goalkeeper Box Flow (`resolve_goalkeeper_inbound_outbound_passes`)**: Attributes inbound opponent shots (`kick_f .. claim_f - 1`) to the shooter's team and switches to the GK's team once secured (`claim_f`).
- **3-State Viterbi HMM (`hmm_engine.py`)**: Enforces a `4.30`-nat switch cost (`3.0 s` expected minimum spell) to prevent short flickers.

### Restarts (`D -> H / A`)
- **Set Pieces & Goals (`rule_setpiece`, `rule_goal`)**: Remains `D` until the ball is struck (`> 4.0 m/s`) and players accelerate into play.
- **Kickoff Hold (`resolve_kickoff_hold`)**: Holds possession for the kickoff taker's team in the center circle until the opponent clearly gains control or challenges inside `4.5 m`.
- **GK Restart Pass (`resolve_gk_restart_passes`)**: Goal kicks and keeper restarts to teammates are assigned to the goalkeeper's team.

---

## 5. Limitations & Future Improvements

### Current Weaknesses
1. **Wrong Player Team Marking**: Even when the ball is clearly detected, if a player on the pitch is marked with the wrong team label in the tracking data, possession gets assigned to the wrong team.
2. **Ball Position Marking Mismatch**: Small offsets between the predicted ball position and the actual ball location on the pitch (e.g., near touchlines or fast passes) can misalign dead-ball crossings and player-ball proximity.
3. **Lack of Depth Information**: Tactical 2D monocular homography projects everything onto a flat ground plane (`z = 0`), making it harder to tell aerial balls apart from ground passes during overhead occlusions.

### Future Improvements
1. **Ball Detector + Tracking + More Real-World Data (`10% -> 25%`)**: **In our case**, increasing real-world match imagery in the ball training set from **10% to 25%** together with tighter ball-detector + continuous tracking integration will reach more occluded/low-contrast frames and improve accuracy.
2. **Clean Player Team-Marking Data**: Building a dedicated, accurate player team-marking model/dataset will prevent team-label errors from flipping possession.
3. **Synchronized Multi-Camera Views (2–3 Cameras)**: Just like tactical multi-angle football setups, using **2 to 3 synchronized camera views** will resolve occluded moments where a single view hides the ball or player, directly improving ball and possession accuracy.
4. **Depth Estimation Layer (2D-to-3D Mapping)**: We tried converting the 2D tactical video into a 3D mapping using homography, which scored lower than our current pipeline; however, bringing in a true third dimension with depth estimation data (an additional spatial layer) will resolve several possession issues (separating aerial flight height from ground passes and true foot touches from body occlusions) and improve accuracy.

---

## 6. Reproduction

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Place tactical data in ./data/tactical/ (or set export MTAH_DATA_DIR=/path/to/data)

# 3. Ball detection (uses model/best.pt):
#    - Provided match: skip this step (cleaned ball JSON is included at src/ball_detections_cleaned.json).
#    - Fresh / replaced data folder: run on GPU with larger batch size (CPU is much slower):
python3 detect_ball_play_time.py --device 0 --batch-size 32
# (Or target a custom folder: python3 detect_ball_play_time.py --data-dir /path/to/data --device 0 --batch-size 32)

# 4. Build predictions -> evaluation/results/Matra.csv
#    Automatically uses <data-dir>/ball_detections_cleaned.json if present, else src/ball_detections_cleaned.json
python3 build_submission.py
# (Add --data-dir /path/to/data or --fps <FPS> if running on a custom dataset)

# 5. Validate & score
python3 evaluation/validate_submission.py evaluation/results/Matra.csv
python3 evaluation/run_evaluation.py --no-detail        # public 5-min GT (--gt private for full match)
```
