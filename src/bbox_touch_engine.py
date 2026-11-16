"""
Hard-coded bbox-touch possession rules (see rules.txt at the repo root).

Possession is decided only by pixel contact between the ball bbox (cleaned ball JSON)
and player bboxes (ATD foot position projected through the tactical homography,
box [u-16, v-52, u+16, v+4] -- the exact box drawn in the mismatch videos).

    R1 touch      ball bbox overlaps a player bbox                        -> that player touches
    R2 transit    no player bbox touches the ball                         -> possession unchanged
    R3 control    the other team takes over only after >= CONTROL_FRAMES touch frames
                  within CONTROL_WINDOW frames with no owner touch in between (one-touch
                  flick / deflection = no change); the switch is back-dated to the first
                  of those touch frames
    R4 collision  ball bbox overlaps players of BOTH teams                -> no change; the team
                  that next controls the ball cleanly owns it from the start of the collision
    R5 restart    after a dead ball, the first team to touch owns it at once (kickoff, goal kick)
    R6 keeper     ball last seen inside a penalty area, then unseen, with that area's keeper
                  the nearest player to it                                -> keeper's team
    R7 dead       Stage-1 dead mask is final (D)
    R8 fallback   no touch has happened yet since the last dead ball       -> HMM label
"""

import numpy as np

PLAYER_HALF_W = 16.0
PLAYER_TOP = 20.0           # touch zone = feet part of the drawn box [v-52, v+4] -> [v-20, v+4]
PLAYER_BOTTOM = 4.0         # (a ball in the air passing a player's body/head is not a touch)
CONTROL_FRAMES = 3          # R3: touch frames needed for the other team to take over
CONTROL_WINDOW = 25         # R3: ... within this many frames (1 s), with no owner touch between
GK_SLOTS = {23: "H", 45: "A"}   # slot index -> team (slots 24 and 46)
GK_NEAR_M = 6.0             # R6: keeper must be within this of the last seen ball
GK_MIN_GAP = 10             # R6: ball unseen at least this many frames
PEN_DEPTH = 16.5
PEN_HALF_W = 20.16
ATD_BALL_HALF = 7.0         # half size (px) of the ball box built from an ATD ball point


def _load_ball_px(ball_json_path, n):
    import json
    with open(ball_json_path, encoding="utf-8") as fh:
        frames = json.load(fh).get("frames", {})
    box = np.full((n, 4), np.nan)
    for k, dets in frames.items():
        f = int(k)
        if dets and 0 <= f < n and dets[0].get("bbox_xyxy"):
            box[f] = dets[0]["bbox_xyxy"]
    return box


def _project_players(P, H, valid, frames, pitch_l, pitch_w):
    """(len(frames), slots, 2) pixel foot positions; NaN where unknown."""
    Pn = P[frames].astype(float)
    nx = Pn[..., 0] / pitch_l
    ny = Pn[..., 1] / pitch_w
    Hf = H[frames]
    den = Hf[:, 2, 0, None] * nx + Hf[:, 2, 1, None] * ny + Hf[:, 2, 2, None]
    u = (Hf[:, 0, 0, None] * nx + Hf[:, 0, 1, None] * ny + Hf[:, 0, 2, None]) / den
    v = (Hf[:, 1, 0, None] * nx + Hf[:, 1, 1, None] * ny + Hf[:, 1, 2, None]) / den
    bad = ~valid[frames][:, None] | (np.abs(den) < 1e-7)
    u[bad] = np.nan
    v[bad] = np.nan
    return u, v


def touch_teams(trk, ball_box, H, valid, pitch_l, pitch_w):
    """Per frame: 0 none, 1 home only, 2 away only, 3 both (collision)."""
    n = len(ball_box)
    out = np.zeros(n, dtype=np.int8)
    seen = np.where(~np.isnan(ball_box[:, 0]))[0]
    if len(seen) == 0:
        return out
    u, v = _project_players(trk["P"], H, valid, seen, pitch_l, pitch_w)
    bx1, by1, bx2, by2 = (ball_box[seen, i][:, None] for i in range(4))
    hit = (bx2 >= u - PLAYER_HALF_W) & (bx1 <= u + PLAYER_HALF_W) & \
          (by2 >= v - PLAYER_TOP) & (by1 <= v + PLAYER_BOTTOM)
    hit &= ~np.isnan(u)
    home = np.asarray(trk["slot_team"]) == 0
    h = hit[:, home].any(axis=1)
    a = hit[:, ~home].any(axis=1)
    out[seen] = h.astype(np.int8) + 2 * a.astype(np.int8)
    return out


def _keeper_owner(trk, ball_m, f_last, pitch_l, pitch_w):
    """R6: team of the keeper guarding the penalty area the ball was last seen in, or None."""
    bx, by = ball_m[0][f_last], ball_m[1][f_last]
    if np.isnan(bx) or abs(by - pitch_w / 2.0) > PEN_HALF_W:
        return None
    if not (bx < PEN_DEPTH or bx > pitch_l - PEN_DEPTH):
        return None
    P = trk["P"][f_last]
    d = np.hypot(P[:, 0] - bx, P[:, 1] - by)
    if np.all(np.isnan(d)):
        return None
    near = int(np.nanargmin(d))
    if near in GK_SLOTS and d[near] <= GK_NEAR_M:
        return GK_SLOTS[near]
    return None


def apply_bbox_rules(hmm_labels, dead, graded, trk, ball_json_path, H, valid, ball_m,
                     pitch_l, pitch_w, verbose=True, atd_fill=None):
    n = len(hmm_labels)
    ball_box = _load_ball_px(ball_json_path, n)
    if atd_fill is not None:
        # Frames with no cleaned ball but an ATD ball: project the ATD point, ball box +-ATD_BALL_HALF px
        fill = np.where(atd_fill & np.isnan(ball_box[:, 0]) & valid[:n])[0]
        if len(fill):
            nx = ball_m[0][fill] / pitch_l
            ny = ball_m[1][fill] / pitch_w
            Hf = H[fill]
            den = Hf[:, 2, 0] * nx + Hf[:, 2, 1] * ny + Hf[:, 2, 2]
            u = (Hf[:, 0, 0] * nx + Hf[:, 0, 1] * ny + Hf[:, 0, 2]) / den
            v = (Hf[:, 1, 0] * nx + Hf[:, 1, 1] * ny + Hf[:, 1, 2]) / den
            ball_box[fill] = np.stack([u - ATD_BALL_HALF, v - ATD_BALL_HALF,
                                       u + ATD_BALL_HALF, v + ATD_BALL_HALF], axis=1)
            if verbose:
                print(f"  BBox-Touch Rules: {len(fill):,} missing ball frames filled from ATD ball")
    tt = touch_teams(trk, ball_box, H, valid, pitch_l, pitch_w)
    seen = ~np.isnan(ball_box[:, 0])

    out = hmm_labels.copy()
    state = None              # current owner "H"/"A" or None (no touch since last dead)
    challenger = []           # recent touch frames by the non-owning team
    coll_start = None
    last_seen = -1
    stats = {"switch": 0, "flick_ignored": 0, "collision_resolved": 0, "keeper": 0, "restart": 0}

    for f in range(n):
        if not graded[f]:
            continue
        if dead[f]:
            out[f] = "D"
            state, challenger, coll_start = None, [], None
            continue

        t = tt[f]
        if seen[f]:
            # R6: ball reappears after a gap that started inside a penalty area
            if last_seen >= 0 and f - last_seen > GK_MIN_GAP:
                owner = _keeper_owner(trk, ball_m, last_seen, pitch_l, pitch_w)
                if owner is not None and owner != state:
                    out[last_seen + 1:f] = owner
                    state = owner
                    stats["keeper"] += 1
            last_seen = f

        if t == 3:                                    # R4 collision
            if coll_start is None:
                coll_start = f
        elif t in (1, 2):                             # R1 touch
            team = "H" if t == 1 else "A"
            if state is None:                         # R5 restart
                state = team
                stats["restart"] += 1
                coll_start = None
                challenger = []
            elif team == state:
                if challenger:
                    stats["flick_ignored"] += 1
                challenger = []                       # owner touched again -> challenge over
                coll_start = None                     # owner kept it through the collision
            else:
                challenger = [c for c in challenger if f - c <= CONTROL_WINDOW] + [f]
                if len(challenger) >= CONTROL_FRAMES:  # R3 control
                    start = challenger[0]
                    if coll_start is not None:
                        start = min(start, coll_start)
                        stats["collision_resolved"] += 1
                    out[start:f] = team
                    state = team
                    stats["switch"] += 1
                    coll_start = None
                    challenger = []

        if state is not None:                         # R2 transit keeps state
            out[f] = state
        # else R8: keep HMM label

    out[dead & graded] = "D"
    if verbose:
        print("  BBox-Touch Rules: touch frames H {} | A {} | collision {}".format(
            int((tt == 1).sum()), int((tt == 2).sum()), int((tt == 3).sum())))
        print("  BBox-Touch Rules: {switch} switches, {flick_ignored} flicks ignored, "
              "{collision_resolved} collisions resolved, {keeper} keeper holds, "
              "{restart} restarts".format(**stats))
    return out
