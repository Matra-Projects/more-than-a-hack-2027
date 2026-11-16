"""
Module 1 - DEADBALL ENGINE (Stage 1: owns the D label, runs FIRST, is final)

Decides, for every video frame, whether the ball is DEAD (D). Stage 2 (possession)
only ever sees the frames this module leaves alive.

TACTICAL SOURCE ONLY: ATD, homography and Smart Tagging all come from
games.require()/asset_paths() with the default source="tactical".

Rules, each kept as its own mask so the contribution of every rule is measurable
(build_dead_mask(..., return_reasons=True)):

  trap       published boundary: graded frames 110808-113741 are 100% D
  out        ball seen beyond a line (homography-trusted)  -> D until the restart
  ballistic  ball lost while travelling: drag/friction + curving extrapolation;
             if the path crosses a line before the ball reappears -> D until restart.
             If it stays in play the gap is NOT D (Stage 2 holds possession).
  setpiece   Smart Tagging SET PIECES: D from the stoppage (walked back from the tag
             while the ball is settled) until the ball is actually struck back into
             play AND the players react
  goal       GOALS: D from the last live play before the kick-off until the kick-off
  stoppage   whole-field motion collapse (median player speed) sustained, confirmed
             by ball lost / near a line / a tag / a full freeze, vetoed when it looks
             like a keeper holding the ball in play

Key data facts this module is built around (all measured on the tactical files):
  * The 3rd ball field of the ATD (declared speed_kph) is NOT a usable ball speed:
    values 0-7, correlation -0.08 with the displacement speed. Ball speed is derived
    from positions here, with a median pre-filter and a 42 m/s jump cap.
  * Set-piece tag windows are 875 frames (35 s) except kick-offs (125 frames). The
    tag start sits inside the dead stretch; the restart is somewhere after it.
  * GOALS windows are 70-250 s and start with the attack that led to the goal, so
    they must not be marked D wholesale.
  * When the referee whistles the whole field stops: median player speed drops
    from ~2 m/s to < 1 m/s within ~2 s. This is the strongest stoppage signal when
    the ball itself is lost (41% of graded frames) or being carried back.
  * Homography projects normalised pitch coordinates to pixels; a ball pixel lies
    inside the projected pitch quadrilateral exactly when its normalised coordinates
    lie inside [0,1]^2 (a projective map preserves incidence). So "outside the pitch
    quad" is tested in pitch coordinates; the homography contributes its
    reprojection_error as a trust gate.

numpy + pandas + standard library only.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, half_video_blocks  # noqa: E402
import load_smart_tagging as st  # noqa: E402
import load_homography as lh  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters (physical meaning, not fitted to the public 5 minutes)
# ----------------------------------------------------------------------------
FPS = 25
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]          # 105.3 x 68.0 m
N_SLOTS = 46

# The published 2nd-half trap: graded from 110808, kickoff at 113742.
SECOND_HALF_GRADED_START = 110808
SECOND_HALF_KICKOFF = 113742

# --- ball kinematics / trust ---------------------------------------------------
BALL_MEDIAN_FRAMES = 5        # median pre-filter on ball position (kills 1-2 frame tracker jumps)
SPEED_HALF_WIN = 2            # ball speed from displacement over +-2 frames (0.16 s)
MAX_BALL_SPEED = 42.0         # m/s (~150 km/h, a hard shot): faster = tracker jump -> unseen
REPROJ_TRUST_MAX = 0.8        # homography reprojection error (median 0.23 in play); above
                              # this the camera is off the pitch and the ball is distrusted

# --- out of bounds -------------------------------------------------------------
BOUND_TOL_M = 1.0             # ball centre this far beyond a line = out (line width + noise)
OUT_PERSIST = 2               # consecutive frames, so a single jittery sample cannot fire

# --- restarts --------------------------------------------------------------------
STATIC_SPEED = 1.5            # m/s: ball at rest / walked with (placed for a set piece)
RESTART_SPEED = 4.0           # m/s: ball clearly struck or thrown back into play
RESTART_CONFIRM_S = 3.0       # after the strike the players must react within this ...
                              # (median speed above RELEASE_MOTION), else it was a tap-back
REACT_RISE = 0.3              # m/s: ... and the field must ACCELERATE by this much (a ball
                              # fetched while players are still retreating is not a restart)
SETTLE_S = 1.0                # a tagged set piece needs this much settled ball/players first
MAX_DEAD_S = 90.0             # cap on one dead stretch opened by a crossing (injury pauses)

# --- whole-field motion ----------------------------------------------------------
MOTION_HALF_WIN = 12          # player speed from displacement over +-0.48 s
MAX_PLAYER_SPEED = 11.0       # m/s, faster than a sprint = track swap, ignored
MOTION_SMOOTH_S = 2.0         # rolling median of the field speed
STOP_MOTION = 0.9             # m/s median player speed: the whole field is walking
RELEASE_MOTION = 1.3          # m/s: the field is moving again (hysteresis release)
FREEZE_MOTION = 0.7           # m/s: everyone standing - a stoppage by itself
STOP_MIN_S = 3.0              # sustained this long before it counts
COLLAPSE_RATIO = 2.0          # whistle signature: field speed falls to <= 1/2 of its level ...
COLLAPSE_S = 2.0              # ... within this many seconds, then stays below RELEASE_MOTION
WHISTLE_LEAD_S = 1.0          # players decelerate ~1 s after the whistle
BOUNDARY_NEAR_M = 2.0         # ball this close to a line supports a stoppage
LOST_BALL_FRAC = 0.5          # ball unseen for this share of the stoppage supports it
KEEPER_BOX_X = 16.5           # penalty-area depth (m)
KEEPER_HOLD_R = 1.5           # a player this close to a visible ball = ball in hand / at feet

# --- ballistic extrapolation of a lost ball ----------------------------------------
GAP_MIN_FRAMES = 3
EXTRAP_MIN_SPEED = 6.0        # m/s: slower lost balls are at a player's feet -> in play
EXTRAP_MAX_S = 2.0            # do not trust an extrapolation further than this
ROLL_DECEL = 1.5              # m/s^2 rolling resistance on grass
AERIAL_SPEED = 15.0           # m/s: faster than this the ball is assumed airborne
AIR_DRAG_K = 0.013            # 1/m  quadratic drag rho*Cd*A/(2m) for a size-5 ball
MAX_YAW_RATE = 0.6            # rad/s: curl of a struck ball, cap for the heading change

# --- set pieces / goals -----------------------------------------------------------
SETPIECE_BACK_MAX_S = 15.0    # walk back from the tag start at most this far (ball fetch)
SETPIECE_FWD_PAD_S = 10.0     # search for the restart up to tag end + this
OUT_CAUSED = ("throw-in", "goal kick", "corner kick")   # restarts caused by the ball
                              # leaving the pitch: D starts at the crossing

# --- cleanup ------------------------------------------------------------------------
MIN_DEAD_S = 1.5              # a real stoppage lasts longer than this
MIN_ALIVE_S = 1.0             # an alive sliver shorter than this between two dead runs is D


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_tracking(game=GAME, data_dir=None):
    """One pass over the tactical ATD + homography trust.

    Returns a dict:
        ball_x, ball_y   metres, NaN when unseen or distrusted (reprojection spike)
        ball_s           m/s, derived from positions (the ATD 3rd field is not speed)
        P                (n, 46, 2) float32 player positions in metres, slot-indexed
        slot_team        (46,) 0 = home (first <Team>), 1 = away
        reproj           per-frame homography reprojection error
    Coordinates: normalised 0-1 -> metres with the anisotropic pitch scaling.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = GAMES[game]["n_video_frames"]

    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    team_ids = [t.get("id") for t in teams]
    team_names = []
    for t in teams:
        nm = t.find("Name")
        team_names.append((nm.text or "").strip().lower() if nm is not None else "")
    home_id = team_ids[0]
    players = list(root.iter("Player"))
    slot_team = np.array([0 if p.get("teamId") == home_id else 1
                          for p in players], dtype=np.int8)
    gk_slots = []
    for idx_p, p in enumerate(players):
        for pp in p.iter("ProviderParameter"):
            nm = pp.find("Name")
            val = pp.find("Value")
            if nm is not None and val is not None and (nm.text or "").strip() == "position_type":
                if "goalkeeper" in (val.text or "").strip().lower():
                    gk_slots.append(idx_p)

    P = np.full((n, N_SLOTS, 2), np.nan, dtype=np.float32)
    ball_x = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1                       # frame_id is 1-based
            if not (0 <= f < n):
                continue
            row = P[f]
            for slot, g in enumerate(parts[1].split(";")):
                if g[0] == "N":
                    continue
                c1 = g.index(","); c2 = g.index(",", c1 + 1)
                row[slot, 0] = float(g[:c1]) * PITCH_L
                row[slot, 1] = float(g[c1 + 1:c2]) * PITCH_W
            b = parts[2]
            if b[0] != "N":
                c1 = b.index(","); c2 = b.index(",", c1 + 1)
                ball_x[f] = float(b[:c1]) * PITCH_L
                ball_y[f] = float(b[c1 + 1:c2]) * PITCH_W

    _, reproj, valid = lh.load(game, data_dir)            # tactical homography
    reproj = np.asarray(reproj, dtype=float)[:n]
    distrust = ~np.asarray(valid, dtype=bool)[:n] | (reproj > REPROJ_TRUST_MAX)
    ball_x[distrust] = np.nan
    ball_y[distrust] = np.nan

    if not gk_slots:
        in_box = ((P[:, :, 0] < 18.0) | (P[:, :, 0] > PITCH_L - 18.0)) & (P[:, :, 1] >= 13.84) & (P[:, :, 1] <= 54.16)
        vis_p = ~np.isnan(P[:, :, 0])
        vc = vis_p.sum(axis=0)
        bf = np.where(vc > 0, in_box.sum(axis=0) / np.maximum(1, vc), 0.0)
        gk_slots = np.where(bf >= 0.50)[0].tolist()

    ball_x, ball_y, ball_s = _clean_ball(ball_x, ball_y)
    return {"ball_x": ball_x, "ball_y": ball_y, "ball_s": ball_s, "P": P,
            "slot_team": slot_team[:N_SLOTS], "reproj": reproj,
            "gk_slots": tuple(gk_slots), "team_names": tuple(team_names)}


def _clean_ball(bx, by):
    """Median pre-filter, displacement speed, jump rejection."""
    k = BALL_MEDIAN_FRAMES
    vis = ~np.isnan(bx)
    sx = pd.Series(bx).rolling(k, center=True, min_periods=1).median().to_numpy(dtype=float, copy=True)
    sy = pd.Series(by).rolling(k, center=True, min_periods=1).median().to_numpy(dtype=float, copy=True)
    sx[~vis] = np.nan
    sy[~vis] = np.nan

    h = SPEED_HALF_WIN
    n = len(bx)
    s = np.full(n, np.nan)
    # centred difference, fall back to one-sided at gap edges
    for lo, hi in ((h, h), (0, h), (h, 0), (1, 1), (0, 1), (1, 0)):
        if lo + hi == 0:
            continue
        a = np.full(n, np.nan)
        dx = np.full(n, np.nan); dy = np.full(n, np.nan)
        idx = np.arange(lo, n - hi)
        dx[idx] = sx[idx + hi] - sx[idx - lo]
        dy[idx] = sy[idx + hi] - sy[idx - lo]
        a = np.hypot(dx, dy) * FPS / (lo + hi)
        s = np.where(np.isnan(s), a, s)
    jump = s > MAX_BALL_SPEED
    sx[jump] = np.nan; sy[jump] = np.nan; s[jump] = np.nan
    return sx, sy, s


def players_by_team(trk):
    """Per-frame home/away position arrays (the legacy list format Module 2 uses)."""
    P = trk["P"]
    home = trk["slot_team"] == 0
    n = len(P)
    hx, hy, ax, ay = [None] * n, [None] * n, [None] * n, [None] * n
    for f in range(n):
        row = P[f]
        ok = ~np.isnan(row[:, 0])
        h = ok & home; a = ok & ~home
        hx[f] = row[h, 0].astype(float); hy[f] = row[h, 1].astype(float)
        ax[f] = row[a, 0].astype(float); ay[f] = row[a, 1].astype(float)
    return hx, hy, ax, ay


def load_ball_and_players(game=GAME, data_dir=None):
    """Legacy interface: (ball_x, ball_y, ball_s, home_x, home_y, away_x, away_y)."""
    trk = load_tracking(game, data_dir)
    hx, hy, ax, ay = players_by_team(trk)
    return trk["ball_x"], trk["ball_y"], trk["ball_s"], hx, hy, ax, ay


def graded_mask(game=GAME):
    """True on the 158,857 graded frames; False on pre-match/halftime/post-match."""
    n = GAMES[game]["n_video_frames"]
    m = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        m[first:last + 1] = True
    return m


def smart_tag_intervals(game=GAME, data_dir=None):
    """SET PIECES as (start, end, team, type) and GOALS as (start, end)."""
    events = st.load(game, data_dir)
    restarts, goals = [], []
    for e in events:
        code = (e["code"] or "").strip().lower()
        if code == "set pieces":
            restarts.append((e["start_frame"], e["end_frame"],
                             (e["tags"].get("Team") or "").lower(),
                             (e["tags"].get("Type") or "").lower()))
        elif code == "goals":
            goals.append((e["start_frame"], e["end_frame"]))
    return restarts, goals


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def field_motion(P):
    """Smoothed median player speed (m/s) per frame. NaN-safe, track-swap-safe."""
    n = len(P)
    k = MOTION_HALF_WIN
    v = np.full((n, P.shape[1]), np.nan, dtype=np.float32)
    d = P[2 * k:] - P[:-2 * k]
    v[k:n - k] = np.hypot(d[..., 0], d[..., 1]) * (FPS / (2.0 * k))
    v[v > MAX_PLAYER_SPEED] = np.nan
    cnt = np.sum(~np.isnan(v), axis=1)
    med = np.full(n, np.nan)
    ok = cnt >= 6
    med[ok] = np.nanmedian(v[ok], axis=1)
    w = int(MOTION_SMOOTH_S * FPS)
    return pd.Series(med).rolling(w, center=True, min_periods=w // 3).median().to_numpy(dtype=float, copy=True)


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


class _Ctx:
    """Per-frame arrays shared by the rules."""

    def __init__(self, trk):
        self.bx, self.by, self.s = trk["ball_x"], trk["ball_y"], trk["ball_s"]
        self.n = len(self.bx)
        self.vis = ~np.isnan(self.bx)
        bx = np.where(self.vis, self.bx, PITCH_L / 2)
        by = np.where(self.vis, self.by, PITCH_W / 2)
        self.inside = self.vis & (bx >= -BOUND_TOL_M) & (bx <= PITCH_L + BOUND_TOL_M) & \
            (by >= -BOUND_TOL_M) & (by <= PITCH_W + BOUND_TOL_M)
        self.line_dist = np.where(self.vis, np.minimum(np.minimum(bx, PITCH_L - bx),
                                                       np.minimum(by, PITCH_W - by)), np.inf)
        self.motion = trk.get("motion")
        if self.motion is None:
            self.motion = field_motion(trk["P"])
        self.mot = np.nan_to_num(self.motion, nan=RELEASE_MOTION + 1.0)  # unknown = moving
        s = np.nan_to_num(self.s, nan=0.0)
        self.strike = self.inside & (s > RESTART_SPEED)
        self.settled = (~self.vis) | (s < STATIC_SPEED) | (self.mot < STOP_MOTION)
        # nearest player of either team to the ball (keeper-hold & live-play veto)
        P = trk["P"]
        self.slot_team = trk["slot_team"]
        dx = P[..., 0] - self.bx[:, None]
        dy = P[..., 1] - self.by[:, None]
        with np.errstate(invalid="ignore"):
            d_mat = np.where(np.isnan(dx), np.inf, np.hypot(dx, dy))
            self.d_near = np.min(d_mat, axis=1)
            self.near_slot = np.argmin(d_mat, axis=1)
        self.d_near[~self.vis] = np.inf

    def restart(self, f0, f_max, need_settle, strict=True):
        """First frame >= f0 where the ball is struck into play and the field reacts
        (moves above RELEASE_MOTION and, if strict, accelerates by REACT_RISE)."""
        n = self.n
        f_max = min(f_max, n)
        conf = int(RESTART_CONFIRM_S * FPS)
        f = f0
        if need_settle:
            need = int(SETTLE_S * FPS)
            got = 0
            while f < f_max and got < need:
                got = got + 1 if self.settled[f] else 0
                f += 1
            if got < need:
                return None
        cand = np.where(self.strike[f:f_max])[0]
        for c in cand:
            g = f + c
            after = np.max(self.mot[g:min(n, g + conf)])
            if after <= RELEASE_MOTION:
                continue
            if strict and after - np.min(self.mot[max(0, g - FPS):g + 1]) < REACT_RISE:
                continue
            return g
        if strict:
            return self.restart(f0, f_max, need_settle, strict=False)
        return None

    def walk_back(self, f, f_min):
        """Earliest frame of the settled stretch ending at f (ball lost/out/slow or
        field walking), not before f_min."""
        while f - 1 >= f_min and (self.settled[f - 1] or not self.inside[f - 1]):
            f -= 1
        return f


# ----------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------
def rule_out(ctx):
    """Returns (mask, crossing frames)."""
    m = np.zeros(ctx.n, dtype=bool)
    crossings = []
    out = ctx.vis & ~ctx.inside
    cap = int(MAX_DEAD_S * FPS)
    for a, b in _runs(out):
        if b - a < OUT_PERSIST:
            continue
        crossings.append(a)
        r = ctx.restart(b, a + cap, need_settle=False)
        m[a:(r if r is not None else b)] = True
    return m, crossings


def rule_ballistic(ctx):
    """Returns (mask, crossing frames)."""
    m = np.zeros(ctx.n, dtype=bool)
    crossings = []
    bx, by, vis = ctx.bx, ctx.by, ctx.vis
    cap = int(MAX_DEAD_S * FPS)
    dt = 1.0 / FPS
    for g0, g1 in _runs(~vis):
        if g1 - g0 < GAP_MIN_FRAMES or g0 < 5 or not vis[g0 - 5:g0].all():
            continue
        p1 = np.array([bx[g0 - 1], by[g0 - 1]])
        p3 = np.array([bx[g0 - 3], by[g0 - 3]])
        p5 = np.array([bx[g0 - 5], by[g0 - 5]])
        v = (p1 - p3) / (2 * dt)
        speed = float(np.hypot(*v))
        if speed < EXTRAP_MIN_SPEED or speed > MAX_BALL_SPEED:
            continue
        h1 = np.arctan2(*(p1 - p3)[::-1])
        h0 = np.arctan2(*(p3 - p5)[::-1])
        yaw = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
        yaw = float(np.clip(yaw / (2 * dt), -MAX_YAW_RATE, MAX_YAW_RATE))
        aerial = speed > AERIAL_SPEED
        x, y, hd = p1[0], p1[1], h1
        cross = None
        for k in range(min(g1 - g0, int(EXTRAP_MAX_S * FPS))):
            if aerial:
                speed -= AIR_DRAG_K * speed * speed * dt
            else:
                speed = max(0.0, speed - ROLL_DECEL * dt)
            if speed <= 0.0:
                break
            hd += yaw * dt
            x += speed * dt * np.cos(hd)
            y += speed * dt * np.sin(hd)
            if x < 0 or x > PITCH_L or y < 0 or y > PITCH_W:
                cross = g0 + k
                break
        if cross is None:
            continue                  # stays in play: Stage 2 holds possession
        if (g1 - g0) <= 2 * FPS and g1 < ctx.n and ctx.inside[g1] and ctx.line_dist[g1] > 2.5:
            continue                  # reappeared well inside pitch after brief occlusion
        crossings.append(cross)
        r = ctx.restart(g1, cross + cap, need_settle=False)
        m[cross:(r if r is not None else g1)] = True
    return m, crossings


def rule_setpiece(ctx, restarts, crossings=()):
    m = np.zeros(ctx.n, dtype=bool)
    back = int(SETPIECE_BACK_MAX_S * FPS)
    pad = int(SETPIECE_FWD_PAD_S * FPS)
    for (s0, s1, _team, _typ) in sorted(restarts):
        s0 = int(np.clip(s0, 0, ctx.n - 1)); s1 = int(np.clip(s1, s0, ctx.n - 1))
        r = ctx.restart(s0, s1 + pad, need_settle=True)
        if r is None:
            r = ctx.restart(s0, s1 + pad, need_settle=False)
        if r is None:
            r = s0
        start = ctx.walk_back(s0, max(0, s0 - back))
        if any(k in _typ for k in OUT_CAUSED):
            before = [c for c in crossings if s0 - back <= c <= s0]
            if before:
                start = min(start, before[-1])
        m[start:r] = True
    return m


def rule_goal(ctx, goals, restarts):
    m = np.zeros(ctx.n, dtype=bool)
    kos = sorted(s0 for (s0, _s1, _t, typ) in restarts if "kick off" in typ)
    pad = int(SETPIECE_FWD_PAD_S * FPS)
    for (g0, g1) in goals:
        after = [k for k in kos if g0 <= k <= g1 + pad]
        ko = after[0] if after else g1
        r = ctx.restart(ko, ko + pad, need_settle=False)
        r = r if r is not None else ko
        # last live play inside the window: walk back while ball lost / slow / field calm
        start = ctx.walk_back(min(r, ko), g0)
        m[start:r] = True
    return m


def rule_stoppage(ctx, tags):
    m = np.zeros(ctx.n, dtype=bool)
    mot = ctx.mot
    minlen = int(STOP_MIN_S * FPS)
    lead = int(WHISTLE_LEAD_S * FPS)
    tag_mask = np.zeros(ctx.n, dtype=bool)
    for a, b in tags:
        tag_mask[max(0, a):min(ctx.n, b)] = True
    cands = set()
    for a, b in _runs(mot < STOP_MOTION):                  # deep calm
        if b - a < minlen:
            continue
        while a > 0 and mot[a - 1] < RELEASE_MOTION:
            a -= 1
        while b < ctx.n and mot[b] < RELEASE_MOTION:
            b += 1
        cands.add((a, b))
    cw = int(COLLAPSE_S * FPS)
    for a, b in _runs(mot < RELEASE_MOTION):               # whistle collapse
        if b - a < minlen or a < cw:
            continue
        if np.max(mot[a - cw:a]) >= COLLAPSE_RATIO * np.median(mot[a:b]):
            cands.add((a, b))
    for a, b in sorted(cands):
        seg = slice(a, b)
        vis = ctx.vis[seg]
        lost = (~vis).mean() >= LOST_BALL_FRAC
        near_line = bool(np.any(ctx.line_dist[max(0, a - lead):b] < BOUNDARY_NEAR_M))
        tagged = bool(tag_mask[seg].any())
        freeze = float(np.min(mot[seg])) < FREEZE_MOTION
        ball_still = bool(vis.any() and np.nanmedian(ctx.s[seg]) < 1.2)
        if not (lost or near_line or tagged or freeze or ball_still):
            continue
        if not tagged and vis.mean() >= 0.8:
            in_box = (ctx.bx[seg] < KEEPER_BOX_X) | (ctx.bx[seg] > PITCH_L - KEEPER_BOX_X)
            held = ctx.d_near[seg] <= KEEPER_HOLD_R
            if np.nanmean(in_box & held) >= 0.8:
                continue              # keeper with the ball in play: live possession
        start = max(0, a - lead)
        r = ctx.restart(max(a, b - int(RESTART_CONFIRM_S * FPS)), b + lead, need_settle=False)
        end = r if r is not None else b
        seg_dead = slice(a, max(a + 1, end))
        if not tagged and ctx.inside[seg_dead].mean() >= 0.75:
            in_seg = ctx.inside[seg_dead]
            touch = in_seg & (ctx.d_near[seg_dead] <= 2.5)
            fast = in_seg & (np.nan_to_num(ctx.s[seg_dead], nan=0.0) > 4.0)
            span_m = float(np.hypot(
                np.nanmax(ctx.bx[seg_dead]) - np.nanmin(ctx.bx[seg_dead]),
                np.nanmax(ctx.by[seg_dead]) - np.nanmin(ctx.by[seg_dead]),
            )) if in_seg.any() else 0.0
            if touch.any():
                tms = ctx.slot_team[ctx.near_slot[seg_dead][touch]]
                dom_share = max(float(np.mean(tms == 0)), float(np.mean(tms == 1)))
            else:
                dom_share = 0.0
            if touch.mean() >= 0.60 and dom_share >= 0.70 and int(fast.sum()) >= 20 and span_m >= 10.0:
                continue              # live open-play passing / build-up inside homography
        m[start:end] = True
    return m


def _cleanup(dead, protect):
    out = dead.copy()
    for a, b in _runs(~out):
        if 0 < a and b < len(out) and (b - a) < int(MIN_ALIVE_S * FPS):
            out[a:b] = True
    for a, b in _runs(out):
        if (b - a) < int(MIN_DEAD_S * FPS) and not protect[a:b].any():
            out[a:b] = False
    return out


# ----------------------------------------------------------------------------
# Stage 1 entry point
# ----------------------------------------------------------------------------
RULES = ("trap", "out", "ballistic", "setpiece", "goal", "stoppage")


def build_dead_mask(trk, game=GAME, data_dir=None, verbose=True, return_reasons=False,
                    rules=RULES):
    """Boolean array, True = D. `trk` is load_tracking()'s dict."""
    n = len(trk["ball_x"])
    graded = graded_mask(game)
    ctx = _Ctx(trk)
    restarts, goals = smart_tag_intervals(game, data_dir)
    if verbose:
        print("  smart tagging: {} set pieces, {} goals".format(len(restarts), len(goals)))

    reasons = {}
    if "trap" in rules:
        t = np.zeros(n, dtype=bool)
        t[SECOND_HALF_GRADED_START:SECOND_HALF_KICKOFF] = True
        reasons["trap"] = t
    crossings = []
    if "out" in rules:
        reasons["out"], c = rule_out(ctx)
        crossings += c
    if "ballistic" in rules:
        reasons["ballistic"], c = rule_ballistic(ctx)
        crossings += c
    if "setpiece" in rules:
        reasons["setpiece"] = rule_setpiece(ctx, restarts, sorted(crossings))
    if "goal" in rules:
        reasons["goal"] = rule_goal(ctx, goals, restarts)
    if "stoppage" in rules:
        tags = [(a, b) for (a, b, _t, _y) in restarts] + list(goals)
        reasons["stoppage"] = rule_stoppage(ctx, tags)

    dead = np.zeros(n, dtype=bool)
    for k in reasons:
        reasons[k] &= graded
        dead |= reasons[k]
    protect = reasons.get("trap", np.zeros(n, dtype=bool))
    dead = _cleanup(dead, protect) & graded
    dead |= protect

    if verbose:
        g = int(graded.sum())
        print("  dead frames: {} of {} graded ({:.1f}%)".format(int(dead.sum()), g,
                                                                100.0 * dead.sum() / g))
        for k, v in reasons.items():
            print("    {:<10} {:>6} frames ({:.1f}% of graded)".format(k, int(v.sum()),
                                                                       100.0 * v.sum() / g))
    if return_reasons:
        return dead, reasons
    return dead
