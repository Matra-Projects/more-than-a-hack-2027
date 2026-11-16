"""
src/ball_speed.py - ball speed pipeline + homography trust gate.

Outputs, per video frame (0..199318, tactical timeline):
    ball_x, ball_y   ball position in METRES (anisotropic 0-1 -> pitch scaling)
    speed            denoised ball speed in m/s (NaN where the ball is unseen
                     or UNTRUSTED - treat those frames exactly like missing)
    trusted          boolean: ball position passed the homography trust gate

Speed comes from the ATD positions directly - homography is used ONLY as a
trust gate (is the projected ball pixel actually on the visible pitch? is the
frame's homography fit good?), never to compute speed.

    from ball_speed import build_speed_and_trust
    bx, by, speed, trusted = build_speed_and_trust()

Changes from the submitted draft (each measured on the full tactical match):
  * pandas 3 returns read-only arrays from .values; copy before writing NaN.
  * The ATD 3rd ball column is NOT blended into the speed any more. It is present
    on only 9,889 frames, its median is 1.23 against 6.19 m/s for the position
    speed, and the correlation is -0.12; averaging halved the speed on those frames
    (frames > 9 m/s fell from 3,218 to 1,234). It is kept as a printed cross-check.
  * _inside_quad: a near-edge point (|cross| <= QUAD_TOL_PX) counts as inside.
    The draft's "|sum of signs| == 4" made every near-edge point fail, so 165 of
    411 balls just inside a line were rejected - the opposite of the intent.
  * Speed needs BOTH endpoints trusted, not just visible.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]        # 105.3 x 68.0 m
FPS = GAMES[GAME]["fps"]                        # 25
REPROJ_MAX = 1.0                                # frames above this are untrusted
SPEED_WINDOW = 3                                # rolling-median denoise (frames)
QUAD_TOL_PX = 2.0                               # slack on the quadrilateral edge test


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_ball_track(game=GAME, data_dir=None):
    """Parse the raw ATD once -> per-frame ball x, y (metres) and ATD's own
    3rd ball column (diagnostic only). NaN wherever the tracker lost the ball."""
    _, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = GAMES[game]["n_video_frames"]
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bs = np.full(n, np.nan)

    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1                    # frame_id is 1-based
            if not (0 <= f < n):
                continue
            b = parts[2]
            if b[0] == "N":                          # missing ball
                continue
            c1 = b.index(",")
            c2 = b.index(",", c1 + 1)
            bx[f] = float(b[:c1]) * PITCH_L
            by[f] = float(b[c1 + 1:c2]) * PITCH_W
            bs[f] = float(b[c2 + 1:])
    return bx, by, bs


def load_homography(game=GAME, data_dir=None):
    """Per-frame 3x3 matrices, reprojection errors and valid flags."""
    path = require(["homography"], game, data_dir)[0]
    df = pd.read_csv(path)
    cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    H = df[cols].to_numpy(dtype=float).reshape(len(df), 3, 3)
    return (H, df["reprojection_error"].to_numpy(dtype=float),
            df["valid"].to_numpy().astype(bool))


# ----------------------------------------------------------------------------
# Homography helpers
# ----------------------------------------------------------------------------
def project(H, x, y):
    """Normalised pitch coords -> image pixels (vectorised over frames)."""
    pts = np.stack([x, y, np.ones_like(x)], axis=-1)      # (n, 3)
    p = np.einsum("nij,nj->ni", H, pts)
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def _inside_quad(px, py, quad):
    """Point-in-convex-quadrilateral via edge cross products (vectorised).

    quad: (n, 4, 2) corner pixels, ordered around the perimeter.
    True where (px, py) is inside or within QUAD_TOL_PX of an edge: a near-edge
    cross product is neutral (0) and does not veto.
    """
    pos = np.zeros(len(px), dtype=bool)
    neg = np.zeros(len(px), dtype=bool)
    for i in range(4):
        a = quad[:, i, :]
        b = quad[:, (i + 1) % 4, :]
        ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        cross = ex * (py - a[:, 1]) - ey * (px - a[:, 0])
        cross = cross / (np.hypot(ex, ey) + 1e-9)          # pixels from the edge line
        pos |= cross > QUAD_TOL_PX
        neg |= cross < -QUAD_TOL_PX
    return ~(pos & neg)          # never strictly on both sides of the perimeter


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def build_speed_and_trust(game=GAME, data_dir=None, verbose=True, ball_override=None):
    """Returns (ball_x, ball_y, speed, trusted) - see module docstring.

    ball_override: optional (bx, by) arrays in metres that replace the ATD ball entirely.
    """
    if ball_override is not None:
        bx = np.asarray(ball_override[0], dtype=float).copy()
        by = np.asarray(ball_override[1], dtype=float).copy()
        bs_atd = np.full(len(bx), np.nan)
    else:
        bx, by, bs_atd = load_ball_track(game, data_dir)
    H, reproj, valid = load_homography(game, data_dir)
    n = len(bx)
    visible = ~np.isnan(bx)

    # --- 1. homography trust gate -------------------------------------------
    seen = visible & valid
    px = np.full(n, np.nan); py = np.full(n, np.nan)
    quad = np.full((n, 4, 2), np.nan)
    if seen.any():
        px[seen], py[seen] = project(H[seen], bx[seen] / PITCH_L, by[seen] / PITCH_W)
        for i, (cx, cy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            quad[seen, i, 0], quad[seen, i, 1] = project(
                H[seen], np.full(seen.sum(), float(cx)), np.full(seen.sum(), float(cy)))
    inside = np.zeros(n, dtype=bool)
    inside[seen] = _inside_quad(px[seen], py[seen], quad[seen])
    trusted = seen & (np.nan_to_num(reproj, nan=np.inf) <= REPROJ_MAX) & inside

    # --- 2. speed from consecutive TRUSTED positions (anisotropic metres) -----
    pair_ok = np.zeros(n, dtype=bool)
    pair_ok[1:] = trusted[1:] & trusted[:-1]         # no diffs across gaps / untrusted
    dx = np.zeros(n); dy = np.zeros(n)
    dx[1:] = bx[1:] - bx[:-1]
    dy[1:] = by[1:] - by[:-1]
    speed_calc = np.where(pair_ok, np.hypot(dx, dy) * FPS, np.nan)

    # cross-check against the ATD column (diagnostic only, never blended)
    both = ~np.isnan(speed_calc) & ~np.isnan(bs_atd)
    if verbose and both.any():
        sane = both & (speed_calc < 45)
        print("  speed cross-check vs ATD 3rd column: {} frames, mean |calc - atd| = {:.2f} m/s, "
              "corr {:.3f} -> column not used".format(
                  int(both.sum()), float(np.abs(speed_calc - bs_atd)[both].mean()),
                  float(np.corrcoef(speed_calc[sane], bs_atd[sane])[0, 1])))

    # --- 3. denoise: rolling median, NaN stays NaN ----------------------------
    speed = pd.Series(speed_calc).rolling(SPEED_WINDOW, center=True, min_periods=1) \
        .median().to_numpy(dtype=float, copy=True)
    speed[~trusted] = np.nan                          # final guard
    speed[np.isnan(speed_calc) & ~trusted] = np.nan

    if verbose:
        print("  ball seen on      {:>6} frames ({:.1f}%)".format(
            int(visible.sum()), 100.0 * visible.mean()))
        print("  trusted           {:>6} frames ({:.1f}% of seen)".format(
            int(trusted.sum()), 100.0 * trusted.sum() / max(visible.sum(), 1)))
        print("  rejected: bad homography {:>6}, reproj>{:.1f} {:>6}, outside quad {:>6}".format(
            int((visible & ~valid).sum()),
            REPROJ_MAX,
            int((visible & valid & (reproj > REPROJ_MAX)).sum()),
            int((visible & valid & (reproj <= REPROJ_MAX) & ~inside).sum())))

    bx_t = np.where(trusted, bx, np.nan)
    by_t = np.where(trusted, by, np.nan)
    return bx_t, by_t, speed, trusted


if __name__ == "__main__":
    bx, by, speed, trusted = build_speed_and_trust()
    print("\nOK: arrays ready for the exit-speed classifier / HMM emissions.")
