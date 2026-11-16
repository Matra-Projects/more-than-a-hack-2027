"""
Legacy orchestrator (non-HMM path) - Stage 1 deadball (module_1) then Module 2's
causal dwell-hysteresis labeller (module_2), writes the 199,319-row CSV.
The submission pipeline is src/build_submission.py (Stage 1 -> HMM); this one is
kept as the reference path.

    python src/build_csv.py                # -> evaluation/results/team_engine.csv
    python src/build_csv.py --out my.csv

Then ALWAYS:
    python evaluation/validate_submission.py evaluation/results/team_engine.csv
    python evaluation/run_evaluation.py
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, half_video_blocks  # noqa: E402

# the engines live in module_1.py / module_2.py; module_2 imports its sibling as
# `deadball_engine`, so register both aliases before importing it
import module_1  # noqa: E402
sys.modules.setdefault("deadball_engine", module_1)
import module_2  # noqa: E402
sys.modules.setdefault("possession_engine", module_2)
from module_1 import load_tracking, players_by_team, build_dead_mask, graded_mask  # noqa: E402
from module_2 import build_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "team_engine.csv"))
    args = ap.parse_args()

    print("[1/3] loading tactical ATD + homography ...")
    trk = load_tracking(args.game, args.data_dir)
    bx, by, bs = trk["ball_x"], trk["ball_y"], trk["ball_s"]
    hx, hy, ax, ay = players_by_team(trk)

    print("[2/3] deadball engine (Module 1) ...")
    dead = build_dead_mask(trk, args.game, args.data_dir)

    print("[3/3] possession engine (Module 2) ...")
    labels = build_labels(bx, by, bs, hx, hy, ax, ay, dead, args.game, args.data_dir)

    # contract: ungraded frames -> X, graded -> H/A/D (never leave a graded frame X
    # unless it is in the pre-kickoff trap, which Module 1 owns)
    graded = graded_mask(args.game)
    labels[~graded] = "X"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        for i, lab in enumerate(labels):
            fh.write("{},{}\n".format(i, lab))

    n = len(labels)
    counts = {c: int((labels == c).sum()) for c in "HADX"}
    print("\nwrote {}  ({} rows)".format(args.out, n))
    for c in "HADX":
        print("  {} {:>7}  ({:.1f}%)".format(c, counts[c], 100.0 * counts[c] / n))
    print("\nnext: python evaluation/validate_submission.py {}".format(args.out))
    print("       python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
