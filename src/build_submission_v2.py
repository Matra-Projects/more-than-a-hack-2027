"""
Orchestrator v2 - CHALLENGE.md control physics + dynamic transitions.

    python src/build_submission_v2.py                 # -> evaluation/results/team_engine_v2.csv
    python src/build_submission_v2.py --no-turnover   # same, clean-turnover P(stay) rule off

  Stage 1  deadball_engine_v2   module_1 rules + v2 out-of-bounds + v2 set-piece windows
  Stage 2  possession_engine_v2 control regimes (strong / weak / deflection / loose / contested)
           hmm_engine_v2        frame-dependent P(stay), missing-ball interception (2 passes)

A separate entry point: src/build_submission.py (-> team_engine_hmm.csv) is the frozen
path and is not modified. Tactical source only; the same audit hook is reused.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, DATA_DIR  # noqa: E402
import build_submission as frozen  # noqa: E402  (only its audit hook is reused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "team_engine_v2.csv"))
    ap.add_argument("--no-turnover", action="store_true",
                    help="disable the clean-turnover P(stay)=0.85 rule")
    args = ap.parse_args()
    protected = {"team_engine_hmm.csv", "team_engine_hmm_speed.csv"}
    if os.path.basename(args.out) in protected:
        sys.exit("refusing to overwrite a protected submission: " + args.out)

    frozen._install_audit(args.data_dir or DATA_DIR)

    import hmm_engine
    import ball_speed
    import deadball_engine_v2 as db2
    import hmm_engine_v2 as h2
    deadball_engine, _ = hmm_engine.import_engines()
    if args.no_turnover:
        h2.P_STAY_TURNOVER = h2.P_STAY_DEFAULT

    t0 = time.time()
    print("[1/4] loading tactical ATD + homography (full match) ...")
    trk = deadball_engine.load_tracking(args.game, args.data_dir)
    print("[2/4] ball speed + homography trust gate (ball_speed.py) ...")
    bx, by, speed, _trusted = ball_speed.build_speed_and_trust(args.game, args.data_dir)
    print("[3/4] STAGE 1 deadball_engine_v2 ...")
    dead = db2.build_dead_mask_v2(trk, bx, by, speed, args.game, args.data_dir)
    print("[4/4] STAGE 2 possession_engine_v2 + hmm_engine_v2 ...")
    labels, _diag = h2.decode_v2(trk, bx, by, speed, dead, args.game, args.data_dir)

    graded = deadball_engine.graded_mask(args.game)
    labels[~graded] = "X"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    n = len(labels)
    trap = labels[db2.SECOND_HALF_GRADED_START:db2.SECOND_HALF_KICKOFF]
    print("\nwrote {}  ({} rows)  in {:.0f}s".format(args.out, n, time.time() - t0))
    for c in "HADX":
        k = int((labels == c).sum())
        print("  {} {:>7}  ({:.1f}%)".format(c, k, 100.0 * k / n))
    print("sanity:")
    print("  graded labels                     : {}".format(sorted(set(labels[graded]))))
    print("  2nd-half trap 110808-113741 all D : {}".format(bool(np.all(trap == "D"))))
    print("  graded frames labelled X          : {}".format(int((labels[graded] == "X").sum())))
    print("  dead frames not D                 : {}".format(int((labels[dead & graded] != "D").sum())))
    print("  data files opened (tactical only) :")
    for p in sorted(frozen._OPENED):
        print("    data/" + p)
    assert all(p.startswith("tactical" + os.sep) for p in frozen._OPENED), "non-tactical data opened"


if __name__ == "__main__":
    main()
