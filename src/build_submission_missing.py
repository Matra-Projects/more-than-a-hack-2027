"""
Orchestrator - Stage 1 dead mask + missing-ball engine + unchanged HMM decode.

    python src/build_submission_missing.py
        -> evaluation/results/team_engine_hmm_missing.csv   (labels H / A / D, X ungraded)
    python src/build_submission_missing.py --tags-out /some/where/tags.csv
        -> also writes the INTERNAL missing-ball tags (CARRY_H, AIR_A, LANDED_H, ...)
           for inspection. Keep that file OUT of evaluation/results/: every CSV
           there is scored as a submission.

Does not touch src/build_submission.py or team_engine_hmm.csv (frozen).
Tactical source only; same audit hook as build_submission.py.
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
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                  "team_engine_hmm_missing.csv"))
    ap.add_argument("--tags-out", default=None, help="optional CSV of internal tags")
    args = ap.parse_args()
    results_dir = os.path.realpath(os.path.join(ROOT, "evaluation", "results"))
    if args.tags_out and os.path.realpath(os.path.dirname(os.path.abspath(args.tags_out))) == results_dir:
        sys.exit("--tags-out must not be in evaluation/results/ (it would be scored)")

    frozen._install_audit(args.data_dir or DATA_DIR)

    import hmm_engine
    import missing_ball_engine as mb
    deadball_engine, _ = hmm_engine.import_engines()

    t0 = time.time()
    print("[1/3] loading tactical ATD + homography (full match) ...")
    trk = deadball_engine.load_tracking(args.game, args.data_dir)

    print("[2/3] STAGE 1 deadball engine (module_1, unchanged) ...")
    dead1 = deadball_engine.build_dead_mask(trk, args.game, args.data_dir)

    print("[3/3] STAGE 2 missing-ball engine + HMM decode ...")
    labels, dead, tags, _stats = mb.decode_missing(trk, dead1, args.game, args.data_dir)

    graded = deadball_engine.graded_mask(args.game)
    labels[~graded] = "X"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    if args.tags_out:
        names = np.array(mb.TAGS)[tags]
        with open(args.tags_out, "w") as fh:
            fh.write("frame,tag,label\n")
            fh.writelines("{},{},{}\n".format(i, t, lab)
                          for i, (t, lab) in enumerate(zip(names, labels)))
        print("internal tags written to", args.tags_out)

    n = len(labels)
    trap = labels[deadball_engine.SECOND_HALF_GRADED_START:deadball_engine.SECOND_HALF_KICKOFF]
    print("\nwrote {}  ({} rows)  in {:.0f}s".format(args.out, n, time.time() - t0))
    for c in "HADX":
        k = int((labels == c).sum())
        print("  {} {:>7}  ({:.1f}%)".format(c, k, 100.0 * k / n))
    print("sanity:")
    print("  graded labels                     : {}".format(sorted(set(labels[graded]))))
    print("  2nd-half trap 110808-113741 all D : {}".format(bool(np.all(trap == "D"))))
    print("  graded frames labelled X          : {}".format(int((labels[graded] == "X").sum())))
    print("  dead frames not D                 : {}".format(int((labels[dead] != "D").sum())))
    print("  data files opened (tactical only) :")
    for p in sorted(frozen._OPENED):
        print("    data/" + p)
    assert all(p.startswith("tactical" + os.sep) for p in frozen._OPENED), "non-tactical data opened"


if __name__ == "__main__":
    main()
