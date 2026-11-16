"""
Orchestrator - frozen two-stage pipeline + the ADDITIONAL speed-evidence stage.

    python src/build_submission_speed.py     # -> evaluation/results/team_engine_hmm_speed.csv

  Stage 1  module_1 dead mask (unchanged)
  Stage 2a hmm_engine decode (unchanged) -> last-possession team per gap
  Stage 2b speed_evidence: ball speed from raw ATD positions, homography trust
           gate, exit-speed classifier, reception prediction, reappearance check
           -> llr_home / llr_away added to the emissions -> same Viterbi core

src/build_submission.py (-> team_engine_hmm.csv) is a separate entry point and is
not modified, so the frozen CSV is reproducible bit for bit.
Tactical source only; the same audit hook as build_submission.py.
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
                                                  "team_engine_hmm_speed.csv"))
    args = ap.parse_args()

    frozen._install_audit(args.data_dir or DATA_DIR)

    import hmm_engine
    import speed_evidence
    deadball_engine, _ = hmm_engine.import_engines()

    t0 = time.time()
    print("[1/3] loading tactical ATD + homography (full match) ...")
    trk = deadball_engine.load_tracking(args.game, args.data_dir)

    print("[2/3] STAGE 1 deadball engine (module_1, unchanged) ...")
    dead1 = deadball_engine.build_dead_mask(trk, args.game, args.data_dir)

    print("[3/3] STAGE 2 HMM + speed evidence (two-pass) ...")
    labels, dead, ev = speed_evidence.decode_speed(trk, dead1, args.game, args.data_dir)

    graded = deadball_engine.graded_mask(args.game)
    labels[~graded] = "X"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    n = len(labels)
    trap = labels[deadball_engine.SECOND_HALF_GRADED_START:deadball_engine.SECOND_HALF_KICKOFF]
    print("\nwrote {}  ({} rows)  in {:.0f}s".format(args.out, n, time.time() - t0))
    for c in "HADX":
        k = int((labels == c).sum())
        print("  {} {:>7}  ({:.1f}%)".format(c, k, 100.0 * k / n))
    print("sanity:")
    print("  2nd-half trap 110808-113741 all D : {}".format(bool(np.all(trap == "D"))))
    print("  graded frames labelled X          : {}".format(int((labels[graded] == "X").sum())))
    print("  dead frames not D                 : {}".format(int((labels[dead] != "D").sum())))
    print("  NaN in Viterbi log-likelihoods    : none (asserted in emission_loglik_speed)")
    print("  data files opened (tactical only) :")
    for p in sorted(frozen._OPENED):
        print("    data/" + p)
    assert all(p.startswith("tactical" + os.sep) for p in frozen._OPENED), "non-tactical data opened"


if __name__ == "__main__":
    main()
