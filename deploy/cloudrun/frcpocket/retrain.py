"""Retrain both models in one go: the aesthetic scorer, then the rib generator against it.

Uses each module's defaults (all grades in data/curation, both training sets built in
parallel) and prints how long each stage took. The whole run is meant to finish in
under 20 minutes on a laptop.

    python -I -m frcpocket.retrain
"""

import argparse
import json
import time

from frcpocket import scorer, trainGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer-out", default="models/aestheticScorer.pt")
    parser.add_argument("--plates", default="data/processed/plates.jsonl")
    args = parser.parse_args()
    started = time.time()
    report = scorer.train(args.scorer_out, args.plates, cnnEpochs=20, parallel=scorer.defaultParallel())
    scorerSeconds = time.time() - started
    print(f"scorer: {scorerSeconds / 60:.1f} min, graded holdout {report['gradedHoldoutAcc']}, gold holdout {report['goldHoldoutAcc']}",
          flush=True)
    generatorArgs = trainGenerator.buildParser().parse_args(["--scorer", args.scorer_out, "--plates", args.plates])
    trainGenerator.train(generatorArgs)
    total = time.time() - started
    print(json.dumps({"scorerMinutes": round(scorerSeconds / 60, 1), "generatorMinutes": round((total - scorerSeconds) / 60, 1),
                      "totalMinutes": round(total / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
