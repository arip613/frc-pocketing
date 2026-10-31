"""Taste model: what the user and the expert reward in a pocketed plate.

A linear model on the scorer's design features (designFeatures.describe) plus the
scorer's own looks score, fitted only to people's within-plate preferences: the user's
sheet grades and model-set picks (rounds p1, p2), and the expert's 1-9 ratings and A-vs-B
picks. Claude's grades are left out (cross-validated, they made the scorer agree with the
expert less). The user's and the expert's preferences carry equal total weight. It ranks
designs of one plate, so only feature differences within a plate matter.

    python -I -m frcpocket.taste train --out models/marathon/v2/taste.json

Training reports, per source, how often the model agrees with preferences on plates it
did not see (5 folds by plate), next to the looks score alone, material removed alone and
the live ranking. A taste .json can stand in for a scorer .pt anywhere a Judge takes one.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from frcpocket import designFeatures
from frcpocket.plate import Plate, PocketParams
from frcpocket.splits import isHoldout

sheetRoot = Path("data/curation/sheets")
userGradesPath = Path("data/curation/userGrades.json")
expertRatingsPath = Path("data/curation/expertRatings.jsonl")
expertPairsPath = Path("data/curation/expertPairs.jsonl")
looksScorerPath = "models/aestheticScorer.pt"
featureNames = designFeatures.featureNames + ["looks"]
pickRounds = ("p1", "p2")  # the user's model-set picks (favourite 5, the rest 3)
userGradeGap = 1.0  # within-plate grade gaps that count as a preference
expertRatingGap = 2  # on his 1-9 scale
folds = 5
penalties = (0.3, 1.0, 3.0, 10.0)  # L2 strengths tried; the best mean held-out agreement wins


def sheetEntries():
    entries = {}
    for folder in sorted(sheetRoot.iterdir()):
        path = folder / "candidates.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                entries[f"sheet:{folder.name}:{entry['sheet']}:{entry['candidate']}"] = entry
    return entries


def preferences(entries):
    """(better id, worse id, source) for every within-plate preference people stated."""
    out = []

    def addGraded(source, graded, gap):
        byPlate = defaultdict(list)
        for designId, grade in graded:
            byPlate[entries[designId]["plateKey"]].append((designId, grade))
        for designs in byPlate.values():
            for (a, ga), (b, gb) in itertools.combinations(designs, 2):
                if abs(ga - gb) >= gap:
                    out.append((a, b, source) if ga > gb else (b, a, source))

    user = []
    for key, grade in json.loads(userGradesPath.read_text()).items():
        roundName, sheet, candidate = key.split("__")
        designId = f"sheet:{roundName}:{sheet}:{candidate}"
        if designId in entries:
            user.append((designId, grade))
    addGraded("user grades", [x for x in user if x[0].split(":")[1] not in pickRounds], userGradeGap)
    addGraded("user picks", [x for x in user if x[0].split(":")[1] in pickRounds], userGradeGap)
    byRater = defaultdict(list)
    for line in expertRatingsPath.read_text().splitlines():
        rating = json.loads(line)
        if rating["id"] in entries:
            byRater[rating["rater"]].append((rating["id"], rating["rating"]))
    for rated in byRater.values():
        addGraded("expert ratings", rated, expertRatingGap)
    for line in expertPairsPath.read_text().splitlines():
        pick = json.loads(line)
        if pick["winner"] in ("a", "b") and pick["a"] in entries and pick["b"] in entries:
            better, worse = (pick["a"], pick["b"]) if pick["winner"] == "a" else (pick["b"], pick["a"])
            out.append((better, worse, "expert picks"))
    return [p for p in out if not isHoldout(entries[p[0]]["plateId"])]


def plateOf(entry):
    return Plate(outline=Polygon(entry["outline"]), features=[Polygon(hole) for hole in entry["holes"]],
                 thicknessIn=entry.get("thicknessIn") or 0.25)


def paramsOf(entry):
    return PocketParams(**entry["params"]) if entry.get("params") else PocketParams()


def pocketsOf(entry):
    return [polygon if polygon.is_valid else polygon.buffer(0) for polygon in map(Polygon, entry["pockets"])]


def describeJob(styleReference, items):
    """Design features for the designs of one plate (in a worker process)."""
    designFeatures.styleReference.update(styleReference)
    plate = plateOf(items[0][1])
    symmetry = designFeatures.plateSymmetry(plate)
    return [(designId, designFeatures.describe(plate, paramsOf(entry), pocketsOf(entry), symmetry)) for designId, entry in items]


def designMatrix(entries, designIds, parallel=4):
    """designId -> feature row (design features + looks), using the live scorer for looks."""
    from frcpocket.scorer import AestheticScorer

    looksScorer = AestheticScorer(looksScorerPath)  # also loads the gold style reference describe uses
    byPlate = defaultdict(list)
    for designId in designIds:
        byPlate[entries[designId]["plateKey"]].append((designId, entries[designId]))
    rows = {}
    with ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(describeJob, dict(designFeatures.styleReference), items) for items in byPlate.values()]
        for future in futures:
            rows.update(dict(future.result()))
    looks = {}
    for items in byPlate.values():
        byParams = defaultdict(list)
        for designId, entry in items:
            byParams[json.dumps(entry.get("params"), sort_keys=True)].append((designId, entry))
        for group in byParams.values():
            plate, params = plateOf(group[0][1]), paramsOf(group[0][1])
            scores = looksScorer.score(plate, params, [pocketsOf(entry) for _, entry in group])
            looks.update({designId: float(score) for (designId, _), score in zip(group, scores)})
    return {designId: np.nan_to_num(np.append(rows[designId], looks[designId])) for designId in designIds}


def fitLogistic(differences, weights, penalty, iterations=50):
    """Weights w with P(better beats worse) = sigmoid(w . (f_better - f_worse)), L2-penalised (Newton)."""
    w = np.zeros(differences.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-differences @ w))
        gradient = differences.T @ (weights * (p - 1)) + penalty * w
        hessian = (differences * (weights * p * (1 - p))[:, None]).T @ differences + penalty * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def foldOf(plateKey):
    return int(hashlib.sha1(plateKey.encode()).hexdigest(), 16) % folds


def sourceWeights(sources):
    """The user's and the expert's preferences carry equal total weight."""
    person = np.array([source.split()[0] for source in sources])
    weights = np.ones(len(sources))
    for who in ("user", "expert"):
        mask = person == who
        if mask.any():
            weights[mask] = 0.5 * len(sources) / mask.sum()
    return weights


def train(out, parallel=4):
    entries = sheetEntries()
    prefs = preferences(entries)
    ids = sorted({designId for a, b, _ in prefs for designId in (a, b)})
    print(f"{len(prefs)} preferences over {len(ids)} designs:",
          dict(sorted(defaultdict(int, {s: sum(p[2] == s for p in prefs) for s in {p[2] for p in prefs}}).items())), flush=True)
    rows = designMatrix(entries, ids, parallel)
    matrix = np.stack([rows[designId] for designId in ids])
    scale = matrix.std(axis=0) + 1e-6
    mean = matrix.mean(axis=0)
    better = np.stack([rows[a] for a, _, _ in prefs]) / scale
    worse = np.stack([rows[b] for _, b, _ in prefs]) / scale
    differences = better - worse
    sources = [source for _, _, source in prefs]
    foldIds = np.array([foldOf(entries[a]["plateKey"]) for a, _, _ in prefs])
    raw = lambda ids_, k: np.array([rows[i][k] for i in ids_])
    looksIndex, removedIndex = featureNames.index("looks"), featureNames.index("removedFraction")
    baselines = {
        "looks only": raw([a for a, _, _ in prefs], looksIndex) > raw([b for _, b, _ in prefs], looksIndex),
        "removed only": raw([a for a, _, _ in prefs], removedIndex) > raw([b for _, b, _ in prefs], removedIndex),
        "live ranking": (0.7 * raw([a for a, _, _ in prefs], looksIndex) + 0.15 * np.minimum(raw([a for a, _, _ in prefs], removedIndex) / 0.4, 1)
                         > 0.7 * raw([b for _, b, _ in prefs], looksIndex) + 0.15 * np.minimum(raw([b for _, b, _ in prefs], removedIndex) / 0.4, 1)),
    }
    report = {}
    for penalty in penalties:
        hits = np.zeros(len(prefs), dtype=bool)
        for fold in range(folds):
            train_, test = foldIds != fold, foldIds == fold
            w = fitLogistic(differences[train_], sourceWeights([s for s, t in zip(sources, train_) if t]), penalty)
            hits[test] = differences[test] @ w > 0
        report[penalty] = hits
    def bySource(hits):
        return {source: round(float(hits[np.array(sources) == source].mean()), 3) for source in sorted(set(sources))}
    def balanced(hits):
        people = np.array([s.split()[0] for s in sources])
        return round(float(np.mean([hits[people == who].mean() for who in ("user", "expert")])), 3)
    best = max(penalties, key=lambda penalty: balanced(report[penalty]))
    summary = {"heldOut": {"taste model": {**bySource(report[best]), "balanced": balanced(report[best])},
                           **{name: {**bySource(h), "balanced": balanced(h)} for name, h in baselines.items()}},
               "penalty": best, "preferences": len(prefs), "designs": len(ids)}
    for name, row in summary["heldOut"].items():
        print(f"  {name:12s} {row}", flush=True)
    weights = fitLogistic(differences, sourceWeights(sources), best)
    scores = ((matrix - mean) / scale) @ weights
    model = {"featureNames": featureNames, "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
             "scoreSpread": float(scores.std() + 1e-6), "looksScorer": looksScorerPath, "report": summary}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(model, indent=1))
    order = np.argsort(-np.abs(weights))
    print("strongest terms:", ", ".join(f"{featureNames[k]} {weights[k]:+.2f}" for k in order[:8]), flush=True)
    return summary


class TasteScorer:
    """Scores designs of one plate like AestheticScorer.score, from a trained taste .json."""

    def __init__(self, path):
        from frcpocket.scorer import AestheticScorer

        model = json.loads(Path(path).read_text())
        self.looks = AestheticScorer(model["looksScorer"])
        self.mean, self.scale = np.array(model["mean"]), np.array(model["scale"])
        self.weights, self.spread = np.array(model["weights"]), model["scoreSpread"]

    def score(self, plate, params, pocketSets):
        looks = np.asarray(self.looks.score(plate, params, pocketSets), dtype=float)
        symmetry = designFeatures.plateSymmetry(plate)
        features = np.stack([designFeatures.describe(plate, params, pockets, symmetry) for pockets in pocketSets])
        z = (np.nan_to_num(np.column_stack([features, looks])) - self.mean) / self.scale @ self.weights
        return 1 / (1 + np.exp(-z / self.spread))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train"])
    parser.add_argument("--out", default="models/marathon/v2/taste.json")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    train(args.out, args.parallel)


if __name__ == "__main__":
    main()
