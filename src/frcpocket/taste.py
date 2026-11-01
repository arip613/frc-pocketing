"""Taste model: what the expert rewards in a pocketed plate.

A linear model on the scorer's design features (designFeatures.describe) plus the
scorer's own looks score, fitted to the expert's preferences only (the user's call, 2026-11-01:
his taste is what matters): his 1-9 ratings, his A-vs-B picks, and his 1-9 ratings of the
live tool's own output (rounds L1f and L1x; L2f holds the v5 chooser's), which form their
own group with its own share of the weight (--live-share). Ratings are compared within a plate (a 2+ gap) and, with --cross-plate,
across plates too (same rater). --people user,expert brings the user's grades and picks back
(each person then carries equal weight). Claude's grades are never used.
It ranks designs of one plate, so only feature differences within a plate matter.

    python -I -m frcpocket.taste train --out models/marathon/v5/taste.json

Training reports, per source, how often the model agrees with preferences on plates it
did not see (5 folds by plate), next to the looks score alone, material removed alone, the
old hand ranking and the live chooser; and the expert's average rating on design A: on live
plates where he rated two or more of the designs shown, the one each ranking would put first.
A taste .json can stand in for a scorer .pt anywhere a Judge takes one.
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
pickRounds = ("p1", "p2", "p4", "p5", "p6")  # the user's blind picks (favourite 5, the rest 3)
keptRounds = ("k1",)  # a plate the user kept (5) against our designs for the same plate (3)
liveRounds = ("L1f", "L1x", "L2f")  # the live tool's and the v5 chooser's designs, rated 1-9 by the expert
# Chosen by his mean rating of the design shown first (design A), with taste scores held out by
# plate, on the 56 plates where he rated 2+ designs: cross-plate comparisons ("live" or "all")
# lowered it and lowered agreement with his within-plate preferences, so his live ratings count
# within a plate, like his other ratings.
liveShare = 0.0  # share of the total weight his live ratings carry (0: they count like his other ratings)
people = ("expert",)  # whose preferences train the model
crossPlate = "none"  # which of his ratings are also compared across plates: "none", "live" or "all"
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


def preferences(entries, who=None, across=None):
    """(better id, worse id, source) for every preference the chosen people stated."""
    who, across = who or people, across or crossPlate
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
    roundOf = lambda designId: designId.split(":")[1]
    addGraded("user grades", [x for x in user if roundOf(x[0]) not in pickRounds + keptRounds], userGradeGap)
    addGraded("user picks", [x for x in user if roundOf(x[0]) in pickRounds], userGradeGap)
    addGraded("user kept", [x for x in user if roundOf(x[0]) in keptRounds], userGradeGap)
    byRater, liveByRater = defaultdict(list), defaultdict(list)
    for line in expertRatingsPath.read_text().splitlines():
        rating = json.loads(line)
        if rating["id"] in entries:
            target = liveByRater if roundOf(rating["id"]) in liveRounds else byRater
            target[rating["rater"]].append((rating["id"], rating["rating"]))
    def addAcross(source, rated):
        for (a, ga), (b, gb) in itertools.combinations(rated, 2):
            if abs(ga - gb) >= expertRatingGap:
                out.append((a, b, source) if ga > gb else (b, a, source))

    for rated in byRater.values():
        (addAcross if across == "all" else lambda s, r: addGraded(s, r, expertRatingGap))("expert ratings", rated)
    for rated in liveByRater.values():
        (addAcross if across in ("all", "live") else lambda s, r: addGraded(s, r, expertRatingGap))("expert live", rated)
    for line in expertPairsPath.read_text().splitlines():
        pick = json.loads(line)
        if pick["winner"] in ("a", "b") and pick["a"] in entries and pick["b"] in entries:
            better, worse = (pick["a"], pick["b"]) if pick["winner"] == "a" else (pick["b"], pick["a"])
            out.append((better, worse, "expert picks"))
    out = [p for p in out if groupOf(p[2]).split()[0] in who]
    return [p for p in out if not isHoldout(entries[p[0]]["plateId"]) and not isHoldout(entries[p[1]]["plateId"])]


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


def designMatrix(entries, designIds, parallel=4, looksPath=looksScorerPath):
    """designId -> feature row (design features + looks), using a looks scorer (the live one by default)."""
    from frcpocket.scorer import AestheticScorer

    looksScorer = AestheticScorer(looksPath)  # also loads the gold style reference describe uses
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


def groupOf(source):
    return "expert live" if source == "expert live" else source.split()[0]


def sourceWeights(sources, share=None):
    """His ratings of live designs carry `share` of the total weight; the other groups present
    (the user, the expert's other ratings and picks) split the rest evenly. With share 0 his
    live ratings count like his other ratings (one group)."""
    share = liveShare if share is None else share
    group = np.array([groupOf(source) for source in sources])
    if share <= 0:
        group[group == "expert live"] = "expert"
    present = [g for g in ("user", "expert", "expert live") if (group == g).any()]
    others = [g for g in present if g != "expert live"]
    if not others:
        shares = {"expert live": 1.0}
    elif "expert live" not in present or share <= 0:
        shares = {g: 1.0 for g in others}
    else:
        shares = {"expert live": share, **{g: (1 - share) / len(others) for g in others}}
    total = sum(shares.values())
    weights = np.zeros(len(sources))
    for g, part in shares.items():
        mask = group == g
        weights[mask] = part / total * len(sources) / mask.sum()
    return weights


def liveRatings(entries):
    """plateKey -> [(designId, his mean rating)] for live designs (liveRounds) he rated."""
    ratings = defaultdict(list)
    for line in expertRatingsPath.read_text().splitlines():
        rating = json.loads(line)
        if rating["id"] in entries and rating["id"].split(":")[1] in liveRounds and not isHoldout(entries[rating["id"]]["plateId"]):
            ratings[rating["id"]].append(rating["rating"])
    byPlate = defaultdict(list)
    for designId, values in ratings.items():
        byPlate[entries[designId]["plateKey"]].append((designId, float(np.mean(values))))
    return byPlate


def ratingOnA(byPlate, score):
    """His mean rating of the design `score` ranks first, over plates with 2+ rated designs."""
    picks = [max(designs, key=lambda item: score(item[0]))[1] for designs in byPlate.values() if len(designs) >= 2]
    return round(float(np.mean(picks)), 2) if picks else None


def ruleChecks(entries, prefs, rows):
    """How often his within-plate preferences side with the lighter design, and on symmetric
    plates (plate symmetry 0.9+, the live rule's threshold) with the more symmetric one."""
    removed, shortfall = featureNames.index("removedFraction"), featureNames.index("symmetryShortfall")
    within = [(a, b) for a, b, _ in prefs if entries[a]["plateKey"] == entries[b]["plateKey"]]
    plateSym = {}
    for a, _ in within:
        key = entries[a]["plateKey"]
        if key not in plateSym:
            plateSym[key] = designFeatures.plateSymmetry(plateOf(entries[a]))
    lighter = [rows[a][removed] > rows[b][removed] for a, b in within if abs(rows[a][removed] - rows[b][removed]) >= 0.02]
    symmetric = [rows[a][shortfall] < rows[b][shortfall] for a, b in within
                 if plateSym[entries[a]["plateKey"]] >= 0.9 and abs(rows[a][shortfall] - rows[b][shortfall]) >= 0.01]
    share = lambda xs: (round(float(np.mean(xs)), 3) if xs else None, len(xs))
    return {"lighter wins": share(lighter), "more symmetric wins (symmetric plates)": share(symmetric)}


def train(out, parallel=4, looksPath=looksScorerPath, share=None, who=None, across=None):
    entries = sheetEntries()
    prefs = preferences(entries, who, across)
    live = liveRatings(entries)
    liveIds = {designId for designs in live.values() for designId, _ in designs}
    ids = sorted({designId for a, b, _ in prefs for designId in (a, b)} | liveIds)
    print(f"{len(prefs)} preferences over {len(ids)} designs:",
          dict(sorted(defaultdict(int, {s: sum(p[2] == s for p in prefs) for s in {p[2] for p in prefs}}).items())), flush=True)
    rows = designMatrix(entries, ids, parallel, looksPath)
    matrix = np.stack([rows[designId] for designId in ids])
    scale = matrix.std(axis=0) + 1e-6
    mean = matrix.mean(axis=0)
    better = np.stack([rows[a] for a, _, _ in prefs]) / scale
    worse = np.stack([rows[b] for _, b, _ in prefs]) / scale
    differences = better - worse
    sources = [source for _, _, source in prefs]
    foldA = np.array([foldOf(entries[a]["plateKey"]) for a, _, _ in prefs])
    foldB = np.array([foldOf(entries[b]["plateKey"]) for _, b, _ in prefs])
    raw = lambda ids_, k: np.array([rows[i][k] for i in ids_])
    looksIndex, removedIndex = featureNames.index("looks"), featureNames.index("removedFraction")
    baselines = {
        "looks only": raw([a for a, _, _ in prefs], looksIndex) > raw([b for _, b, _ in prefs], looksIndex),
        "removed only": raw([a for a, _, _ in prefs], removedIndex) > raw([b for _, b, _ in prefs], removedIndex),
        "live ranking": (0.7 * raw([a for a, _, _ in prefs], looksIndex) + 0.15 * np.minimum(raw([a for a, _, _ in prefs], removedIndex) / 0.4, 1)
                         > 0.7 * raw([b for _, b, _ in prefs], looksIndex) + 0.15 * np.minimum(raw([b for _, b, _ in prefs], removedIndex) / 0.4, 1)),
    }
    # The live tool's order: its A, B, C (rounds L1f, L1x); the v5 chooser's designs (L2f) come last.
    onA = {"live order (as shown)": ratingOnA(live, lambda d: -int(d.split(":")[-1]) if d.split(":")[1] != "L2f" else -99),
           "lightest": ratingOnA(live, lambda d: rows[d][removedIndex]),
           "best possible": ratingOnA(live, lambda d: next(r for ds in live.values() for i, r in ds if i == d))}
    livePath = Path("models/marathon/v2/taste.json")  # the chooser live now, when it builds on the same looks scorer
    if livePath.exists() and json.loads(livePath.read_text())["looksScorer"] == str(looksPath):
        liveModel = json.loads(livePath.read_text())
        liveScore = lambda ids_: np.stack([rows[i] for i in ids_]) / np.array(liveModel["scale"]) @ np.array(liveModel["weights"])
        baselines["live taste (v2)"] = liveScore([a for a, _, _ in prefs]) > liveScore([b for _, b, _ in prefs])
        onA["live taste (v2)"] = ratingOnA(live, lambda d: float(liveScore([d])[0]))
    report, outOfFold = {}, {}
    # A pair is held out when both its plates are in the fold; it trains only when neither is.
    tested = np.zeros(len(prefs), dtype=bool)
    for penalty in penalties:
        hits, scored = np.zeros(len(prefs), dtype=bool), {}
        for fold in range(folds):
            train_, test = (foldA != fold) & (foldB != fold), (foldA == fold) & (foldB == fold)
            w = fitLogistic(differences[train_], sourceWeights([s for s, t in zip(sources, train_) if t], share), penalty)
            hits[test] = differences[test] @ w > 0
            tested |= test
            scored.update({d: float(rows[d] / scale @ w) for d in liveIds if foldOf(entries[d]["plateKey"]) == fold})
        report[penalty], outOfFold[penalty] = hits, scored
    for hits in list(report.values()) + list(baselines.values()):
        hits &= tested  # pairs never held out (their plates in different folds) don't count
    def bySource(hits):
        return {source: round(float(hits[(np.array(sources) == source) & tested].mean()), 3) for source in sorted(set(sources))}
    def balanced(hits):
        group = np.array([groupOf(s) for s in sources])
        if (liveShare if share is None else share) <= 0:
            group[group == "expert live"] = "expert"
        return round(float(np.mean([hits[(group == g) & tested].mean() for g in sorted(set(group))])), 3)
    best = max(penalties, key=lambda penalty: balanced(report[penalty]))
    onA["taste model (held out)"] = ratingOnA(live, lambda d: outOfFold[best][d])
    summary = {"heldOut": {"taste model": {**bySource(report[best]), "balanced": balanced(report[best])},
                           **{name: {**bySource(h), "balanced": balanced(h)} for name, h in baselines.items()}},
               "ratingOnA": {"plates": sum(len(ds) >= 2 for ds in live.values()), **onA},
               "rules": ruleChecks(entries, prefs, rows),
               "penalty": best, "preferences": len(prefs), "designs": len(ids),
               "people": list(who or people), "crossPlate": across or crossPlate}
    for name, row in summary["heldOut"].items():
        print(f"  {name:12s} {row}", flush=True)
    print("  his rating on design A:", summary["ratingOnA"], flush=True)
    print("  his within-plate preferences:", summary["rules"], flush=True)
    weights = fitLogistic(differences, sourceWeights(sources, share), best)
    scores = ((matrix - mean) / scale) @ weights
    model = {"featureNames": featureNames, "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
             "scoreSpread": float(scores.std() + 1e-6), "looksScorer": str(looksPath), "report": summary,
             "liveShare": liveShare if share is None else share}
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
    parser.add_argument("--out", default="models/marathon/v5/taste.json")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--looks", default=looksScorerPath, help="looks scorer the model builds on")
    parser.add_argument("--live-share", type=float, default=liveShare, help="weight share of the expert's live ratings")
    parser.add_argument("--people", default=",".join(people), help="whose preferences train it: expert, or user,expert")
    parser.add_argument("--cross-plate", default=crossPlate, choices=["none", "live", "all"],
                        help="which of his ratings are also compared across plates")
    args = parser.parse_args()
    train(args.out, args.parallel, args.looks, args.live_share, tuple(args.people.split(",")), args.cross_plate)


if __name__ == "__main__":
    main()
