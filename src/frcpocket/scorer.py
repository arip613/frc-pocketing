"""Aesthetic scorer: learns what good FRC pocketing looks like from real gold plates.

Two models are trained with a pairwise ranking loss on same-plate pairs (good design
vs. bad design of the *same* plate, so the outline can't be used as a shortcut):
  - a feature model: small MLP on the scale-free design features (designFeatures.py)
  - a CNN on rasterized designs (raster.py)
Positives are the real gold designs plus their rib-graph projections. Plates the user
rated thumbs-up count double, and candidates rated in the UI join as extra same-plate
pairs (liked > disliked and > generated negatives). Negatives come from negatives.py. Held-out gold plates
measure ranking accuracy, and the final score blends the two models by how well each
ranks on the holdout.

    python -I -m frcpocket.scorer --out models/aestheticScorer.pt
"""

import argparse
import json
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frcpocket.layout import defaultGraph
from frcpocket import designFeatures, geom2d, negatives, raster
from frcpocket.evaluate import styleParamsFromRecord
from frcpocket.plate import plateFromRecord
from frcpocket.projection import projectRealDesign
from frcpocket.splits import goldIds, isHoldout, loadRecords, userRatings

torch.set_num_threads(4)
# The CNN sees rasters average-pooled by this factor (a quarter of the work per image at 2),
# and every pair each epoch: sampling fewer pairs per epoch left it barely trained.
cnnInputPool = 2


@dataclass
class PlateSet:
    plateId: str
    features: np.ndarray  # (designs, nFeatures)
    rasters: np.ndarray  # (designs, 3, H, W)
    isPositive: np.ndarray  # (designs,) bool
    kinds: list
    weight: float
    grades: np.ndarray = None  # (designs,) graded sets: every higher-graded design beats lower ones


def buildPlateSet(record, rng, perKind=4):
    plate = plateFromRecord(record)
    params = styleParamsFromRecord(record)
    real = [geom2d.toPolygon(pocket) for pocket in record["pockets"]]
    projection = projectRealDesign(plate, params, real)
    designs = [("real", real), ("projected", projection.layout.pockets)]
    designs += negatives.negativeSet(plate, params, projection.layout.graph, projection.layout, rng, perKind=perKind)
    plateSym = designFeatures.plateSymmetry(plate)
    toPixels = raster.plateFrame(plate)
    features = np.stack([designFeatures.describe(plate, params, pockets, plateSym) for _, pockets in designs])
    rasters = np.stack([raster.rasterize(plate, pockets, toPixels) for _, pockets in designs])
    kinds = [name for name, _ in designs]
    liked = userRatings().get(record["id"], 0) > 0
    return PlateSet(record["id"], features, rasters, np.array([kind in ("real", "projected") for kind in kinds]), kinds,
                    weight=2.0 if liked else 1.0)


def defaultParallel():
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def plateRng(seed, key):
    """A plate's own random stream, so the sets come out the same however they're scheduled."""
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def initWorker(styleReference):
    torch.set_num_threads(1)  # one core per worker; the pool spreads the plates
    designFeatures.styleReference.update(styleReference)  # the gold fit made in the main process


def runPlateJobs(groups, parallel):
    """{name: (job, [args, ...])} -> {name: [job(*args), ...]} in order, all queued on one
    pool of `parallel` spawned processes (or run here when parallel is 1)."""
    if parallel <= 1:
        return {name: [job(*args) for args in items] for name, (job, items) in groups.items()}
    with ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initWorker, initargs=(dict(designFeatures.styleReference),)) as pool:
        futures = {name: [pool.submit(job, *args) for args in items] for name, (job, items) in groups.items()}
        return {name: [future.result() for future in group] for name, group in futures.items()}


def goldSetJob(record, seed, perKind):
    return buildPlateSet(record, plateRng(seed, record["id"]), perKind)


def uiRatingSets(seed, ratingsPath=Path("data/curation/ratings.jsonl"), perKind=2):
    """Plate sets from thumbs up/down in the UI: liked candidates beat disliked ones and
    generated negatives on the same plate."""
    from shapely.geometry import Polygon

    from frcpocket.layout import candidateGraph
    from frcpocket.plate import Plate, PocketParams

    if not ratingsPath.exists():
        return []
    byPlate = {}
    for line in ratingsPath.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("kind") == "uiCandidate":
            byPlate.setdefault(entry["plateHash"], []).append(entry)
    sets = []
    for plateHash, entries in byPlate.items():
        liked = [entry for entry in entries if entry["rating"] > 0]
        if not liked:
            continue
        first = entries[0]
        plate = Plate(outline=Polygon(first["outline"]), features=[Polygon(hole) for hole in first["holes"]],
                      thicknessIn=first["thicknessIn"])
        params = PocketParams(**first["params"])
        designs = [("liked", [Polygon(p) for p in entry["pockets"]]) for entry in liked]
        designs += [("disliked", [Polygon(p) for p in entry["pockets"]]) for entry in entries if entry["rating"] < 0]
        designs += negatives.negativeSet(plate, params, defaultGraph(plate, params), None, plateRng(seed, plateHash),
                                         perKind=perKind)
        plateSym = designFeatures.plateSymmetry(plate)
        toPixels = raster.plateFrame(plate)
        sets.append(PlateSet(
            plateId=f"ui:{plateHash}",
            features=np.stack([designFeatures.describe(plate, params, pockets, plateSym) for _, pockets in designs]),
            rasters=np.stack([raster.rasterize(plate, pockets, toPixels) for _, pockets in designs]),
            isPositive=np.array([kind == "liked" for kind, _ in designs]),
            kinds=[kind for kind, _ in designs],
            weight=2.0,
        ))
    return sets


def sheetSetItems(seed, records, perKind=1, weight=2.0, claudeGrades=True):
    """Job arguments for every graded plate on the rating sheets (ratingSheets.py)."""
    from frcpocket import ratingSheets

    items = []
    for key, (plate, rated) in sorted(ratingSheets.loadSheetRatings(claudeGrades=claudeGrades).items()):
        plateId = rated[0][4]["plateId"]
        real = records[plateId]["pockets"] if plateId in records else []
        graded = [(params, pockets, grade) for params, pockets, _, grade, _ in rated]
        items.append((key, plateId, plate, graded, real, seed, perKind, weight))
    return items


def sheetSetJob(key, plateId, plate, graded, real, seed, perKind, weight):
    """One graded plate's set: the graded candidates, the real design on gold plates
    (grade 5), and a few generated bad layouts (grade 1)."""
    params = replace(graded[0][0], pocketSizeFactor=1.0)
    designs = [(f"graded{grade:g}", pockets, grade) for _, pockets, grade in graded]
    if real:
        designs.append(("real", [geom2d.toPolygon(p) for p in real], 5.0))
    designs += [(kind, pockets, 1.0) for kind, pockets in
                negatives.negativeSet(plate, params, defaultGraph(plate, params), None, plateRng(seed, key), perKind=perKind)]
    plateSym = designFeatures.plateSymmetry(plate)
    toPixels = raster.plateFrame(plate)
    return key, PlateSet(
        plateId=f"sheet:{plateId}",
        features=np.stack([designFeatures.describe(plate, params, pockets, plateSym) for _, pockets, _ in designs]),
        rasters=np.stack([raster.rasterize(plate, pockets, toPixels) for _, pockets, _ in designs]),
        isPositive=np.array([grade >= 4 for _, _, grade in designs]),
        kinds=[kind for kind, _, _ in designs],
        weight=weight,
        grades=np.array([grade for _, _, grade in designs], dtype=float),
    )


def isHeldOutSheet(key):
    """One graded plate in five is held out, by plate key, to measure how well the scorer
    learned the grader's eye."""
    return int(key, 16) % 5 == 0


def priorSetItems(seed, platesPath, count=30, weight=0.3):
    """Job arguments for the low-weight prior on non-gold plates: a plain truss beats
    gutting the plate or slotting it.

    Gold plates are all gearbox-style, so this tells the scorer what "don't gut it" means
    on gussets, arms, and brackets too, without making the busy truss look a positive
    anywhere a gold example exists.
    """
    from frcpocket.trainGenerator import loadPlates

    _, inputs, _ = loadPlates(platesPath)
    picks = np.random.default_rng(seed).choice(len(inputs), size=min(count, len(inputs)), replace=False)
    return [(inputs[int(index)], seed, weight) for index in picks]


def priorSetJob(record, seed, weight):
    from frcpocket.baseline import trussBaseline
    from frcpocket.trainGenerator import sampleParams

    rng = plateRng(seed, record["id"])
    plate = plateFromRecord(record)
    params = sampleParams(plate, rng)
    graph = defaultGraph(plate, params)
    designs = [("truss", trussBaseline(plate, params).pockets)]
    designs += [("gutted", negatives.gutted(plate, params, graph, rng, keepFraction=fraction)) for fraction in (0.0, 0.05)]
    designs.append(("ladder", negatives.ladder(plate, params, graph, rng)))
    designs = [(kind, pockets) for kind, pockets in designs if pockets]
    if len(designs) < 2 or designs[0][0] != "truss":
        return None
    plateSym = designFeatures.plateSymmetry(plate)
    toPixels = raster.plateFrame(plate)
    return PlateSet(
        plateId=f"prior:{record['id']}",
        features=np.stack([designFeatures.describe(plate, params, pockets, plateSym) for _, pockets in designs]),
        rasters=np.stack([raster.rasterize(plate, pockets, toPixels) for _, pockets in designs]),
        isPositive=np.array([kind == "truss" for kind, _ in designs]),
        kinds=[kind for kind, _ in designs],
        weight=weight,
    )


class FeatureModel(nn.Module):
    def __init__(self, nFeatures):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nFeatures, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CnnModel(nn.Module):
    def __init__(self, inputPool=1):
        super().__init__()
        self.inputPool = inputPool
        def block(cIn, cOut):
            return nn.Sequential(nn.Conv2d(cIn, cOut, 3, padding=1), nn.BatchNorm2d(cOut), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 48), block(48, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        if self.inputPool > 1:
            x = nn.functional.avg_pool2d(x, self.inputPool)
        return self.head(self.features(x)).squeeze(-1)


def dihedral(batch, rng):
    k = int(rng.integers(4))
    batch = torch.rot90(batch, k, dims=(2, 3))
    if rng.random() < 0.5:
        batch = torch.flip(batch, dims=(3,))
    return batch


def pairsOf(plateSets):
    """(plateIndex, betterIndex, worseIndex, weight) for every same-plate pair.

    Positive/negative sets pair every positive with every negative. Graded sets pair
    designs at least a grade apart, weighted up for bigger gaps."""
    pairs = []
    for plateIndex, plateSet in enumerate(plateSets):
        if plateSet.grades is not None:
            for better in range(len(plateSet.grades)):
                for worse in range(len(plateSet.grades)):
                    gap = plateSet.grades[better] - plateSet.grades[worse]
                    if gap >= 1:
                        pairs.append((plateIndex, better, worse, plateSet.weight * min(gap, 3.0) / 2))
            continue
        positives = np.flatnonzero(plateSet.isPositive)
        negativesIdx = np.flatnonzero(~plateSet.isPositive)
        for p in positives:
            for n in negativesIdx:
                pairs.append((plateIndex, int(p), int(n), plateSet.weight))
    return pairs


def trainModel(model, inputs, pairs, epochs, lr, rng, augment=False, weightDecay=1e-3, pairsPerEpoch=None):
    """Pairwise ranking loss over same-plate pairs. Designs are stacked into one tensor so
    a batch is gathered by index; pairsPerEpoch draws a fresh random subset each epoch
    (the CNN sees every pair over its epochs without paying for all of them each time)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    stacked = torch.cat(inputs)
    offsets = np.cumsum([0] + [len(x) for x in inputs])
    better = torch.tensor([offsets[p] + i for p, i, _, _ in pairs])
    worse = torch.tensor([offsets[p] + j for p, _, j, _ in pairs])
    weights = torch.tensor([float(w) for _, _, _, w in pairs])
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(pairs))[:pairsPerEpoch]
        for start in range(0, len(order), 64):
            batch = torch.from_numpy(order[start:start + 64])
            positives, negativesBatch = stacked[better[batch]], stacked[worse[batch]]
            if augment:
                positives, negativesBatch = dihedral(positives, rng), dihedral(negativesBatch, rng)
            margin = model(positives) - model(negativesBatch)
            loss = (nn.functional.softplus(-margin) * weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def pairAccuracy(model, inputs, pairs):
    model.eval()
    with torch.no_grad():
        scores = [model(stack).numpy() for stack in inputs]
    correct = [scores[p][i] > scores[p][j] for p, i, j, _ in pairs]
    return float(np.mean(correct)) if correct else float("nan"), scores


class AestheticScorer:
    """Loaded scorer: score(plate, params, pockets) -> 0..1 (higher looks better)."""

    def __init__(self, path):
        bundle = torch.load(path, weights_only=False)
        self.meta = bundle["meta"]
        self.featureMean = torch.tensor(bundle["featureMean"])
        self.featureStd = torch.tensor(bundle["featureStd"])
        # The features this bundle was trained on (older bundles have fewer).
        self.featureColumns = [designFeatures.featureNames.index(name) for name in self.meta["featureNames"]]
        self.featureModel = FeatureModel(len(self.featureColumns))
        self.featureModel.load_state_dict(bundle["featureModel"])
        self.cnnModel = CnnModel(inputPool=self.meta.get("cnnInputPool", 1))
        self.cnnModel.load_state_dict(bundle["cnnModel"])
        self.featureModel.eval()
        self.cnnModel.eval()
        self.blend = bundle["blend"]  # weight on the CNN, 0..1
        if "styleReference" in self.meta:
            designFeatures.styleReference.update(self.meta["styleReference"])
        self.calibration = bundle["calibration"]  # (mean, std) of blended logits

    def logits(self, plate, params, designs, plateSym=None):
        plateSym = designFeatures.plateSymmetry(plate) if plateSym is None else plateSym
        features = torch.tensor(np.stack([designFeatures.describe(plate, params, pockets, plateSym)[self.featureColumns]
                                          for pockets in designs]))
        with torch.no_grad():
            featureLogit = self.featureModel((features - self.featureMean) / self.featureStd)
            if self.blend > 0:
                toPixels = raster.plateFrame(plate)
                images = torch.tensor(np.stack([raster.rasterize(plate, pockets, toPixels) for pockets in designs]))
                cnnLogit = self.cnnModel(images)
            else:
                cnnLogit = torch.zeros_like(featureLogit)
        fm, fs, cm, cs = self.meta["logitStats"]
        return (1 - self.blend) * (featureLogit - fm) / fs + self.blend * (cnnLogit - cm) / cs

    def score(self, plate, params, designs, plateSym=None):
        mean, std = self.calibration
        return torch.sigmoid((self.logits(plate, params, designs, plateSym) - mean) / std).numpy()


def train(outPath, platesPath, seed=0, perKind=4, cnnEpochs=25, parallel=1, claudeGrades=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    records = {record["id"]: record for record in loadRecords(platesPath)}
    gold = sorted(plateId for plateId in goldIds() if plateId in records)
    started = time.time()
    designFeatures.styleReference.update(designFeatures.goldStyleReference(
        [(plateFromRecord(records[plateId]), [geom2d.toPolygon(p) for p in records[plateId]["pockets"]])
         for plateId in gold if not isHoldout(plateId)]))
    print(f"gold style reference: {designFeatures.styleReference}", flush=True)
    built = runPlateJobs({
        "sheets": (sheetSetJob, sheetSetItems(seed, records, claudeGrades=claudeGrades)),
        "goldTrain": (goldSetJob, [(records[plateId], seed, perKind) for plateId in gold if not isHoldout(plateId)]),
        "goldTest": (goldSetJob, [(records[plateId], seed, perKind) for plateId in gold if isHoldout(plateId)]),
        "prior": (priorSetJob, priorSetItems(seed, platesPath)),
    }, parallel)
    sheetTrain = [plateSet for key, plateSet in built["sheets"] if not isHeldOutSheet(key)]
    sheetHeld = [plateSet for key, plateSet in built["sheets"] if isHeldOutSheet(key)]
    trainSets = built["goldTrain"] + uiRatingSets(seed) + sheetTrain + [s for s in built["prior"] if s is not None]
    testSets = built["goldTest"]
    goldTestCount = len(testSets)
    testSets += sheetHeld
    print(f"built {sum(len(s.kinds) for s in trainSets)} train / {sum(len(s.kinds) for s in testSets)} test designs "
          f"on {len(trainSets)}/{len(testSets)} plates in {time.time() - started:.0f}s ({parallel} processes)", flush=True)

    allFeatures = np.concatenate([plateSet.features for plateSet in trainSets])
    featureMean, featureStd = allFeatures.mean(axis=0), allFeatures.std(axis=0) + 1e-6

    def featureInputs(sets):
        return [torch.tensor((plateSet.features - featureMean) / featureStd, dtype=torch.float32) for plateSet in sets]

    def rasterInputs(sets):
        return [torch.tensor(plateSet.rasters, dtype=torch.float32) for plateSet in sets]

    trainPairs, testPairs = pairsOf(trainSets), pairsOf(testSets)
    fitStarted = time.time()
    featureModel = trainModel(FeatureModel(len(designFeatures.featureNames)), featureInputs(trainSets), trainPairs, 300, 3e-3, rng)
    featureSeconds = time.time() - fitStarted
    cnnModel = trainModel(CnnModel(inputPool=cnnInputPool), rasterInputs(trainSets), trainPairs, cnnEpochs, 1e-3, rng,
                          augment=True)
    cnnSeconds = time.time() - fitStarted - featureSeconds
    print(f"fit {len(trainPairs)} pairs: feature model {featureSeconds:.0f}s, CNN {cnnSeconds:.0f}s", flush=True)

    featureTrainAcc, _ = pairAccuracy(featureModel, featureInputs(trainSets), trainPairs)
    featureTestAcc, featureTestScores = pairAccuracy(featureModel, featureInputs(testSets), testPairs)
    cnnTrainAcc, _ = pairAccuracy(cnnModel, rasterInputs(trainSets), trainPairs)
    cnnTestAcc, cnnTestScores = pairAccuracy(cnnModel, rasterInputs(testSets), testPairs)

    with torch.no_grad():
        featureLogits = torch.cat([featureModel(x) for x in featureInputs(trainSets)])
        cnnLogits = torch.cat([cnnModel(x) for x in rasterInputs(trainSets)])
    logitStats = [float(featureLogits.mean()), float(featureLogits.std() + 1e-6), float(cnnLogits.mean()), float(cnnLogits.std() + 1e-6)]

    # Blend by holdout ranking skill above chance (0.5).
    featureSkill, cnnSkill = max(0.0, featureTestAcc - 0.5), max(0.0, cnnTestAcc - 0.5)
    blend = cnnSkill / (featureSkill + cnnSkill) if featureSkill + cnnSkill > 0 else 0.0
    fm, fs, cm, cs = logitStats
    blendedTest = [(1 - blend) * (f - fm) / fs + blend * (c - cm) / cs for f, c in zip(featureTestScores, cnnTestScores)]
    blendTestAcc = float(np.mean([blendedTest[p][i] > blendedTest[p][j] for p, i, j, _ in testPairs]))
    blendedTrain = (1 - blend) * (featureLogits - fm) / fs + blend * (cnnLogits - cm) / cs
    calibration = (float(blendedTrain.mean()), float(blendedTrain.std() + 1e-6))

    perKindAcc, goldCorrect, sheetCorrect = {}, [], []
    for p, i, j, _ in testPairs:
        kind = testSets[p].kinds[j]
        correct = blendedTest[p][i] > blendedTest[p][j]
        perKindAcc.setdefault(kind, []).append(correct)
        (goldCorrect if p < goldTestCount else sheetCorrect).append(correct)
    report = {
        "trainPlates": [s.plateId for s in trainSets],
        "testPlates": [s.plateId for s in testSets],
        "pairs": {"train": len(trainPairs), "test": len(testPairs)},
        "featureModel": {"trainAcc": featureTrainAcc, "holdoutAcc": featureTestAcc},
        "cnnModel": {"trainAcc": cnnTrainAcc, "holdoutAcc": cnnTestAcc},
        "blendOnCnn": blend,
        "blendedHoldoutAcc": blendTestAcc,
        "goldHoldoutAcc": float(np.mean(goldCorrect)) if goldCorrect else None,
        "gradedHoldoutAcc": float(np.mean(sheetCorrect)) if sheetCorrect else None,
        "gradedPlates": {"train": len(sheetTrain), "heldOut": len(sheetHeld)},
        "holdoutAccByNegativeKind": {kind: float(np.mean(values)) for kind, values in sorted(perKindAcc.items())},
        "seconds": time.time() - started,
    }
    outPath = Path(outPath)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "featureModel": featureModel.state_dict(), "cnnModel": cnnModel.state_dict(),
        "featureMean": featureMean.astype(np.float32), "featureStd": featureStd.astype(np.float32),
        "blend": blend, "calibration": calibration,
        "meta": {"logitStats": logitStats, "featureNames": designFeatures.featureNames, "report": report,
                 "cnnInputPool": cnnInputPool,
                 "styleReference": dict(designFeatures.styleReference)},
    }, outPath)
    outPath.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="models/aestheticScorer.pt")
    parser.add_argument("--plates", default="data/processed/plates.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=defaultParallel(),
                        help="processes for building the training sets (one plate per job)")
    args = parser.parse_args()
    print(json.dumps(train(args.out, args.plates, args.seed, cnnEpochs=args.cnn_epochs, parallel=args.parallel), indent=1))


if __name__ == "__main__":
    main()
