"""Train the rib generator: imitate gold plates, then generate -> validate -> score -> retrain.

Round 0 imitates the rib-graph projections of the gold training plates. Each later round:
  1. proposes layouts (most-likely, densified, sampled) for a batch of training plates,
     plus a full triangulation on the same candidate graph as a stiff reference option
  2. rejects anything the validator fails
  3. scores the rest: utility = 0.70 aesthetic + 0.15 structure + 0.15 mass removed, where
     structure mixes stiffness kept, stiffness per mass kept, and peak-stress increase;
     designs below a soundness floor (gutted plates) are penalized
     (FEA runs only on each plate's most promising candidates, to save time)
  4. retrains on the best two layouts per plate plus the imitation data
Validation plates (never trained on) are scored against the truss baseline every round,
and training stops when the validation utility plateaus.

    python -I -m frcpocket.trainGenerator --rounds 8
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path

import numpy as np
import shapely
import torch
from torch import nn

from frcpocket import fea, geom2d, testPlates, validator
from frcpocket.baseline import trussBaseline
from frcpocket.evaluate import styleParamsFromRecord
from frcpocket.generator import RibGenerator, RibGnn, encodeGraph, mirrorMap
from frcpocket.layout import candidateGraph, defaultGraph, makeLayout, removedFraction
from frcpocket.plate import PocketParams, loadPlateFromStep, plateFromRecord
from frcpocket.projection import projectRealDesign
from frcpocket.scorer import AestheticScorer
from frcpocket.splits import drawingGoldWeight, goldIds, isHoldout, loadRecords, userTasteIds, userTasteWeight

torch.set_num_threads(4)
# Looks come first (the user's priority), structure and mass removal second. The floor
# below still rules out structurally unsound designs such as a gutted plate.
aestheticWeight, structureWeight, massWeight = 0.70, 0.15, 0.15
soundnessMinStiffnessKept = 0.08
soundnessMaxStressIncrease = 10.0
unsoundPenalty = 0.3
maxPocketShare = 0.3  # fast-mode gutting guard: no single pocket above this share of the plate
targetRemoved = 0.4
specificStiffnessCap = 1.0
referenceStiffnessKept = 0.5
referenceStressIncrease = 2.5
feaPerPlate = 3
coarseMeshFactor = 2.0
feaTimeoutSeconds = 120
feaFailurePenalty = 0.05
imitationWeight = 5.0  # the gold plates are the look to match; weigh imitating them heavily
gradedImitationWeights = ((4.5, 4.0), (4.0, 2.0))  # (min grade, weight) for designs graded on the rating sheets


def structuralTerm(stiffnessKept, specificStiffness, stressIncrease):
    """Stiffness kept (full marks at half the solid plate's), stiffness per mass kept, and
    peak-stress increase (full marks at or below x2.5)."""
    kept = min(max(stiffnessKept, 0.0) / referenceStiffnessKept, 1.0)
    specific = min(max(specificStiffness, 0.0) / specificStiffnessCap, 1.0)
    stress = min(1.0, referenceStressIncrease / max(stressIncrease, 1e-9))
    return 0.4 * kept + 0.3 * specific + 0.3 * stress


def isSound(stiffnessKept, stressIncrease):
    return stiffnessKept >= soundnessMinStiffnessKept and stressIncrease <= soundnessMaxStressIncrease


def utilityOf(aesthetic, stiffnessKept, specificStiffness, removed, stressIncrease):
    utility = (aestheticWeight * aesthetic
               + structureWeight * structuralTerm(stiffnessKept, specificStiffness, stressIncrease)
               + massWeight * min(removed / targetRemoved, 1.0))
    return utility if isSound(stiffnessKept, stressIncrease) else utility - unsoundPenalty


def isValidationPlate(plateId):
    return int(hashlib.sha1(("val" + plateId.split("_")[0]).encode()).hexdigest(), 16) % 100 < 15


def sampleParams(plate, rng):
    """FRC-typical parameters, kept sensible for the plate's size."""
    scale = math.sqrt(plate.outline.area)
    walls = [0.125, 0.1875, 0.25] if scale > 3 else [0.1, 0.125]
    ribs = [0.125, 0.1875, 0.25] if scale > 3 else [0.1, 0.125]
    corners = [0.0625, 0.125] if scale > 3 else [0.0625]
    return PocketParams(wallIn=float(rng.choice(walls)), ribIn=float(rng.choice(ribs)), cornerRadiusIn=float(rng.choice(corners)))


def withRibStyle(params, rng):
    """The same parameters in one of the UI's rib styles, picked at random."""
    from frcpocket.pocketer import ribStyles

    angle, cap = ribStyles[int(rng.integers(len(ribStyles)))]
    return replace(params, minRibAngleDegrees=angle, maxHoleRibs=cap)


def loadPlates(platesPath):
    records = loadRecords(platesPath)
    gold = goldIds()
    goldTrain, inputs, validation = [], [], []
    for record in records:
        plate = plateFromRecord(record)
        if not plate.outline.is_valid or plate.outline.area < 4 or plate.outline.area > 100:
            continue
        if not (2 <= len(plate.features) <= 60):
            continue
        if record["id"] in gold:
            (validation if isHoldout(record["id"]) else goldTrain).append(record)
        elif isHoldout(record["id"]) or isValidationPlate(record["id"]):
            validation.append(record)
        else:
            inputs.append(record)
    return goldTrain, inputs, validation


class Example:
    """One training target: a plate's graph tensors plus rib labels."""

    def __init__(self, tensors, keepLabels, widthLabels, weight):
        self.tensors, self.keepLabels, self.widthLabels, self.weight = tensors, keepLabels, widthLabels, weight


def imitationExamples(goldTrain):
    examples = []
    for record in goldTrain:
        plate = plateFromRecord(record)
        params = styleParamsFromRecord(record)
        graph = defaultGraph(plate, params)
        projection = projectRealDesign(plate, params, [geom2d.toPolygon(pocket) for pocket in record["pockets"]], graph)
        chosen = set(projection.layout.ribs)
        keep = torch.tensor([float(edge in chosen) for edge in graph.edges])
        widths = torch.tensor(projection.widthClass, dtype=torch.long)
        weight = imitationWeight * (userTasteWeight if record["id"] in userTasteIds else 1.0)
        if record.get("origin") == "drawing":
            weight *= drawingGoldWeight
        examples.append(Example(encodeGraph(plate, params, graph), keep, widths, weight=weight))
    return examples


def gradedImitationExamples(parallel=None):
    """Designs graded 4 or 5 on the rating sheets, as imitation targets.

    Ribs are matched to this plate's candidate graph by their end points. Designs that
    came from another graph (e.g. the truss baseline, or polish-added ribs) are projected
    onto it instead, the same way real designs are, which takes ~20 s each, so the work is
    spread over processes. Width labels of -1 mean "unknown" and are not trained on."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from frcpocket import ratingSheets

    jobs = []
    for plate, rated in ratingSheets.loadSheetRatings().values():
        for params, pockets, ribs, grade, entry in rated:
            weight = next((weight for minGrade, weight in gradedImitationWeights if grade >= minGrade), None)
            if weight is not None:
                jobs.append((plate, params, pockets, ribs, entry.get("ribWidths"), weight))
    workers = parallel or max(1, min(6, (os.cpu_count() or 2)))
    if workers <= 1 or len(jobs) < 8:
        return [gradedExample(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(gradedExample, *zip(*jobs), chunksize=4))


def gradedExample(plate, params, pockets, ribs, ribWidths, weight):
    graph = defaultGraph(plate, params)
    nodeIndex = {tuple(np.round(node.point, 6)): k for k, node in enumerate(graph.nodes)}
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    keep = torch.zeros(len(graph.edges))
    widths = torch.full((len(graph.edges),), -1, dtype=torch.long)
    ribWidths = ribWidths or [None] * len(ribs)
    matched = True
    for (a, b), width in zip(ribs, ribWidths):
        i, j = nodeIndex.get(tuple(np.round(a, 6))), nodeIndex.get(tuple(np.round(b, 6)))
        edge = tuple(sorted((i, j))) if i is not None and j is not None else None
        if edge not in lookup:
            matched = False
            break
        keep[lookup[edge]] = 1.0
        if width is not None:
            widths[lookup[edge]] = int(np.argmin([abs(width - factor) for factor in (1.0, 1.6, 2.4)]))
    if not matched:
        projection = projectRealDesign(plate, params, pockets, graph)
        chosen = set(projection.layout.ribs)
        keep = torch.tensor([float(edge in chosen) for edge in graph.edges])
        widths = torch.tensor(projection.widthClass, dtype=torch.long)
    return Example(encodeGraph(plate, params, graph), keep, widths, weight=weight)


def fit(model, examples, epochs, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for index in np.random.permutation(len(examples)):
            example = examples[index]
            keepLogits, widthLogits = model(example.tensors)
            positive = example.keepLabels.mean().clamp(0.02, 0.98)
            loss = nn.functional.binary_cross_entropy_with_logits(
                keepLogits, example.keepLabels, pos_weight=(1 - positive) / positive)
            kept = (example.keepLabels > 0.5) & (example.widthLabels >= 0)
            if kept.any():
                loss = loss + 0.3 * nn.functional.cross_entropy(widthLogits[kept], example.widthLabels[kept])
            optimizer.zero_grad()
            (loss * example.weight).backward()
            optimizer.step()
    model.eval()
    return model


class Judge:
    """Validator + aesthetic + (coarse) FEA. FEA runs in worker processes, so a crash in
    the native mesher costs one candidate instead of the whole run."""

    def __init__(self, scorerPath, workers=4, useFea=True):
        self.scorer = AestheticScorer(scorerPath)
        self.workers = workers
        self.useFea = useFea
        self.pool = None

    def feaPool(self):
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
        return self.pool

    def structural(self, plate, params, pocketSets, meshSize):
        plateKey = hashlib.sha1(shapely.to_wkb(plate.outline) + bytes(len(plate.features))).hexdigest()
        outline = shapely.to_wkb(plate.outline)
        features = [shapely.to_wkb(feature) for feature in plate.features]
        paramsDict = dict(vars(params))
        futures = []
        for pockets in pocketSets:
            try:
                futures.append(self.feaPool().submit(
                    fea.analyzeJob, outline, features, plate.thicknessIn,
                    [shapely.to_wkb(pocket) for pocket in pockets], paramsDict, meshSize, plateKey))
            except BrokenProcessPool:
                self.pool = None
                futures.append(None)
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=feaTimeoutSeconds) if future is not None else None)
            except BrokenProcessPool:
                self.pool = None
                results.append(None)
            except Exception:
                results.append(None)
        return results

    def rank(self, plate, params, layouts, feaCount=feaPerPlate):
        results = []
        valid = [layout for layout in layouts if layout.pockets and validator.validateLayout(plate, params, layout.pockets).passed]
        if not valid:
            return results
        aesthetic = self.scorer.score(plate, params, [layout.pockets for layout in valid])
        removed = [removedFraction(plate, layout.pockets) for layout in valid]
        prelim = [aestheticWeight * a + massWeight * min(m / targetRemoved, 1) for a, m in zip(aesthetic, removed)]
        if not self.useFea:
            # Fast mode for training: looks + mass, with a geometric guard against gutting
            # (one pocket taking over the plate). Full FEA still runs in evaluation and the UI.
            for index in np.argsort(prelim)[::-1]:
                largest = max(pocket.area for pocket in valid[index].pockets) / plate.materialArea
                penalty = unsoundPenalty if largest > maxPocketShare else 0.0
                results.append({
                    "layout": valid[index], "aesthetic": float(aesthetic[index]), "removed": removed[index],
                    "stiffnessKept": None, "stressIncrease": None, "specificStiffness": None,
                    "utility": (prelim[index] - penalty) / (aestheticWeight + massWeight),
                })
            return sorted(results, key=lambda result: -result["utility"])
        order = list(np.argsort(prelim)[::-1][:feaCount])
        meshSize = fea.defaultMeshSize(plate, params) * coarseMeshFactor
        structures = self.structural(plate, params, [valid[index].pockets for index in order], meshSize)
        for index, structure in zip(order, structures):
            if structure is None:
                # FEA failed (rare mesher trouble): keep the candidate, ranked on looks and
                # mass only and slightly penalized, with structure reported as unknown.
                results.append({
                    "layout": valid[index], "aesthetic": float(aesthetic[index]), "removed": removed[index],
                    "stiffnessKept": None, "stressIncrease": None, "specificStiffness": None,
                    "utility": prelim[index] - feaFailurePenalty,
                })
                continue
            specific, stiffness, stress = structure["specificStiffness"], structure["stiffnessKept"], structure["stressIncrease"]
            results.append({
                "layout": valid[index], "aesthetic": float(aesthetic[index]), "removed": removed[index],
                "stiffnessKept": stiffness, "stressIncrease": stress, "specificStiffness": specific,
                "utility": utilityOf(float(aesthetic[index]), stiffness, specific, removed[index], stress),
            })
        return sorted(results, key=lambda result: -result["utility"])


def summarizeResults(results):
    if not results:
        return {}
    keys = ("aesthetic", "removed", "stiffnessKept", "stressIncrease", "specificStiffness", "utility")
    summary = {}
    for key in keys:
        values = [result[key] for result in results if result[key] is not None]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def validationPlates(validation, testPlateDir, rng, limit=None):
    """Held-out gold plates and test plates always; other validation plates up to `limit`."""
    gold = goldIds()
    others = [record for record in validation if record["id"] not in gold]
    if limit is not None:
        others = [others[k] for k in sorted(rng.choice(len(others), size=min(limit, len(others)), replace=False))]
    plates = []
    for record in [record for record in validation if record["id"] in gold] + others:
        plate = plateFromRecord(record)
        params = styleParamsFromRecord(record) if record["id"] in goldIds() else sampleParams(plate, rng)
        plates.append((record["id"], plate, params))
    for name, path in testPlates.writeTestPlates(testPlateDir).items():
        plates.append((f"test:{name}", loadPlateFromStep(path), PocketParams()))
    return plates


_workerJudge = None


def initWorker(scorerPath, useFea):
    """Each round worker keeps its own scorer (and FEA pool, if used) for the whole run."""
    global _workerJudge
    torch.set_num_threads(1)
    _workerJudge = Judge(scorerPath, workers=1, useFea=useFea)


def litePlate(plate):
    """The plate without its CAD solid, which can't be sent to another process."""
    from frcpocket.plate import Plate

    return Plate(outline=plate.outline, features=plate.features, thicknessIn=plate.thicknessIn, name=plate.name)


def numbersOnly(result):
    return {key: value for key, value in result.items() if key != "layout"}


def trainPlateJob(stateDict, record, params, seed, proposals):
    """Propose, validate, and score layouts for one training plate; return elite targets."""
    model = RibGnn()
    model.load_state_dict(stateDict)
    generator = RibGenerator(model=model)
    plate = plateFromRecord(record)
    try:
        prepared = generator.prepare(plate, params)
        layouts = generator.propose(plate, params, count=proposals, seed=seed, prepared=prepared)
    except Exception:
        return None
    # A full triangulation on the same graph joins the pool as a stiff reference
    # option, so selection can pull the model toward density where structure needs it.
    graph, tensors, _ = prepared
    fullTruss = makeLayout(plate, params, graph, [edge for edge, planar in zip(graph.edges, graph.isDelaunay) if planar])
    ranked = _workerJudge.rank(plate, params, layouts + [fullTruss], feaCount=feaPerPlate + 1)
    validCount = sum(1 for layout in layouts if layout.pockets and validator.validateLayout(plate, params, layout.pockets).passed)
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    targets = []
    for result in ranked[:2]:
        layout = result["layout"]
        keep = torch.zeros(len(graph.edges))
        widths = torch.zeros(len(graph.edges), dtype=torch.long)
        for edge in layout.ribs:
            keep[lookup[edge]] = 1.0
            width = layout.ribWidth(edge, params) / params.ribIn
            widths[lookup[edge]] = int(np.argmin([abs(width - factor) for factor in (1.0, 1.6, 2.4)]))
        targets.append((keep, widths, 1.0 + result["utility"]))
    return tensors, targets, len(layouts), validCount


def evalPlateJob(stateDict, plate, params, proposals):
    """Greedy, best-of-K, and baseline results for one validation plate (numbers only)."""
    model = RibGnn()
    model.load_state_dict(stateDict)
    generator = RibGenerator(model=model)
    try:
        layouts = generator.propose(plate, params, count=proposals, seed=1)
    except Exception:
        return None
    greedy = _workerJudge.rank(plate, params, layouts[:1], feaCount=1)
    best = _workerJudge.rank(plate, params, layouts)
    baseline = _workerJudge.rank(plate, params, [trussBaseline(plate, params)], feaCount=1)
    return ([numbersOnly(greedy[0])] if greedy else [], [numbersOnly(best[0])] if best else [],
            [numbersOnly(baseline[0])] if baseline else [])


def summarizeEval(outcomes):
    outcomes = [outcome for outcome in outcomes if outcome is not None]
    ml = [row for greedy, _, _ in outcomes for row in greedy]
    mlBest = [row for _, best, _ in outcomes for row in best]
    base = [row for _, _, baseline in outcomes for row in baseline]
    total = len(outcomes)
    return {
        "plates": total,
        "mlGreedyValidRate": len(ml) / max(total, 1),
        "mlGreedy": summarizeResults(ml),
        "mlBestOfK": summarizeResults(mlBest),
        "baseline": summarizeResults(base),
        "baselineValidRate": len(base) / max(total, 1),
    }


def evaluateRoundParallel(pool, model, plates, proposals):
    stateDict = model.state_dict()
    futures = [pool.submit(evalPlateJob, stateDict, litePlate(plate), params, proposals) for _, plate, params in plates]
    return summarizeEval([future.result() for future in futures])


def evaluateRound(generator, judge, plates, proposals):
    ml, mlBest, base, validCount, total = [], [], [], 0, 0
    for plateId, plate, params in plates:
        try:
            layouts = generator.propose(plate, params, count=proposals, seed=1)
        except Exception:
            continue
        total += 1
        greedyRank = judge.rank(plate, params, layouts[:1], feaCount=1)
        if greedyRank:
            validCount += 1
            ml.append(greedyRank[0])
        best = judge.rank(plate, params, layouts)
        if best:
            mlBest.append(best[0])
        baseline = judge.rank(plate, params, [trussBaseline(plate, params)], feaCount=1)
        if baseline:
            base.append(baseline[0])
    return {
        "plates": total,
        "mlGreedyValidRate": validCount / max(total, 1),
        "mlGreedy": summarizeResults(ml),
        "mlBestOfK": summarizeResults(mlBest),
        "baseline": summarizeResults(base),
        "baselineValidRate": len(base) / max(total, 1),
    }


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    goldTrain, inputs, validation = loadPlates(args.plates)
    print(f"plates: {len(goldTrain)} gold train, {len(inputs)} training inputs, {len(validation)} validation", flush=True)
    judge = Judge(args.scorer, workers=args.workers, useFea=args.fea)
    roundPool = None
    if args.parallel > 1:
        roundPool = ProcessPoolExecutor(max_workers=args.parallel, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=initWorker, initargs=(args.scorer, args.fea))
        print(f"spreading each round over {args.parallel} processes", flush=True)

    def evaluate(model):
        if roundPool is not None:
            return evaluateRoundParallel(roundPool, model, valPlates, args.proposals)
        return evaluateRound(RibGenerator(model=model), judge, valPlates, args.proposals)

    valPlates = validationPlates(validation, args.test_plates, np.random.default_rng(123), limit=args.val_limit)
    outDir = Path(args.models)
    outDir.mkdir(parents=True, exist_ok=True)

    model = RibGnn()
    imitation = imitationExamples(goldTrain)
    graded = gradedImitationExamples(parallel=args.parallel)
    print(f"imitation: {len(imitation)} gold, {len(graded)} graded 4-5 on rating sheets", flush=True)
    imitation += graded
    logPath = outDir / "trainingLog.json"
    if args.resume_round is not None:
        model.load_state_dict(torch.load(outDir / f"generator_round{args.resume_round}.pt", weights_only=True))
        log = [row for row in json.loads(Path(args.resume_log).read_text())] if args.resume_log else []
        startRound = args.resume_round + 1
        print(f"resuming after round {args.resume_round}", flush=True)
    else:
        fit(model, imitation, epochs=args.imitation_epochs)
        log = []
        startRound = 1
        started = time.time()
        metrics = evaluate(model)
        metrics.update({"round": 0, "seconds": time.time() - started})
        log.append(metrics)
        torch.save(model.state_dict(), outDir / "generator_round0.pt")
        print(json.dumps(metrics), flush=True)
    generator = RibGenerator(model=model)
    best = max(log, key=lambda row: row["mlBestOfK"].get("utility", -1))
    bestUtility, bestRound, stale = best["mlBestOfK"].get("utility", -1), best["round"], 0
    buffer = []

    for roundIndex in range(startRound, args.rounds + 1):
        roundStart = time.time()
        batch = list(goldTrain) + [inputs[k] for k in rng.choice(len(inputs), size=min(args.batch, len(inputs)), replace=False)]
        elites, attempted, validProposals = [], 0, 0
        if roundPool is not None:
            stateDict = model.state_dict()
            jobs = []
            for record in batch:
                params = styleParamsFromRecord(record) if record["id"] in goldIds() else sampleParams(plateFromRecord(record), rng)
                params = withRibStyle(params, rng)
                jobs.append(roundPool.submit(trainPlateJob, stateDict, record, params, int(rng.integers(1 << 30)), args.proposals))
            for job in jobs:
                outcome = job.result()
                if outcome is None:
                    continue
                tensors, targets, count, valid = outcome
                attempted += count
                validProposals += valid
                elites += [Example(tensors, keep, widths, weight=weight) for keep, widths, weight in targets]
            batch = []  # already handled in parallel
        for record in batch:
            plate = plateFromRecord(record)
            params = withRibStyle(styleParamsFromRecord(record) if record["id"] in goldIds() else sampleParams(plate, rng), rng)
            try:
                prepared = generator.prepare(plate, params)
                layouts = generator.propose(plate, params, count=args.proposals, seed=int(rng.integers(1 << 30)), prepared=prepared)
            except Exception:
                continue
            attempted += len(layouts)
            # A full triangulation on the same graph joins the pool as a stiff reference
            # option, so selection can pull the model toward density where structure needs it.
            graph = prepared[0]
            fullTruss = makeLayout(plate, params, graph, [edge for edge, planar in zip(graph.edges, graph.isDelaunay) if planar])
            ranked = judge.rank(plate, params, layouts + [fullTruss], feaCount=feaPerPlate + 1)
            validProposals += sum(1 for layout in layouts
                                  if layout.pockets and validator.validateLayout(plate, params, layout.pockets).passed)
            graph, tensors, _ = prepared
            lookup = {edge: k for k, edge in enumerate(graph.edges)}
            for result in ranked[:2]:
                layout = result["layout"]
                keep = torch.zeros(len(graph.edges))
                widths = torch.zeros(len(graph.edges), dtype=torch.long)
                for edge in layout.ribs:
                    keep[lookup[edge]] = 1.0
                    width = layout.ribWidth(edge, params) / params.ribIn
                    widths[lookup[edge]] = int(np.argmin([abs(width - factor) for factor in (1.0, 1.6, 2.4)]))
                elites.append(Example(tensors, keep, widths, weight=1.0 + result["utility"]))
        buffer = (buffer + elites)[-args.buffer:]
        fit(model, imitation + buffer, epochs=args.epochs)
        generator = RibGenerator(model=model)
        metrics = evaluate(model)
        metrics.update({
            "round": roundIndex,
            "trainProposalValidRate": validProposals / max(attempted, 1),
            "elites": len(elites),
            "seconds": time.time() - roundStart,
        })
        log.append(metrics)
        torch.save(model.state_dict(), outDir / f"generator_round{roundIndex}.pt")
        logPath.write_text(json.dumps({"bestRound": bestRound, "rounds": log}, indent=1))
        print(json.dumps(metrics), flush=True)
        utility = metrics["mlBestOfK"].get("utility", -1)
        if utility > bestUtility + args.plateau:
            bestUtility, bestRound, stale = utility, roundIndex, 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"plateau: no gain > {args.plateau} for {args.patience} rounds", flush=True)
                break
    if roundPool is not None:
        roundPool.shutdown()
    best = torch.load(outDir / f"generator_round{bestRound}.pt", weights_only=True)
    torch.save(best, outDir / "generator.pt")
    logPath.write_text(json.dumps({"bestRound": bestRound, "rounds": log}, indent=1))
    print(f"best round {bestRound}, utility {bestUtility:.3f}", flush=True)


def buildParser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plates", default="data/processed/plates.jsonl")
    parser.add_argument("--scorer", default="models/aestheticScorer.pt")
    parser.add_argument("--models", default="models")
    parser.add_argument("--test-plates", default="data/testPlates")
    # Imitation (round 0) or round 1 has won every training run so far, so by default the
    # loop runs one generate -> score -> retrain round after imitation and keeps the better.
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imitation-epochs", type=int, default=100)
    parser.add_argument("--buffer", type=int, default=400)
    parser.add_argument("--plateau", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2, help="FEA worker processes")
    parser.add_argument("--parallel", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)),
                        help="processes to spread each round's plates over (1 = sequential)")
    parser.add_argument("--fea", action="store_true", help="run FEA inside the training loop (slow)")
    parser.add_argument("--val-limit", type=int, default=12, help="non-gold validation plates per round")
    parser.add_argument("--resume-round", type=int, default=None, help="continue from models/generator_round<N>.pt")
    parser.add_argument("--resume-log", default=None, help="JSON list of earlier round metrics")
    return parser


def main():
    train(buildParser().parse_args())


if __name__ == "__main__":
    main()
