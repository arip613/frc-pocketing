"""Rating sheets: candidate designs drawn blind for a person (or Claude) to grade by eye.

Each sheet is one plate: up to 8 valid candidates (ML at every pocket density, sampled
and densified variants, the truss baseline), shuffled and numbered with no source shown,
drawn in the colours of the user's example screenshots (red plate on dark grey). Gold
plates also get the real design as panel R for reference. Grades are 1-5:

  5  belongs with the example plates: clean hole-to-hole web, even pocket sizes, ribs
     meeting each other and the wall at open angles, symmetric where the plate is
  4  good, one small flaw
  3  acceptable but generic, or several small flaws
  2  clearly off: starbursts, slivers, crumbs, one huge pocket, much too sparse or busy
  1  ugly

Grades train the aesthetic scorer (higher-graded beats lower-graded on the same plate)
and the generator (it imitates designs graded 4 and 5).

    python -I -m frcpocket.ratingSheets make --round r1
    # grade data/curation/sheets/r1/*.png into data/curation/sheets/r1/grades.json:
    #   {"<sheet>": {"1": 4, "2": 2, ...}, ...}
    python -I -m frcpocket.ratingSheets ingest --round r1 --rater claude
"""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PatchPolygon
from shapely.geometry import Polygon
from shapely.ops import unary_union

from frcpocket import geom2d, polish, validator
from frcpocket.baseline import trussBaseline
from frcpocket.evaluate import styleParamsFromRecord
from frcpocket.generator import RibGenerator
from frcpocket.plate import Plate, PocketParams, loadPlateFromStep, plateFromRecord
from frcpocket.pocketer import ribStyles
from frcpocket.scorer import AestheticScorer
from frcpocket.splits import goldIds, isHoldout, loadRecords

sheetRoot = Path("data/curation/sheets")
ratingsPath = Path("data/curation/sheetRatings.jsonl")
densityFactors = (0.6, 1.0)  # Open (1.5) graded no better than these two in rounds 1-3
proposalsPerStyle = 2
candidatesPerSheet = 9
duplicateIou = 0.95  # pocket sets this similar count as the same design
backgroundColour = "#333333"
plateColour = "#8a1f12"
edgeColour = "#1c0b08"
testPlateNames = ("bearingPlate", "rectangleBoltPattern", "gusset", "irregularArm")


def plateKey(plate):
    """Stable id for a plate's geometry (outline + features)."""
    return hashlib.sha1(json.dumps([[list(map(lambda v: round(v, 5), point)) for point in plate.outline.exterior.coords],
                                    [[list(map(lambda v: round(v, 5), point)) for point in f.exterior.coords]
                                     for f in plate.features]]).encode()).hexdigest()[:12]


def sheetPlates(rng, extraCount=10, platesPath="data/processed/plates.jsonl", extraIds=()):
    """(name, plate, params, realPockets or None) for every sheet of a round.

    The user's own plates (every STEP under data/userPlates), every gold training plate
    (never held-out ones, so the scorer's holdout stays clean), the synthetic test plates,
    and a few other real plate outlines.
    """
    from frcpocket.trainGenerator import loadPlates, sampleParams

    plates = [(f"userPlate_{path.stem}", loadPlateFromStep(path), PocketParams(), None)
              for path in sorted(Path("data/userPlates").glob("*/*.step"))]
    records = {record["id"]: record for record in loadRecords(platesPath)}
    for plateId in sorted(goldIds()):
        if plateId in records and not isHoldout(plateId):
            record = records[plateId]
            plates.append((plateId, plateFromRecord(record), styleParamsFromRecord(record),
                           [geom2d.toPolygon(pocket) for pocket in record["pockets"]]))
    for name in testPlateNames:
        plates.append((f"test_{name}", loadPlateFromStep(f"data/testPlates/{name}.step"), PocketParams(), None))
    _, inputs, _ = loadPlates(platesPath)
    picked = [inputs[int(index)] for index in rng.choice(len(inputs), size=min(extraCount, len(inputs)), replace=False)]
    picked += [record for record in inputs if record["id"] in set(extraIds) and record not in picked]
    for record in picked:
        plate = plateFromRecord(record)
        plates.append((record["id"], plate, sampleParams(plate, rng), None))
    return plates


def candidatePool(plate, params, generator, seed):
    """Every valid design on offer: ML proposals for each rib style and density (polished,
    like the UI's), plus the truss baseline."""
    pool = []
    for angle, cap in ribStyles:
        for factor in densityFactors:
            styleParams = replace(params, pocketSizeFactor=factor, minRibAngleDegrees=angle, maxHoleRibs=cap)
            for index, layout in enumerate(generator.propose(plate, styleParams, count=proposalsPerStyle, seed=seed)):
                if polish.enabled and layout.pockets:
                    layout = polish.polishLayout(plate, styleParams, layout).layout  # same pass as the UI
                pool.append((f"ml a{angle:g}c{cap} d{factor} p{index}", styleParams, layout))
    pool.append(("baseline", params, trussBaseline(plate, params)))
    return [(source, candidateParams, layout) for source, candidateParams, layout in pool
            if layout.pockets and validator.validateLayout(plate, candidateParams, layout.pockets).passed]


def pickCandidates(pool, rng, count=candidatesPerSheet):
    """Distinct designs: each style's most likely design at the middle and dense settings,
    and the baseline, first; then random others. Shown in random order."""
    def priority(entry):
        source = entry[0]
        if source == "baseline":
            return 1
        if source.endswith("p0"):
            return 0 if " d1.0 " in source else 1 if " d0.6 " in source else 2
        return 3
    ordered = sorted(pool, key=lambda entry: (priority(entry), float(rng.random())))
    chosen, unions = [], []
    for entry in ordered:
        union = unary_union(entry[2].pockets)
        if any(geom2d.iou(union, other) >= duplicateIou for other in unions):
            continue
        chosen.append(entry)
        unions.append(union)
        if len(chosen) == count:
            break
    return [chosen[int(k)] for k in rng.permutation(len(chosen))]


def drawPanel(axis, plate, pockets, label):
    axis.set_facecolor(backgroundColour)
    axis.add_patch(PatchPolygon(list(plate.outline.exterior.coords), closed=True, facecolor=plateColour,
                                edgecolor=edgeColour, linewidth=0.8))
    for shape in list(plate.features) + list(pockets):
        axis.add_patch(PatchPolygon(list(shape.exterior.coords), closed=True, facecolor=backgroundColour,
                                    edgecolor=edgeColour, linewidth=0.6))
    minX, minY, maxX, maxY = plate.outline.bounds
    pad = 0.04 * max(maxX - minX, maxY - minY)
    axis.set_xlim(minX - pad, maxX + pad)
    axis.set_ylim(minY - pad, maxY + pad)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(0.02, 0.97, label, transform=axis.transAxes, color="white", fontsize=15, fontweight="bold", va="top")


def drawSheet(path, plate, panels):
    minX, minY, maxX, maxY = plate.outline.bounds
    aspect = (maxX - minX) / max(maxY - minY, 1e-9)
    columns = 2 if aspect > 1.8 else 3
    rows = math.ceil(len(panels) / columns)
    panelWidth = 6.0 if columns == 2 else 4.6
    panelHeight = min(panelWidth / aspect, 6.0) + 0.2
    figure, axes = plt.subplots(rows, columns, figsize=(columns * panelWidth, rows * panelHeight), dpi=70,
                                facecolor=backgroundColour, squeeze=False)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, (label, pockets) in zip(axes.flat, panels):
        axis.set_visible(True)
        drawPanel(axis, plate, pockets, label)
    figure.tight_layout(pad=0.4)
    figure.savefig(path, facecolor=backgroundColour)
    plt.close(figure)


workerGenerator = None


def initWorker(generatorPath, scorerPath):
    global workerGenerator
    workerGenerator = RibGenerator(generatorPath)
    if Path(scorerPath).exists():
        AestheticScorer(scorerPath)  # sets the fitted gold pocket size, as in the UI


def sheetJob(name, plate, params, real, seed, outDir):
    """Build, draw, and describe one plate's sheet (runs in a worker process)."""
    rng = np.random.default_rng(seed)
    try:
        pool = candidatePool(plate, params, workerGenerator, seed=int(rng.integers(1 << 30)))
    except Exception as error:  # a plate the generator can't handle shouldn't stop the round
        return name, [], f"skip: {error}"
    chosen = pickCandidates(pool, rng)
    if len(chosen) < 2:
        return name, [], f"skip: only {len(chosen)} distinct valid designs"
    panels = [("R", real)] if real else []
    lines = []
    for number, (source, candidateParams, layout) in enumerate(chosen, start=1):
        panels.append((str(number), layout.pockets))
        lines.append({
            "sheet": name, "candidate": number, "source": source, "plateKey": plateKey(plate),
            "plateId": name, "thicknessIn": plate.thicknessIn, "params": vars(candidateParams),
            "outline": list(plate.outline.exterior.coords),
            "holes": [list(feature.exterior.coords) for feature in plate.features],
            "pockets": [list(pocket.exterior.coords) for pocket in layout.pockets],
            "ribs": [[list(layout.graph.nodes[i].point), list(layout.graph.nodes[j].point)] for i, j in layout.ribs],
            "ribWidths": [layout.ribWidth(edge, candidateParams) / candidateParams.ribIn for edge in layout.ribs],
        })
    drawSheet(Path(outDir) / f"{name}.png", plate, panels)
    return name, lines, f"{len(chosen)} candidates from {len(pool)} valid"


def make(roundName, generatorPath, seed, extraCount, workers=3, scorerPath="models/aestheticScorer.pt", only=None):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rng = np.random.default_rng(seed)
    outDir = sheetRoot / roundName
    outDir.mkdir(parents=True, exist_ok=True)
    candidatesPath = outDir / "candidates.jsonl"
    candidatesPath.write_text("")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initWorker, initargs=(generatorPath, scorerPath)) as pool:
        plates = [entry for entry in sheetPlates(rng, extraCount, extraIds=only or ()) if not only or entry[0] in only]
        seeds = [int(rng.integers(1 << 30)) for _ in plates]
        # Biggest plates first, so the slowest sheets don't start last and hold up the round.
        order = sorted(range(len(plates)), key=lambda k: -plates[k][1].materialArea * (1 + len(plates[k][1].features)))
        jobs = [pool.submit(sheetJob, *plates[k], seeds[k], str(outDir)) for k in order]
        for job in as_completed(jobs):
            name, lines, message = job.result()
            with open(candidatesPath, "a") as handle:  # written as each sheet lands
                for line in lines:
                    handle.write(json.dumps(line) + "\n")
            print(f"{name}: {message}", flush=True)


def ingest(roundName, rater):
    outDir = sheetRoot / roundName
    grades = json.loads((outDir / "grades.json").read_text())
    seen = set()
    if ratingsPath.exists():  # ingesting a round again only adds sheets graded since
        for line in ratingsPath.read_text().splitlines():
            entry = json.loads(line)
            seen.add((entry.get("round"), entry.get("sheet"), entry.get("candidate"), entry.get("rater")))
    added = 0
    ratingsPath.parent.mkdir(parents=True, exist_ok=True)
    with open(ratingsPath, "a") as handle:
        for line in (outDir / "candidates.jsonl").read_text().splitlines():
            entry = json.loads(line)
            grade = grades.get(entry["sheet"], {}).get(str(entry["candidate"]))
            if grade is None or (roundName, entry["sheet"], entry["candidate"], rater) in seen:
                continue
            entry.update({"kind": "sheetCandidate", "round": roundName, "rater": rater, "grade": float(grade)})
            handle.write(json.dumps(entry) + "\n")
            added += 1
    print(f"added {added} graded designs to {ratingsPath}")


def loadSheetRatings(path=ratingsPath):
    """plateKey -> (plate, [(params, pockets, ribs, grade, entry)]) from every graded round."""
    byPlate = {}
    if not Path(path).exists():
        return byPlate
    for line in Path(path).read_text().splitlines():
        entry = json.loads(line)
        if entry.get("kind") != "sheetCandidate":
            continue
        if entry["plateKey"] not in byPlate:
            plate = Plate(outline=Polygon(entry["outline"]), features=[Polygon(hole) for hole in entry["holes"]],
                          thicknessIn=entry["thicknessIn"])
            byPlate[entry["plateKey"]] = (plate, [])
        byPlate[entry["plateKey"]][1].append((PocketParams(**entry["params"]), [Polygon(p) for p in entry["pockets"]],
                                              entry["ribs"], entry["grade"], entry))
    return byPlate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["make", "ingest"])
    parser.add_argument("--round", required=True)
    parser.add_argument("--generator", default="models/generator.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", type=int, default=10, help="other real plate outlines per round")
    parser.add_argument("--rater", default="claude")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default="", help="comma-separated sheet names to limit a round to")
    args = parser.parse_args()
    if args.action == "make":
        make(args.round, args.generator, args.seed, args.extra, args.workers, only=set(filter(None, args.only.split(","))))
    else:
        ingest(args.round, args.rater)


if __name__ == "__main__":
    main()
