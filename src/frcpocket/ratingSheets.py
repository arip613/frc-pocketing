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
densityFactors = (1.0, 1.25)  # the user and the expert grade 1.0 above 0.6 (and the user 1.5 below 1.0); 1.25 is the live default
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
                    handle.write(json.dumps(compactEntry(line)) + "\n")
            print(f"{name}: {message}", flush=True)


def compactEntry(entry, digits=5):
    """Geometry rounded to 1e-5 in (far below any tolerance), so a round's candidates.jsonl
    stays well under GitHub's 100 MB file limit."""
    def rounded(value):
        if isinstance(value, float):
            return round(value, digits)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value
    return {key: rounded(value) if key in ("outline", "holes", "pockets") else value for key, value in entry.items()}


def compactRound(roundName):
    path = sheetRoot / roundName / "candidates.jsonl"
    lines = [json.dumps(compactEntry(json.loads(line))) for line in path.read_text().splitlines()]
    path.write_text("".join(line + "\n" for line in lines))


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
            # Just the reference and the grade: the geometry stays in the round's candidates.jsonl.
            handle.write(json.dumps({"kind": "sheetCandidate", "round": roundName, "sheet": entry["sheet"],
                                     "candidate": entry["candidate"], "rater": rater, "grade": float(grade)}) + "\n")
            added += 1
    print(f"added {added} graded designs to {ratingsPath}")


userGradesPath = Path("data/curation/userGrades.json")


def userGradedEntries():
    """The user's grades (from the Plate Curator page) as sheet-candidate entries.

    Stored compactly as {"<round>__<sheet>__<candidate>": grade}; geometry comes from
    each round's candidates.jsonl, which keeps sheetRatings.jsonl under GitHub's file limit.
    """
    if not userGradesPath.exists():
        return []
    grades = json.loads(userGradesPath.read_text())
    entries = []
    for roundName in sorted({key.split("__")[0] for key in grades}):
        for line in (sheetRoot / roundName / "candidates.jsonl").read_text().splitlines():
            entry = json.loads(line)
            grade = grades.get(f"{roundName}__{entry['sheet']}__{entry['candidate']}")
            if grade is not None:
                entry.update({"kind": "sheetCandidate", "round": roundName, "rater": "user", "grade": float(grade)})
                entries.append(entry)
    return entries


def pairStrengths(path=Path("data/curation/expertPairs.jsonl"), iterations=200):
    """Design -> standardised strength from the experts' A-vs-B picks (Plate Pairs page).

    Bradley-Terry per plate, fitted with the MM updates; "about the same" counts half a
    win each way, and every design gets one virtual tie with a plate-average design so a
    design that won (or lost) everything stays finite. Strengths are log-scale and
    standardised within each plate, like a rater's 1-9 ratings."""
    if not Path(path).exists():
        return {}
    byPlate = {}
    for line in Path(path).read_text().splitlines():
        pick = json.loads(line)
        byPlate.setdefault(pick["a"].rsplit(":", 1)[0], []).append(pick)
    out = {}
    for picks in byPlate.values():
        designs = sorted({pick[side] for pick in picks for side in ("a", "b")})
        if len(designs) < 2:
            continue
        index = {design: k for k, design in enumerate(designs)}
        wins = np.full(len(designs), 0.5)  # the virtual tie
        games = []
        for pick in picks:
            a, b = index[pick["a"]], index[pick["b"]]
            share = {"a": 1.0, "b": 0.0, "same": 0.5}[pick["winner"]]
            wins[a] += share
            wins[b] += 1 - share
            games.append((a, b))
        strength = np.ones(len(designs))
        for _ in range(iterations):
            denominator = np.full(len(designs), 0.0)
            for a, b in games:
                denominator[a] += 1 / (strength[a] + strength[b])
                denominator[b] += 1 / (strength[a] + strength[b])
            denominator += 1 / (strength + 1.0)  # the virtual opponent has strength 1
            strength = wins / denominator
            strength /= np.exp(np.mean(np.log(strength)))
        logs = np.log(strength)
        spread = max(logs.std(), 0.5)
        for design, value in zip(designs, (logs - logs.mean()) / spread):
            out[design] = float(value)
    return out


def expertGradedEntries(path=Path("data/curation/expertRatings.jsonl")):
    """Ratings from the Plate Rater page (CAD experts, 1-9) as sheet-candidate entries.

    Each rater's ratings are standardised against their own mean and spread, averaged over
    raters per design, then mapped onto the 1-5 grade scale (mean -> 3, one spread -> 1
    grade). Only generated designs from the rating-sheet rounds become entries; ratings of
    real designs are kept in the file for later use."""
    byRater = {}
    for line in (Path(path).read_text().splitlines() if Path(path).exists() else []):
        entry = json.loads(line)
        byRater.setdefault(entry["rater"], {})[entry["id"]] = entry["rating"]  # a re-export replaces
    scores = {}
    for ratings in byRater.values():
        values = np.array(list(ratings.values()), dtype=float)
        if len(values) < 10:
            continue  # too few to know this rater's scale
        mean, spread = values.mean(), max(values.std(), 0.5)
        for designId, rating in ratings.items():
            scores.setdefault(designId, []).append((rating - mean) / spread)
    for designId, z in pairStrengths().items():
        scores.setdefault(designId, []).append(z)
    grades = {}
    for designId, zs in scores.items():
        if designId.startswith("sheet:"):
            _, roundName, sheet, candidate = designId.split(":")
            grades[(roundName, sheet, int(candidate))] = float(np.clip(3 + np.mean(zs), 1, 5))
    entries = []
    for roundName in sorted({key[0] for key in grades}):
        for line in (sheetRoot / roundName / "candidates.jsonl").read_text().splitlines():
            entry = json.loads(line)
            grade = grades.get((roundName, entry["sheet"], int(entry["candidate"])))
            if grade is not None:
                entry.update({"kind": "sheetCandidate", "round": roundName, "rater": "experts", "grade": grade})
                entries.append(entry)
    return entries


def expandReferences(entries):
    """Fill in geometry for compact grade entries (round, sheet, candidate) from each round's
    candidates.jsonl."""
    byRound = {}
    out = []
    for entry in entries:
        if "outline" not in entry and entry.get("kind") == "sheetCandidate":
            roundName = entry["round"]
            if roundName not in byRound:
                byRound[roundName] = {(c["sheet"], c["candidate"]): c for c in
                                      map(json.loads, (sheetRoot / roundName / "candidates.jsonl").read_text().splitlines())}
            entry = {**byRound[roundName][(entry["sheet"], entry["candidate"])], **entry}
        out.append(entry)
    return out


def compactRatings(path=ratingsPath):
    """Rewrite a ratings file that still carries full geometry as compact references."""
    keep = ("kind", "round", "sheet", "candidate", "rater", "grade")
    lines = [json.dumps({key: entry[key] for key in keep}) for entry in map(json.loads, Path(path).read_text().splitlines())]
    Path(path).write_text("".join(line + "\n" for line in lines))


def loadSheetRatings(path=ratingsPath, claudeGrades=True):
    """plateKey -> (plate, [(params, pockets, ribs, grade, entry)]) from every graded round.

    claudeGrades=False leaves out Claude's own sheet grades, so only the user's and the
    experts' grades train the models (Claude's leaned towards dense, busy designs)."""
    byPlate = {}
    if not Path(path).exists():
        return byPlate
    entries = expandReferences([json.loads(line) for line in Path(path).read_text().splitlines()]) if claudeGrades else []
    entries += userGradedEntries() + expertGradedEntries()
    # One grade per design: the user's beats the experts', which beats Claude's.
    rank = {"user": 2, "experts": 1}
    top = {}
    for e in entries:
        if e.get("kind") == "sheetCandidate":
            key = (e["round"], e["sheet"], e["candidate"])
            top[key] = max(top.get(key, 0), rank.get(e.get("rater"), 0))
    for entry in entries:
        if entry.get("kind") != "sheetCandidate":
            continue
        if rank.get(entry.get("rater"), 0) < top[(entry["round"], entry["sheet"], entry["candidate"])]:
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
