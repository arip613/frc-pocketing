"""Evaluate a layout generator on the test plates and real plates, with previews.

Real plates are used as unpocketed inputs (outline + holes). The generator runs under
parameters measured from the real design (its thinnest wall/boss, thinnest rib, and
tightest corner radius), so both designs compete under the same rules. The real
design is also put through the validator under those rules, which checks the
validator is no stricter than real teams are.

    python -I -m frcpocket.evaluate --generator baseline --round round0_baseline
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

from frcpocket import cadExport, fea, geom2d, render, testPlates, validator
from frcpocket.splits import goldIds, isHoldout, loadRecords
from frcpocket.baseline import trussBaseline
from frcpocket.layout import defaultGraph, removedFraction
from frcpocket.plate import PocketParams, loadPlateFromStep, plateFromRecord

_pocketer = None
lastToolPick = {"source": None}


def mlGenerator(plate, params):
    """Best sound ML candidate by utility, else an empty layout. Also records which design
    the tool (UI) would put first across ML and baseline candidates."""
    global _pocketer
    from frcpocket.layout import Layout, candidateGraph
    from frcpocket.pocketer import Pocketer

    if _pocketer is None:
        _pocketer = Pocketer()
    candidates = _pocketer.candidates(plate, params, count=6)
    lastToolPick["source"] = candidates[0]["source"] if candidates else None
    results = [result for result in candidates if result["source"] == "ML generator" and result["sound"]]
    if results:
        return results[0]["layout"]
    return Layout(graph=defaultGraph(plate, params), ribs=[], pockets=[], name="ml-none")


generators = {"baseline": trussBaseline, "ml": mlGenerator}


@dataclass
class DesignMetrics:
    valid: bool
    failedRules: list
    pocketCount: int
    removedFraction: float
    stiffnessKept: float
    stressIncrease: float
    specificStiffness: float
    maxDisplacementIn: float
    peakStressPsi: float
    aestheticScore: float | None = None  # filled in once the scorer is trained


def effectiveToolRadius(pockets, upper):
    """Largest tool radius (<= upper) that can cut every pocket as drawn."""
    low, high = 1 / 128, upper
    for _ in range(14):
        middle = (low + high) / 2
        cuttable = all(
            shapely.hausdorff_distance(
                pocket.boundary,
                pocket.buffer(-middle, quad_segs=16).buffer(middle, quad_segs=16).boundary,
                densify=0.1,
            ) <= validator.lengthTolIn
            for pocket in pockets
        )
        low, high = (middle, high) if cuttable else (low, middle)
    return low


def styleParamsFromRecord(record):
    """Typical (not minimum) widths of a real design: for projecting it onto the rib graph."""
    stats = record["stats"]
    rules = paramsFromRecord(record)
    wall = max(rules.wallIn, min(stats["wallIn"]) * 0.98) if stats["wallIn"] else rules.wallIn
    boss = max(0.06, float(np.percentile(stats["bossIn"], 25)) * 0.98) if stats["bossIn"] else wall
    rib = float(np.median(stats["ribIn"])) * 0.98 if stats["ribIn"] else rules.ribIn
    return PocketParams(
        wallIn=wall,
        ribIn=max(rules.ribIn, rib),
        cornerRadiusIn=rules.cornerRadiusIn,
        bossIn=boss,  # the real rings, often thinner than the wall
        minPocketAreaIn2=rules.minPocketAreaIn2,
        minPocketWidthIn=rules.minPocketWidthIn,
    )


def paramsFromRecord(record):
    """The rules the real designer actually followed, rounded down slightly."""
    stats = record["stats"]
    wall = min(stats["wallIn"] + stats["bossIn"]) if stats["bossIn"] else min(stats["wallIn"])
    rib = min(stats["ribIn"]) if stats["ribIn"] else wall
    radii = [radius for meta in record["pocketMeta"] for radius in meta["arcRadiiIn"]]
    pockets = [geom2d.toPolygon(pocket) for pocket in record["pockets"]]
    corner = effectiveToolRadius(pockets, min(radii) if radii else 0.25)
    return PocketParams(
        wallIn=max(0.06, wall * 0.98),
        ribIn=max(0.06, rib * 0.98),
        cornerRadiusIn=max(1 / 32, corner * 0.98),
        minPocketAreaIn2=0.05,
        minPocketWidthIn=2 * max(1 / 32, corner * 0.98),
    )


_scorer = None


def aestheticOf(plate, params, pockets):
    global _scorer
    from frcpocket.scorer import AestheticScorer

    if not pockets or not Path("models/aestheticScorer.pt").exists():
        return None
    if _scorer is None:
        _scorer = AestheticScorer("models/aestheticScorer.pt")
    return float(_scorer.score(plate, params, [pockets])[0])


def measure(plate, params, pockets):
    report = validator.validateLayout(plate, params, pockets)
    structure = fea.analyze(plate, pockets, params)
    return DesignMetrics(
        aestheticScore=aestheticOf(plate, params, pockets),
        valid=report.passed,
        failedRules=[result.rule for result in report.failures],
        pocketCount=len(pockets),
        removedFraction=removedFraction(plate, pockets, params),
        stiffnessKept=structure.stiffnessKept,
        stressIncrease=structure.stressIncrease,
        specificStiffness=structure.specificStiffness,
        maxDisplacementIn=structure.maxDisplacementIn,
        peakStressPsi=structure.peakStressPsi,
    )


def drawPanels(path, plate, panels):
    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), dpi=110)
    axes = np.atleast_1d(axes)
    outline = list(plate.outline.exterior.coords)
    holes = [list(feature.exterior.coords) for feature in plate.features]
    for axis, (title, pockets) in zip(axes, panels):
        render.drawPlate(axis, outline, holes, [list(pocket.exterior.coords) for pocket in pockets], title=title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def caption(label, metrics):
    status = "valid" if metrics.valid else "INVALID " + ",".join(rule.split()[0] for rule in metrics.failedRules)
    return (f"{label}: {metrics.removedFraction:.0%} removed, {metrics.stiffnessKept:.0%} stiff, "
            f"x{metrics.stressIncrease:.1f} stress ({status})")


def evaluate(generatorName, roundName, platesPath, testPlateDir, outRoot, exportSteps=True):
    generate = generators[generatorName]
    outDir = Path(outRoot) / roundName
    previewDir = outDir / "previews"
    stepDir = outDir / "steps"
    previewDir.mkdir(parents=True, exist_ok=True)
    stepDir.mkdir(exist_ok=True)
    rows = []

    testPaths = testPlates.writeTestPlates(testPlateDir)
    defaultParams = PocketParams()
    for name, path in testPaths.items():
        plate = loadPlateFromStep(path)
        started = time.time()
        layout = generate(plate, defaultParams)
        seconds = time.time() - started
        metrics = measure(plate, defaultParams, layout.pockets)
        baseLayout = trussBaseline(plate, defaultParams) if generatorName != "baseline" else layout
        baseMetrics = measure(plate, defaultParams, baseLayout.pockets) if generatorName != "baseline" else metrics
        stepOk = None
        if exportSteps and metrics.valid:
            result = cadExport.buildPocketedSolid(plate, defaultParams, layout)
            stepOk = result.report2d.passed and result.report3d.passed
            if stepOk:
                cadExport.exportStep(result, stepDir / f"{name}.step")
        panels = [(caption(generatorName, metrics), layout.pockets)]
        if generatorName != "baseline":
            panels.insert(0, (caption("baseline", baseMetrics), baseLayout.pockets))
        drawPanels(previewDir / f"test_{name}.png", plate, panels)
        rows.append({"plate": name, "kind": "test", "holdout": True, "generator": asdict(metrics),
                     "baseline": asdict(baseMetrics), "stepExport": stepOk, "seconds": seconds,
                     "toolPick": lastToolPick["source"] if generatorName == "ml" else None})
        print(f"test {name}: {caption(generatorName, metrics)}; STEP {stepOk}", flush=True)

    records = loadRecords(platesPath)
    exemplars = [record for record in records
                 if record["stats"]["pocketCount"] >= 3 and record["stats"]["removedFraction"] >= 0.15]
    for record in exemplars:
        plate = plateFromRecord(record)
        params = paramsFromRecord(record)
        realPockets = [geom2d.toPolygon(pocket) for pocket in record["pockets"]]
        real = measure(plate, params, realPockets)
        started = time.time()
        layout = generate(plate, params)
        seconds = time.time() - started
        generated = measure(plate, params, layout.pockets)
        label = f"{(record.get('repo') or '?').split('/')[-1]} {record['id']}"
        panels = [(caption("real " + label[:28], real), realPockets), (caption(generatorName, generated), layout.pockets)]
        row = {"plate": record["id"], "kind": "real", "repo": record.get("repo"), "holdout": isHoldout(record["id"]),
               "gold": record["id"] in goldIds(), "params": asdict(params), "real": asdict(real),
               "generator": asdict(generated), "seconds": seconds}
        if generatorName == "ml":
            row["toolPick"] = lastToolPick["source"]
        if generatorName != "baseline":
            baseLayout = trussBaseline(plate, params)
            row["baseline"] = asdict(measure(plate, params, baseLayout.pockets))
            panels.insert(1, (caption("baseline", DesignMetrics(**row["baseline"])), baseLayout.pockets))
        drawPanels(previewDir / f"real_{record['id']}.png", plate, panels)
        rows.append(row)
        print(f"real {label}: real {caption('', real)} | {caption(generatorName, generated)}", flush=True)

    summary = summarize(rows, generatorName)
    (outDir / "report.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    (outDir / "summary.md").write_text(summaryMarkdown(summary, rows, generatorName, roundName))
    return summary


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize(rows, generatorName):
    groups = {"test plates": [row for row in rows if row["kind"] == "test"],
              "real plates (all 15)": [row for row in rows if row["kind"] == "real"],
              "held-out gold plates": [row for row in rows if row["kind"] == "real" and row["holdout"] and row.get("gold")]}
    labels = {"generator": generatorName + " alone", "tool": "tool pick (ML or baseline)",
              "baseline": "baseline", "real": "real teams"}
    summary = {}
    for groupName, groupRows in groups.items():
        for row in groupRows:
            if row.get("toolPick"):
                row["tool"] = row["generator"] if row["toolPick"] == "ML generator" else row["baseline"]
        for who in ("generator", "tool", "baseline", "real"):
            if who == "baseline" and generatorName == "baseline":
                continue
            designs = [row[who] for row in groupRows if who in row]
            if not designs:
                continue
            summary[f"{groupName} / {labels[who]}"] = {
                "plates": len(designs),
                "validRate": mean([1.0 if design["valid"] else 0.0 for design in designs]),
                "removedFraction": mean([design["removedFraction"] for design in designs]),
                "stiffnessKept": mean([design["stiffnessKept"] for design in designs]),
                "stressIncrease": mean([design["stressIncrease"] for design in designs]),
                "specificStiffness": mean([design["specificStiffness"] for design in designs]),
                "aestheticScore": mean([design.get("aestheticScore") for design in designs]),
            }
    stepRows = [row for row in rows if row.get("stepExport") is not None]
    summary["STEP export pass rate (test plates)"] = mean([1.0 if row["stepExport"] else 0.0 for row in stepRows])
    return summary


def summaryMarkdown(summary, rows, generatorName, roundName):
    lines = [f"# Evaluation: {roundName}", "",
             "| Group | Plates | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Aesthetic |",
             "|---|---|---|---|---|---|---|---|"]
    for name, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        looks = f"{stats['aestheticScore']:.2f}" if stats.get("aestheticScore") is not None else "n/a"
        lines.append(f"| {name} | {stats['plates']} | {stats['validRate']:.0%} | {stats['removedFraction']:.0%} | "
                     f"{stats['stiffnessKept']:.0%} | x{stats['stressIncrease']:.2f} | {stats['specificStiffness']:.2f} | {looks} |")
    picks = [row["toolPick"] for row in rows if row.get("toolPick")]
    if picks:
        lines += ["", f"Tool pick: ML design on {picks.count('ML generator')} of {len(picks)} plates, "
                      f"truss baseline on {picks.count('truss baseline')}."]
    stepRate = summary.get("STEP export pass rate (test plates)")
    if stepRate is not None:
        lines += ["", f"STEP export (exact geometry, rule 5) on test plates: {stepRate:.0%} pass."]
    failures = [row for row in rows if row["kind"] == "real" and not row["real"]["valid"]]
    if failures:
        lines += ["", "Real designs failing the validator under their own measured parameters:"]
        lines += [f"- {row['plate']} ({row['repo']}): {', '.join(row['real']['failedRules'])}" for row in failures]
    lines += ["", "Previews: `previews/` (real design, then baseline, then the generator)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="baseline", choices=sorted(generators))
    parser.add_argument("--round", default="round0_baseline")
    parser.add_argument("--plates", default="data/processed/plates.jsonl")
    parser.add_argument("--test-plates", default="data/testPlates")
    parser.add_argument("--out", default="data/eval")
    parser.add_argument("--no-step", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args.generator, args.round, args.plates, args.test_plates, args.out, exportSteps=not args.no_step)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
