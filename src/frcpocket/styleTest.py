"""Style test: rebuild the user's held-out kept plates from outline and holes alone.

The user kept 95 plates as good pocketing (data/curation/plateKeeper.json). The 13 from
the Sim robot export (ids 1bb1ce9e48_*) are held out of all training (splits.isHoldout),
as are user examples 5 and 8. Each is given to a model set as an unpocketed plate, under
the widths measured from the real design (evaluate.styleParamsFromRecord), and the model
set's best sound ML design (Pocketer.candidates, count=2) is compared with the real one:

- materialIoU: IoU of plate material (plate minus pockets) inside the wall band,
  outline.buffer(-wallIn), as projection.projectRealDesign's pocketIoU. noPocketsIoU is the
  same for the unpocketed plate: the floor any design starts from.
- pocketCountRatio: our pocket count / the real count.
- removedDiff: our removed share minus the real one (share of plate material).
- medianAreaRatio: our median pocket area / the real median.
- tasteOurs, tasteReal: the taste model's 0-1 score (models/marathon/v2/taste.json, the same
  for every model set) of each design; tasteDiff = ours - real, and tastePrefersReal says
  on how many plates the real design scores higher.

    python -I -m frcpocket.styleTest --name live --out <dir>
    python -I -m frcpocket.styleTest --name foo --generators a.pt,b.pt --scorer x.json --out <dir>

Writes <out>/results.jsonl (one line per plate, as each finishes), <out>/sideBySide.png
(per plate: real | ours) and <out>/summary.json, and prints the summary table. --resume
skips plates already in results.jsonl; --report-only redraws the PNG and summary.
"""

import argparse
import json
import multiprocessing
import multiprocessing.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

platesPath = Path("data/processed/plates.jsonl")
tastePath = Path("models/marathon/v2/taste.json")
extraTestIds = ("userImg_example5", "userImg_example8")
candidateCount = 2
metricNames = ("materialIoU", "noPocketsIoU", "pocketCountRatio", "removedOurs", "removedReal", "removedDiff",
               "medianAreaRatio", "tasteOurs", "tasteReal", "tasteDiff")


def liveModelSet():
    """The live Pocketer's generators and scorer."""
    from frcpocket.pocketer import defaultGeneratorPath, defaultScorerPath

    return {"name": "live", "generators": [str(path) for path in defaultGeneratorPath], "scorer": str(defaultScorerPath)}


def normalizeModelSet(modelSet):
    """Fill in the live defaults and check every model file exists (Pocketer skips missing
    generators silently, which would test a different set than asked)."""
    live = liveModelSet()
    modelSet = dict(modelSet or {})
    modelSet.setdefault("name", "live")
    generators = modelSet.get("generators") or live["generators"]
    modelSet["generators"] = [str(path) for path in ([generators] if isinstance(generators, (str, Path)) else generators)]
    modelSet["scorer"] = str(modelSet.get("scorer") or live["scorer"])
    # "truss" names Pocketer's relaxed-truss engine, not a file.
    missing = [path for path in modelSet["generators"] + [modelSet["scorer"]] if path != "truss" and not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"model files not found: {missing}")
    return modelSet


def loadRecords():
    """Records by id: plates.jsonl plus the user's example images (splits.loadRecords)."""
    from frcpocket.splits import loadRecords as splitRecords

    return {record["id"]: record for record in splitRecords(platesPath)}


def testPlates(records=None):
    """The held-out kept plate ids, then user examples 5 and 8 when present."""
    from frcpocket.splits import isHoldout, keeperIds

    records = loadRecords() if records is None else records
    held = sorted((plateId for plateId in keeperIds if isHoldout(plateId) and plateId in records),
                  key=lambda plateId: (plateId.split("_")[0], int(plateId.split("_")[-1]) if plateId.split("_")[-1].isdigit() else 0))
    return held + [plateId for plateId in extraTestIds if plateId in records]


def realPockets(record):
    from frcpocket import geom2d

    return [pocket for pocket in (geom2d.toPolygon(coords) for coords in record["pockets"]) if pocket.area > 0]


def keptMaterial(plate, pockets, region):
    from shapely.ops import unary_union

    material = plate.material.difference(unary_union(pockets)) if pockets else plate.material
    return material.intersection(region)


def materialIoU(plate, wallIn, pocketsA, pocketsB):
    """IoU of the material two designs keep inside the wall band (projection's pocketIoU)."""
    region = plate.outline.buffer(-wallIn)
    a, b = keptMaterial(plate, pocketsA, region), keptMaterial(plate, pocketsB, region)
    return a.intersection(b).area / max(a.union(b).area, 1e-9)


def polygonCoords(polygon, digits=5):
    parts = getattr(polygon, "geoms", [polygon])
    return [[[round(x, digits), round(y, digits)] for x, y in part.exterior.coords] for part in parts]


def compare(plate, params, ours, real, tasteOurs, tasteReal):
    """Metrics of our design against the real one, both on the same plate and params."""
    from frcpocket.layout import removedFraction

    removedOurs, removedReal = removedFraction(plate, ours), removedFraction(plate, real)
    return {
        "materialIoU": materialIoU(plate, params.wallIn, ours, real),
        "noPocketsIoU": materialIoU(plate, params.wallIn, [], real),
        "pocketCountRatio": len(ours) / len(real) if real else None,
        "removedOurs": removedOurs,
        "removedReal": removedReal,
        "removedDiff": removedOurs - removedReal,
        "medianAreaRatio": (float(np.median([p.area for p in ours])) / float(np.median([p.area for p in real])))
        if ours and real else None,
        "tasteOurs": tasteOurs,
        "tasteReal": tasteReal,
        "tasteDiff": tasteOurs - tasteReal,
    }


# Per worker process: its own Pocketer and taste model, built once.
_worker = {}


def initWorker(modelSet, feaWorkers):
    import torch

    from frcpocket import designFeatures
    from frcpocket.pocketer import Pocketer
    from frcpocket.taste import TasteScorer

    # Loading a looks scorer sets designFeatures.styleReference from its metadata; keep both
    # versions so the taste metric and the model set's own ranking each use their own.
    taste = TasteScorer(tastePath)
    tasteReference = dict(designFeatures.styleReference)
    pocketer = Pocketer(modelSet["generators"], modelSet["scorer"], workers=feaWorkers)
    torch.set_num_threads(1)  # the plate workers are the parallelism
    _worker.update(modelSet=modelSet, pocketer=pocketer, taste=taste, tasteReference=tasteReference,
                   pocketerReference=dict(designFeatures.styleReference))
    # The Judge's FEA pool must close before this worker exits: multiprocessing joins a
    # child process's own children before the executor's atexit shutdown would run, so an
    # open pool hangs the worker (and the run) at the end. Priority >= 0 runs before that
    # join, and above 10 runs before the pool's queues close their feeder threads (priority
    # 10), which would otherwise swallow the stop sentinels.
    multiprocessing.util.Finalize(None, closeWorker, exitpriority=100)


def closeWorker():
    pocketer = _worker.get("pocketer")
    if pocketer is not None and pocketer.judge.pool is not None:
        pocketer.judge.pool.shutdown(wait=True, cancel_futures=True)
        pocketer.judge.pool = None


def tasteScores(plate, params, pocketSets):
    from frcpocket import designFeatures

    def use(reference):
        designFeatures.styleReference.clear()
        designFeatures.styleReference.update(reference)

    use(_worker["tasteReference"])
    try:
        return [float(score) for score in _worker["taste"].score(plate, params, pocketSets)]
    finally:
        use(_worker["pocketerReference"])


def plateJob(record):
    """Design one test plate with the worker's model set and compare it with the real design."""
    from frcpocket.evaluate import styleParamsFromRecord
    from frcpocket.plate import plateFromRecord

    started = time.time()
    plate = plateFromRecord(record)
    params = styleParamsFromRecord(record)
    real = realPockets(record)
    row = {
        "id": record["id"], "modelSet": _worker["modelSet"]["name"],
        "params": {"wallIn": params.wallIn, "ribIn": params.ribIn, "bossIn": params.boss, "cornerRadiusIn": params.cornerRadiusIn},
        "realPocketCount": len(real),
    }
    try:
        results = _worker["pocketer"].candidates(plate, params, count=candidateCount)
    except Exception as error:  # one plate's failure must not end the run
        row["error"] = repr(error)
        results = []
    row["candidates"] = [{"source": r["source"], "sound": bool(r["sound"]), "removed": r["removed"]} for r in results]
    picks = [r for r in results if r["source"] == "ML generator" and r["sound"]]
    if picks:
        best = picks[0]  # candidates come sorted: sound first, then by utility
        ours = list(best["layout"].pockets)
        tasteOurs, tasteReal = tasteScores(plate, params, [ours, real])
        row.update(
            removed=best["removed"], pocketCount=len(ours), relaxed=best.get("relaxed"),
            utility=best["utility"], stiffnessKept=best["stiffnessKept"], stressIncrease=best["stressIncrease"],
            pockets=[coords for pocket in ours for coords in polygonCoords(pocket)],
            metrics=compare(plate, params, ours, real, tasteOurs, tasteReal),
        )
    else:
        row.update(removed=None, pocketCount=0, pockets=None, metrics=None, tasteReal=tasteScores(plate, params, [real])[0])
    row["seconds"] = round(time.time() - started, 1)
    return row


def readRows(resultsPath):
    """Rows of a results.jsonl, the last one per plate id, in file order."""
    rows = {}
    if Path(resultsPath).exists():
        for line in Path(resultsPath).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def run(modelSet=None, outDir="styleTest", plates=None, parallel=4, feaWorkers=1, resume=False):
    """Design every test plate (or `plates`, a list of ids) with the model set, in `parallel`
    spawn worker processes (each with its own Pocketer and `feaWorkers` FEA processes).
    modelSet: {"name", "generators": [paths], "scorer": path}; None = the live Pocketer.
    Returns the summary (also written to outDir/summary.json)."""
    modelSet = normalizeModelSet(modelSet)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    records = loadRecords()
    ids = list(plates) if plates else testPlates(records)
    unknown = [plateId for plateId in ids if plateId not in records]
    if unknown:
        raise KeyError(f"plates not in the dataset: {unknown}")
    resultsPath = outDir / "results.jsonl"
    done = {plateId for plateId, row in readRows(resultsPath).items() if "error" not in row} if resume else set()
    if not resume:
        resultsPath.write_text("")
    todo = [plateId for plateId in ids if plateId not in done]
    print(f"style test '{modelSet['name']}': {len(todo)} plates to design ({len(done)} done), {parallel} workers", flush=True)
    started = time.time()
    if todo:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, min(parallel, len(todo))), mp_context=context,
                                 initializer=initWorker, initargs=(modelSet, feaWorkers)) as pool:
            futures = {pool.submit(plateJob, records[plateId]): plateId for plateId in todo}
            for count, future in enumerate(as_completed(futures), 1):
                try:
                    row = future.result()
                except Exception as error:
                    row = {"id": futures[future], "modelSet": modelSet["name"], "error": repr(error), "metrics": None}
                with resultsPath.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                metrics = row.get("metrics") or {}
                status = row.get("error") or (f"IoU {metrics['materialIoU']:.2f}, taste {metrics['tasteOurs']:.2f} vs real "
                                              f"{metrics['tasteReal']:.2f}" if metrics else "no sound ML design")
                print(f"[{count}/{len(todo)}] {row['id']} ({row.get('seconds', '?')} s): {status}", flush=True)
    print(f"designed in {time.time() - started:.0f} s", flush=True)
    return report(outDir, modelSet, ids, records)


def summarize(rows, modelSet):
    designed = [row for row in rows if row.get("metrics")]
    metrics = {}
    for name in metricNames:
        values = [row["metrics"][name] for row in designed if row["metrics"].get(name) is not None]
        metrics[name] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "n": len(values)} if values else None
    return {
        "modelSet": modelSet,
        "plates": len(rows),
        "designed": len(designed),
        "noSoundDesign": [row["id"] for row in rows if not row.get("metrics") and "error" not in row],
        "errors": {row["id"]: row["error"] for row in rows if "error" in row},
        "relaxed": {row["id"]: row["relaxed"] for row in designed if row.get("relaxed")},
        "tastePrefersReal": sum(row["metrics"]["tasteReal"] > row["metrics"]["tasteOurs"] for row in designed),
        "metrics": metrics,
        "perPlate": {row["id"]: row.get("metrics") for row in rows},
    }


def summaryTable(summary):
    name = summary["modelSet"]["name"] if isinstance(summary["modelSet"], dict) else summary["modelSet"]
    lines = [f"style test '{name}': {summary['designed']} of {summary['plates']} plates designed; "
             f"taste model prefers the real design on {summary['tastePrefersReal']} of {summary['designed']}",
             f"{'metric':<18}{'mean':>9}{'median':>9}{'n':>4}"]
    for metricName, values in summary["metrics"].items():
        if values:
            lines.append(f"{metricName:<18}{values['mean']:>9.3f}{values['median']:>9.3f}{values['n']:>4}")
    if summary["noSoundDesign"]:
        lines.append(f"no sound ML design: {', '.join(summary['noSoundDesign'])}")
    if summary["relaxed"]:
        lines.append("relaxed: " + ", ".join(f"{plateId} ({note})" for plateId, note in summary["relaxed"].items()))
    if summary["errors"]:
        lines.append("errors: " + "; ".join(f"{plateId}: {error}" for plateId, error in summary["errors"].items()))
    lines.append("")
    lines.append(f"{'plate':<18}{'IoU':>6}{'(none)':>7}{'count':>7}{'removed':>9}{'area':>7}{'taste':>7}{'real':>6}")
    for plateId, metrics in summary["perPlate"].items():
        if metrics:
            count = metrics["pocketCountRatio"]
            area = metrics["medianAreaRatio"]
            count = "-" if count is None else f"x{count:.2f}"
            area = "-" if area is None else f"x{area:.2f}"
            lines.append(f"{plateId:<18}{metrics['materialIoU']:>6.2f}{metrics['noPocketsIoU']:>7.2f}{count:>7}"
                         f"{metrics['removedDiff']:>+9.2f}{area:>7}{metrics['tasteOurs']:>7.2f}{metrics['tasteReal']:>6.2f}")
        else:
            lines.append(f"{plateId:<18}  (no design)")
    return "\n".join(lines)


def drawSideBySide(path, rows, records, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from frcpocket import render
    from frcpocket.layout import removedFraction
    from frcpocket.plate import plateFromRecord

    figure, axes = plt.subplots(len(rows), 2, figsize=(10, 4.2 * len(rows)), dpi=90, squeeze=False)
    for (realAxis, oursAxis), row in zip(axes, rows):
        record = records[row["id"]]
        plate = plateFromRecord(record)
        outline = list(plate.outline.exterior.coords)
        holes = [list(feature.exterior.coords) for feature in plate.features]
        real = realPockets(record)
        metrics = row.get("metrics")
        tasteReal = metrics["tasteReal"] if metrics else row.get("tasteReal")
        realTitle = (f"real (kept by user)\n{row['id']}: {removedFraction(plate, real):.0%} removed, {len(real)} pockets"
                     + (f", taste {tasteReal:.2f}" if tasteReal is not None else ""))
        render.drawPlate(realAxis, outline, holes, [list(pocket.exterior.coords) for pocket in real], title=realTitle)
        if metrics:
            count, area = metrics["pocketCountRatio"], metrics["medianAreaRatio"]
            oursTitle = (f"{name}: {metrics['removedOurs']:.0%} removed ({metrics['removedDiff']:+.0%}), "
                         f"{row['pocketCount']} pockets (x{count:.2f}), taste {metrics['tasteOurs']:.2f}\n"
                         f"material IoU {metrics['materialIoU']:.2f} (unpocketed {metrics['noPocketsIoU']:.2f}), "
                         f"median pocket area x{area:.2f}" + (f"\nrelaxed: {row['relaxed']}" if row.get("relaxed") else ""))
            render.drawPlate(oursAxis, outline, holes, row["pockets"], title=oursTitle)
        else:
            render.drawPlate(oursAxis, outline, holes, [],
                             title=f"{name}: " + (f"error {row['error'][:60]}" if "error" in row else "no sound ML design"))
        for axis in (realAxis, oursAxis):
            axis.title.set_fontsize(8.5)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def report(outDir, modelSet=None, ids=None, records=None):
    """Summary table, summary.json and sideBySide.png from outDir/results.jsonl."""
    outDir = Path(outDir)
    byId = readRows(outDir / "results.jsonl")
    records = loadRecords() if records is None else records
    if ids is None:  # test-plate order, then any other plates in the file
        ids = testPlates(records) + [plateId for plateId in byId if plateId not in set(testPlates(records))]
    rows = [byId[plateId] for plateId in ids if plateId in byId]
    if not rows:
        raise ValueError(f"no results in {outDir / 'results.jsonl'}")
    if modelSet is None:
        summaryPath = outDir / "summary.json"
        modelSet = json.loads(summaryPath.read_text())["modelSet"] if summaryPath.exists() else {"name": rows[0]["modelSet"]}
    summary = summarize(rows, modelSet)
    (outDir / "summary.json").write_text(json.dumps(summary, indent=1))
    drawSideBySide(outDir / "sideBySide.png", rows, records, modelSet["name"])
    print(summaryTable(summary), flush=True)
    print(f"wrote {outDir / 'summary.json'}, {outDir / 'sideBySide.png'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help="model set name (default: live)")
    parser.add_argument("--generators", default=None, help="comma-separated generator .pt paths (default: the live set)")
    parser.add_argument("--scorer", default=None, help="scorer .pt or taste .json (default: the live scorer)")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--plates", default=None, help="comma-separated plate ids (default: all test plates)")
    parser.add_argument("--parallel", type=int, default=4, help="plate worker processes")
    parser.add_argument("--fea-workers", type=int, default=1, help="FEA processes per plate worker")
    parser.add_argument("--resume", action="store_true", help="skip plates already in <out>/results.jsonl")
    parser.add_argument("--report-only", action="store_true", help="only redraw the PNG and summary from results.jsonl")
    args = parser.parse_args()
    if args.generators and not args.name:
        parser.error("--name is required with --generators")
    modelSet = {"name": args.name or "live",
                "generators": args.generators.split(",") if args.generators else None,
                "scorer": args.scorer}
    # Through the package module, so worker jobs pickle as frcpocket.styleTest.* (not __main__).
    from frcpocket import styleTest

    plates = args.plates.split(",") if args.plates else None
    if args.report_only:
        styleTest.report(args.out, None, plates)
    else:
        styleTest.run(modelSet, args.out, plates=plates, parallel=args.parallel, feaWorkers=args.fea_workers, resume=args.resume)


if __name__ == "__main__":
    main()
