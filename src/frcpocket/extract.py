"""Turn scraped STEP/DXF files into a dataset of flat plates with their pockets.

Each output record (one JSON line in data/processed/plates.jsonl) holds the plate
outline, the holes/features to preserve, and the pockets, all in inches, plus
provenance from the source log.

Usage:
    python -I -m frcpocket.extract --raw data/raw/github data/raw/manual --out data/processed
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import time
import traceback
from pathlib import Path

from shapely.geometry import Point
from shapely.ops import unary_union

from frcpocket import geom2d

fileTimeoutSeconds = 240
maxSolidsPerFile = 400
dxfLineMinLengthIn = 0.1
arcAttachToleranceIn = 0.01
cacheVersion = 2  # bump when the raw geometry format changes
suspectWallIn = 0.03  # a pocket this close to the edge means overlapping DXF parts, not a real plate


def loopInches(loop, scale):
    return {
        "coords": [[x * scale, y * scale] for x, y in loop["coords"]],
        "nLines": loop["nLines"],
        "arcRadii": [radius * scale for radius in loop["arcRadii"]],
    }


def estimateLineCount(polygon):
    return geom2d.straightRunCount(polygon, minLength=dxfLineMinLengthIn)


def rawPlatesFromStep(path):
    """Geometry only (loops + edge metadata, inches); classification happens later."""
    from frcpocket import cadio

    scale = 1 / cadio.mmPerInch
    plates = []
    solids = cadio.importStepSolids(path)
    for solidIndex, solid in enumerate(solids[:maxSolidsPerFile]):
        try:
            plate = cadio.extractPlate(solid)
        except Exception:
            continue
        if plate is None:
            continue
        plates.append({
            "solidIndex": solidIndex,
            "thicknessIn": plate["thicknessMm"] * scale,
            "hasRaisedFeatures": plate["hasRaisedFeatures"],
            "outline": loopInches(plate["outer"], scale)["coords"],
            "through": [loopInches(loop, scale) for loop in plate["through"]],
            "faceLoops": [loopInches(loop, scale) for loop in plate["topLoops"] + plate["bottomLoops"]],
        })
    return plates


def rawPlatesFromDxf(path):
    from frcpocket import dxfio

    parts, arcs = dxfio.dxfParts(path)
    plates = []
    for partIndex, (outer, innerPolygons) in enumerate(parts):
        through = []
        for polygon in innerPolygons:
            boundary = polygon.exterior
            radii = [radius for midpoint, radius in arcs if boundary.distance(Point(midpoint)) < arcAttachToleranceIn]
            through.append({"coords": [list(point) for point in boundary.coords], "nLines": estimateLineCount(polygon), "arcRadii": radii})
        plates.append({
            "solidIndex": partIndex,
            "thicknessIn": None,
            "hasRaisedFeatures": False,
            "outline": [list(point) for point in outer.exterior.coords],
            "through": through,
            "faceLoops": [],
        })
    return plates


def recordFromRaw(raw):
    """Classify loops into holes and pockets and measure the layout."""
    outline = geom2d.toPolygon(raw["outline"])
    inner = [(geom2d.toPolygon(loop["coords"]), loop) for loop in raw["through"]]
    inner = [(polygon, loop) for polygon, loop in inner if not polygon.is_empty and polygon.area > 0]
    holes, pockets, pocketMeta = [], [], []
    for polygon, loop in inner:
        if geom2d.classifyLoop(loop, polygon) == "pocket":
            pockets.append(polygon)
            pocketMeta.append({"nLines": loop["nLines"], "arcRadiiIn": loop["arcRadii"]})
        else:
            holes.append(polygon)
    midMaterial = outline.difference(unary_union([polygon for polygon, _ in inner])) if inner else outline
    blind = []
    for loop in raw["faceLoops"]:
        polygon = geom2d.toPolygon(loop["coords"])
        if midMaterial.contains(polygon.representative_point()) and geom2d.classifyLoop(loop, polygon) == "pocket":
            blind.append(polygon)
    stats = geom2d.pocketStats(outline, holes, pockets)
    return {
        "solidIndex": raw["solidIndex"],
        "isGear": geom2d.isGearOutline(outline),
        "isSuspect": bool(stats["wallIn"]) and min(stats["wallIn"]) < suspectWallIn,
        "thicknessIn": raw["thicknessIn"],
        "hasRaisedFeatures": raw["hasRaisedFeatures"],
        "outline": [list(point) for point in outline.exterior.coords],
        "holes": [[list(point) for point in hole.exterior.coords] for hole in holes],
        "pockets": [[list(point) for point in pocket.exterior.coords] for pocket in pockets],
        "blindPockets": [[list(point) for point in pocket.exterior.coords] for pocket in blind],
        "pocketMeta": pocketMeta,
        "stats": stats,
    }


def platesFromStep(path):
    return [recordFromRaw(raw) for raw in rawPlatesFromStep(path)]


def platesFromDxf(path):
    return [recordFromRaw(raw) for raw in rawPlatesFromDxf(path)]


def processFile(path, cachePath):
    try:
        plates = rawPlatesFromDxf(path) if path.suffix.lower() == ".dxf" else rawPlatesFromStep(path)
        cachePath.write_text(json.dumps({"ok": True, "version": cacheVersion, "plates": plates}))
    except Exception:
        cachePath.write_text(json.dumps({"ok": False, "version": cacheVersion, "error": traceback.format_exc()[-2000:]}))


def cachePathFor(path, cacheDir):
    return cacheDir / f"{hashlib.sha1(str(path).encode()).hexdigest()[:16]}.json"


def isCached(cachePath):
    if not cachePath.exists():
        return False
    try:
        return json.loads(cachePath.read_text()).get("version") == cacheVersion
    except json.JSONDecodeError:
        return False


def loadSourceLog(logPaths):
    sources = {}
    for logPath in [logPaths] if isinstance(logPaths, (str, Path)) else logPaths:
        if Path(logPath).exists():
            for line in Path(logPath).read_text().splitlines():
                entry = json.loads(line)
                if "localPath" in entry:
                    sources[str(Path(entry["localPath"]))] = entry
    return sources


def signature(record):
    outlineArea = record["stats"]["outlineAreaIn2"]
    return (round(outlineArea, 2), record["stats"]["holeCount"], record["stats"]["pocketCount"],
            round(sum(geom2d.toPolygon(p).area for p in record["pockets"]), 2))


def runAll(files, cacheDir, nWorkers, timeoutSeconds=fileTimeoutSeconds):
    pending = []
    for path in files:
        cachePath = cachePathFor(path, cacheDir)
        if not isCached(cachePath):
            pending.append((path, cachePath))
    pending.sort(key=lambda item: item[0].stat().st_size)  # small files first: big assemblies can take minutes each
    print(f"{len(files)} files, {len(pending)} to process", flush=True)
    running = []
    while pending or running:
        while pending and len(running) < nWorkers:
            path, cachePath = pending.pop(0)
            process = multiprocessing.Process(target=processFile, args=(path, cachePath))
            process.start()
            running.append((process, path, cachePath, time.monotonic()))
        still = []
        for process, path, cachePath, startTime in running:
            process.join(timeout=0.2)
            if process.is_alive():
                if time.monotonic() - startTime > timeoutSeconds:
                    process.terminate()
                    cachePath.write_text(json.dumps({"ok": False, "version": cacheVersion, "error": "timeout"}))
                    print(f"  timeout: {path.name}", flush=True)
                else:
                    still.append((process, path, cachePath, startTime))
            else:
                print(f"  done: {path.name}", flush=True)
        running = still


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", nargs="+", default=["data/raw/github", "data/raw/manual"],
                        help="one or more folders of downloaded CAD (missing folders are skipped)")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--log", nargs="+", default=["data/sources.jsonl"], help="one or more source logs")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=fileTimeoutSeconds, help="seconds per file")
    args = parser.parse_args()

    rawDirs = [Path(raw) for raw in args.raw if Path(raw).exists()]
    outDir = Path(args.out)
    cacheDir = outDir / "cache"
    cacheDir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for rawDir in rawDirs for path in rawDir.rglob("*")
        if path.is_file() and "_meta" not in path.parts and path.suffix.lower() in (".step", ".stp", ".dxf")
    )
    runAll(files, cacheDir, args.workers, args.timeout)

    sources = loadSourceLog(args.log)
    seen = set()
    nFiles = nFailed = 0
    records = []
    for path in files:
        result = json.loads(cachePathFor(path, cacheDir).read_text())
        nFiles += 1
        if not result["ok"]:
            nFailed += 1
            continue
        source = sources.get(str(path), {})
        for raw in result["plates"]:
            record = recordFromRaw(raw)
            if record["isGear"] or record["isSuspect"]:
                continue
            key = signature(record)
            if key in seen:
                continue
            seen.add(key)
            record.update({
                "id": f"{hashlib.sha1(str(path).encode()).hexdigest()[:10]}_{record['solidIndex']}",
                "file": str(path),
                "format": path.suffix.lower().lstrip("."),
                "origin": {"manual-export": "manual", "vendor": "vendor"}.get(source.get("source"), "github"),
                "vendor": source.get("vendor"),
                "team": source.get("team"),
                "repo": source.get("repo"),
                "url": source.get("url"),
                "license": source.get("license"),
                "trainingAllowed": source.get("trainingAllowed", True),
            })
            records.append(record)
    with open(outDir / "plates.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    nPocketed = sum(record["stats"]["pocketCount"] > 0 for record in records)
    print(f"{nFiles} files ({nFailed} failed) -> {len(records)} unique plates, {nPocketed} pocketed", flush=True)


if __name__ == "__main__":
    main()
