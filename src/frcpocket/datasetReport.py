"""Summarize the extracted plate dataset: what real FRC pocketing measures out to.

Usage:
    python -I -m frcpocket.datasetReport --plates data/processed/plates.jsonl --out data/processed
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from frcpocket import render

exemplarMinPockets = 3
exemplarMinRemoved = 0.15


def percentiles(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {name: round(float(np.percentile(array, q)), 3) for name, q in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))} | {"n": len(values)}


def pocketShape(meta):
    lines = meta["nLines"]
    if lines == 3:
        return "triangle"
    if lines == 4:
        return "quad"
    if lines in (5, 6):
        return "5-6 sided"
    if lines >= 7:
        return "7+ sided"
    return "curved"


def isExemplar(record):
    """Plates with a real pocketing layout, not one or two cut-outs."""
    stats = record["stats"]
    return stats["pocketCount"] >= exemplarMinPockets and stats["removedFraction"] >= exemplarMinRemoved


def summarize(records):
    walls, ribs, bosses, radii, removed, symmetry, thickness, ribRatio = [], [], [], [], [], [], [], []
    shapes = Counter()
    for record in records:
        stats = record["stats"]
        walls.extend(stats["wallIn"])
        ribs.extend(stats["ribIn"])
        bosses.extend(stats["bossIn"])
        removed.append(stats["removedFraction"])
        symmetry.append(stats["symmetry"])
        if record.get("thicknessIn"):
            thickness.append(record["thicknessIn"])
            ribRatio.extend(rib / record["thicknessIn"] for rib in stats["ribIn"])
        for meta in record["pocketMeta"]:
            radii.extend(meta["arcRadiiIn"])
            shapes[pocketShape(meta)] += 1
    return {
        "plates": len(records),
        "repos": sorted({record["repo"] for record in records if record.get("repo")}),
        "pocketCount": percentiles([record["stats"]["pocketCount"] for record in records]),
        "thicknessIn": percentiles(thickness),
        "minWallPerPlateIn": percentiles([min(record["stats"]["wallIn"]) for record in records]),
        "wallIn": percentiles(walls),
        "ribIn": percentiles(ribs),
        "ribOverThickness": percentiles(ribRatio),
        "bossIn": percentiles(bosses),
        "cornerRadiusIn": percentiles(radii),
        "removedFraction": percentiles(removed),
        "symmetryIoU": percentiles(symmetry),
        "pocketShapes": dict(shapes.most_common()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plates", default="data/processed/plates.jsonl")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.plates).read_text().splitlines()]
    pocketed = [record for record in records if record["stats"]["pocketCount"] > 0]
    exemplars = [record for record in pocketed if isExemplar(record)]
    report = {
        "plates": len(records),
        "pocketedPlates": len(pocketed),
        "exemplarPlates": len(exemplars),
        "exemplarRule": f"pocketCount >= {exemplarMinPockets} and removedFraction >= {exemplarMinRemoved}",
        "allPocketed": summarize(pocketed),
        "exemplars": summarize(exemplars),
    }
    outDir = Path(args.out)
    (outDir / "datasetReport.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    pocketed.sort(key=lambda record: -record["stats"]["pocketCount"])
    titles = [
        f"{(record.get('repo') or '?').split('/')[-1][:20]}  {record['stats']['pocketCount']} pockets, "
        f"{record['stats']['removedFraction']:.0%}"
        for record in pocketed
    ]
    previewDir = outDir / "previews"
    previewDir.mkdir(exist_ok=True)
    for start in range(0, len(pocketed), 36):
        render.contactSheet(pocketed[start:start + 36], previewDir / f"realPocketed_{start // 36:02d}.png", titles[start:start + 36])


if __name__ == "__main__":
    main()
