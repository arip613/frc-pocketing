"""Expert rating page: a self-contained HTML file that shows one design at a time for a
person to rate 1-9, and exports the ratings as JSON to send back.

Designs come from the rating-sheet rounds (generated candidates) and from the real
pocketed plates, shuffled together and never labelled. Held-out plates are never shown.

    python -I -m frcpocket.expertRater build --part 1   # -> data/curation/expert/
    python -I -m frcpocket.expertRater ingest <ratings.json> [...]

Each part holds only designs that no earlier part showed, so a rater can do part 1 now
and part 2 later; design ids are the same in every part.

Ingested ratings become rater "expert:<name>" entries in data/curation/expertRatings.jsonl,
which the scorer reads like the user's own grades (see ratingSheets.loadSheetRatings).
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import Polygon

from frcpocket.splits import isHoldout, loadRecords

outDir = Path("data/curation/expert")
sheetRoot = Path("data/curation/sheets")
ratingsPath = Path("data/curation/expertRatings.jsonl")
templatePath = Path(__file__).parent / "webui" / "expertRater.html"


def compact(loop, tolerance):
    """Ring coordinates simplified and rounded for the page (inches, 3 decimals)."""
    polygon = Polygon(loop)
    if tolerance > 0 and polygon.is_valid:
        polygon = polygon.simplify(tolerance, preserve_topology=True)
    return [round(value, 3) for point in list(polygon.exterior.coords)[:-1] for value in point]


def collectItems(target):
    """Generated sheet candidates (every round) and real pocketed designs, held-out excluded."""
    items = []
    for path in sorted(sheetRoot.glob("*/candidates.jsonl")):
        roundName = path.parent.name
        for line in path.open():
            entry = json.loads(line)
            if isHoldout(entry.get("plateId") or entry["sheet"]):
                continue
            items.append({"id": f"sheet:{roundName}:{entry['sheet']}:{entry['candidate']}", "kind": "generated",
                          "plateId": entry.get("plateId") or entry["sheet"], "thicknessIn": entry.get("thicknessIn"),
                          "params": entry.get("params"), "outline": entry["outline"], "holes": entry["holes"],
                          "pockets": entry["pockets"]})
    for record in loadRecords("data/processed/plates.jsonl"):
        if isHoldout(record["id"]) or not record.get("pockets"):
            continue
        items.append({"id": f"real:{record['id']}", "kind": "real", "plateId": record["id"],
                      "thicknessIn": record.get("thicknessIn"), "params": None, "outline": record["outline"],
                      "holes": record["holes"], "pockets": record["pockets"]})
    rng = random.Random(7)
    generated = [item for item in items if item["kind"] == "generated"]
    real = [item for item in items if item["kind"] == "real"]
    if len(generated) + len(real) > target:
        generated = rng.sample(generated, max(0, target - len(real)))
    return real + generated


def earlierIds(part):
    ids = set()
    for k in range(1, part):
        path = outDir / f"items_part{k}.jsonl"
        if path.exists():
            ids |= {json.loads(line)["id"] for line in path.open()}
    return ids


def build(target, part):
    shown = earlierIds(part)
    items = [item for item in collectItems(target + len(shown)) if item["id"] not in shown][:target]
    outDir.mkdir(parents=True, exist_ok=True)
    with open(outDir / f"items_part{part}.jsonl", "w") as handle:
        for item in items:
            # ids only: the geometry stays in each round's candidates.jsonl and in plates.jsonl
            handle.write(json.dumps({key: item[key] for key in ("id", "kind", "plateId")}) + "\n")
    page = []
    for item in items:
        outline = Polygon(item["outline"])
        minX, minY, maxX, maxY = outline.bounds
        tolerance = 0.0015 * max(maxX - minX, maxY - minY)
        page.append({"i": item["id"], "w": round(maxX - minX, 2), "h": round(maxY - minY, 2),
                     "t": round(item["thicknessIn"], 3) if item.get("thicknessIn") else None,
                     "o": compact(item["outline"], tolerance),
                     "f": [compact(loop, tolerance * 0.5) for loop in item["holes"]],
                     "p": [compact(loop, tolerance) for loop in item["pockets"]]})
    data = json.dumps({"version": str(part), "items": page}, separators=(",", ":"))
    html = templatePath.read_text().replace("/*__DATA__*/null", data).replace("__PART__", str(part))
    out = outDir / f"plate-rater-part{part}.html"
    out.write_text(html)
    kinds = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    print(f"{len(items)} designs {kinds}, {out} {out.stat().st_size / 1e6:.1f} MB")


def ingest(paths):
    known = set()
    for path in outDir.glob("items_part*.jsonl"):
        known |= {json.loads(line)["id"] for line in path.open()}
    added = 0
    with open(ratingsPath, "a") as handle:
        for path in paths:
            export = json.loads(Path(path).read_text())
            rater = "expert:" + (export.get("rater") or Path(path).stem).strip().lower().replace(" ", "_")
            for entry in export["ratings"]:
                if entry["id"] not in known or not 1 <= int(entry["rating"]) <= 9:
                    continue
                handle.write(json.dumps({"id": entry["id"], "rating": int(entry["rating"]), "rater": rater,
                                         "datasetVersion": export.get("version"),
                                         "ingested": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
                added += 1
    print(f"added {added} ratings to {ratingsPath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "ingest"])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--part", type=int, default=1)
    args = parser.parse_args()
    if args.action == "build":
        build(args.count, args.part)
    else:
        ingest(args.files)


if __name__ == "__main__":
    main()
