"""Expert pairs page: two designs for the same plate side by side, pick the better one.

The scorer learns from exactly this comparison (on the same plate, A beats B), so a direct
pick carries no 1-9 scale noise. Pairs are chosen where they teach the most: designs Claude
graded about the same, and pairs where the scorer disagrees with Claude's grades.

    python -I -m frcpocket.expertPairs build --rounds m1 --count 400
    python -I -m frcpocket.expertPairs build --rounds n1 --versus m3,m3b --name 2
    python -I -m frcpocket.expertPairs ingest <pairs.json> [...]

Ingested picks go to data/curation/expertPairs.jsonl; ratingSheets turns them into one
grade per design per plate (Bradley-Terry) for the scorer, like the 1-9 ratings.
"""

import argparse
import hashlib
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import Polygon

from frcpocket.expertRater import compact, isSlop
from frcpocket.splits import isHoldout

outDir = Path("data/curation/expert")
sheetRoot = Path("data/curation/sheets")
pairsPath = Path("data/curation/expertPairs.jsonl")
templatePath = Path(__file__).parent / "webui" / "expertPairs.html"
scorerPath = "models/marathon/r1/aestheticScorer.pt"


def candidates(rounds):
    """Every non-slop generated design of the given rounds, grouped by plate (sheet)."""
    byPlate = {}
    for roundName in rounds:
        for line in (sheetRoot / roundName / "candidates.jsonl").read_text().splitlines():
            entry = json.loads(line)
            if isHoldout(entry.get("plateId") or entry["sheet"]) or isSlop(entry):
                continue
            entry["id"] = f"sheet:{roundName}:{entry['sheet']}:{entry['candidate']}"
            byPlate.setdefault((roundName, entry["sheet"]), []).append(entry)
    return byPlate


def informativePairs(byPlate, count, seed=5):
    """Pairs on the same plate, most informative first: close grades, then scorer-vs-grade
    disagreements, then the scorer's closest calls."""
    from frcpocket.plate import Plate, PocketParams
    from frcpocket.ratingSheets import loadSheetRatings
    from frcpocket.scorer import AestheticScorer

    grades = {}
    for _, entries in loadSheetRatings().values():
        for *_, grade, entry in entries:
            grades[f"sheet:{entry['round']}:{entry['sheet']}:{entry['candidate']}"] = grade
    scorer = AestheticScorer(scorerPath)
    pairs = []
    for designs in byPlate.values():
        if len(designs) < 2:
            continue
        first = designs[0]
        plate = Plate(outline=Polygon(first["outline"]), features=[Polygon(loop) for loop in first["holes"]],
                      thicknessIn=first.get("thicknessIn") or 0.25)
        params = PocketParams(**first["params"]) if first.get("params") else PocketParams()
        scores = scorer.score(plate, params, [[Polygon(loop) for loop in d["pockets"]] for d in designs])
        for (i, a), (j, b) in itertools.combinations(enumerate(designs), 2):
            ga, gb = grades.get(a["id"]), grades.get(b["id"])
            gap = abs(ga - gb) if ga is not None and gb is not None else None
            if gap is not None and gap >= 2:
                continue  # obvious to everyone
            disagree = gap is not None and gap > 0 and (ga - gb) * (scores[i] - scores[j]) < 0
            closeness = (gap if gap is not None else 1.0) + abs(float(scores[i] - scores[j]))
            pairs.append((0 if disagree else 1, closeness, a, b))
    pairs.sort(key=lambda item: item[:2])
    chosen, perPlate = [], {}
    for *_, a, b in pairs:  # spread over plates: at most 8 pairs per plate
        key = a["id"].rsplit(":", 1)[0]
        if perPlate.get(key, 0) >= 8:
            continue
        perPlate[key] = perPlate.get(key, 0) + 1
        chosen.append((a, b))
        if len(chosen) >= count:
            break
    random.Random(seed).shuffle(chosen)
    return chosen


def versusPairs(newRounds, oldRounds, perSide=2, closeNew=1, seed=7):
    """New generator against old on the same plate (each side's best by the live scorer),
    plus the closest calls between new designs: does the new version beat the old one,
    and which of the new looks does he prefer."""
    from frcpocket.plate import Plate, PocketParams
    from frcpocket.scorer import AestheticScorer

    def bySheet(rounds):
        grouped = {}
        for (_, sheet), designs in candidates(rounds).items():
            grouped.setdefault(sheet, []).extend(designs)
        return grouped

    def ranked(designs):
        first = designs[0]
        plate = Plate(outline=Polygon(first["outline"]), features=[Polygon(loop) for loop in first["holes"]],
                      thicknessIn=first.get("thicknessIn") or 0.25)
        params = PocketParams(**first["params"]) if first.get("params") else PocketParams()
        scores = scorer.score(plate, params, [[Polygon(loop) for loop in d["pockets"]] for d in designs])
        return sorted(zip((float(s) for s in scores), range(len(designs)), designs), reverse=True)

    scorer = AestheticScorer("models/aestheticScorer.pt")
    new, old = bySheet(newRounds), bySheet(oldRounds)
    chosen = []
    for sheet in sorted(set(new) & set(old)):
        bestNew, bestOld = ranked(new[sheet]), ranked(old[sheet])
        chosen += [(a, b) for *_, a in bestNew[:perSide] for *_, b in bestOld[:perSide]]
        close = sorted((abs(x[0] - y[0]), x[2], y[2]) for x, y in itertools.combinations(bestNew, 2))
        chosen += [(a, b) for _, a, b in close[:closeNew]]
    random.Random(seed).shuffle(chosen)
    return chosen


def build(rounds, count, name, versus=()):
    pairs = versusPairs(rounds, versus)[:count] if versus else informativePairs(candidates(rounds), count)
    designs = {}
    for a, b in pairs:
        for design in (a, b):
            if design["id"] not in designs:
                outline = Polygon(design["outline"])
                minX, minY, maxX, maxY = outline.bounds
                tolerance = 0.0015 * max(maxX - minX, maxY - minY)
                designs[design["id"]] = {
                    "w": round(maxX - minX, 2), "h": round(maxY - minY, 2),
                    "t": round(design["thicknessIn"], 3) if design.get("thicknessIn") else None,
                    "o": compact(design["outline"], tolerance),
                    "f": [compact(loop, tolerance * 0.5) for loop in design["holes"]],
                    "p": [compact(loop, tolerance) for loop in design["pockets"]]}
    ids = [[a["id"], b["id"]] for a, b in pairs]
    digest = hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:6]
    data = json.dumps({"version": f"pairs-{name}-{digest}", "designs": designs, "pairs": ids}, separators=(",", ":"))
    outDir.mkdir(parents=True, exist_ok=True)
    (outDir / f"pairs_{name}.json").write_text(json.dumps(ids))
    out = outDir / f"plate-pairs-{name}.html"
    out.write_text(templatePath.read_text().replace("/*__DATA__*/null", data))
    print(f"{len(pairs)} pairs over {len(designs)} designs, {out} {out.stat().st_size / 1e6:.1f} MB")


def ingest(paths):
    known = set()
    for path in outDir.glob("pairs_*.json"):
        known |= {tuple(pair) for pair in json.loads(path.read_text())}
    added = 0
    with open(pairsPath, "a") as handle:
        for path in paths:
            export = json.loads(Path(path).read_text())
            rater = "expert:" + (export.get("rater") or Path(path).stem)
            for pick in export["picks"]:
                if (pick["a"], pick["b"]) not in known or pick["winner"] not in ("a", "b", "same"):
                    continue
                handle.write(json.dumps({"a": pick["a"], "b": pick["b"], "winner": pick["winner"], "rater": rater,
                                         "version": export.get("version"),
                                         "ingested": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
                added += 1
    print(f"added {added} picks to {pairsPath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["build", "ingest"])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--rounds", default="m1")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--versus", default="", help="older rounds to pit the new rounds against")
    parser.add_argument("--name", default="1")
    args = parser.parse_args()
    if args.action == "build":
        build(tuple(filter(None, args.rounds.split(","))), args.count, args.name,
              tuple(filter(None, args.versus.split(","))))
    else:
        ingest(args.files)


if __name__ == "__main__":
    main()
