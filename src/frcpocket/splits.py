"""Which real plates are gold examples, and which are held out from all training.

Gold = the most aesthetically pleasing real pocketing (curated by eye from the 15
exemplars, plus the user's thumbs-up ratings). The other exemplars (a cluttered
electronics plate, a spoked wheel, sprocket-like rings, a frame with three plain
rectangles) are kept as neutral: never positives, never negatives.

The holdout is chosen by source file so sibling plates from one CAD file can't leak
between train and test. It includes one of the user's liked plates (Team 221) to
test whether the model generalizes to the preferred style.
"""

import json
from pathlib import Path

goldPlateIds = {
    "f1da8bbfec_90",  # WCP 3-CIM gearbox plate (user liked)
    "0d9cbb6d3b_0",  # WCP SS flipped gearbox plate (user liked)
    "76fd2d183d_0",  # Team 221 slim gearbox triple (user liked)
    "76fd2d183d_1",  # Team 221 slim gearbox triple, other plate
    "2d546bad45_21",  # gearbox / module plate, FRC-2024-CAD
    "18afc8e7d3_0",  # AndyMark Rhino track module plates
    "18afc8e7d3_12",
    "18afc8e7d3_52",
    "f02eefd773_0",  # Team 4400 2023 plates
    "f02eefd773_1",
}
# Example screenshots the user picked as the look to match (data/userExamples/images,
# geometry extracted by imageExtract.py).
userImageGold = {f"userImg_example{n}" for n in range(1, 9)}
goldPlateIds |= userImageGold

holdoutFiles = {
    "76fd2d183d", "f02eefd773",  # CAD gold: by source-file prefix of the plate id
    "userImg_example5", "userImg_example8",  # user examples without a near-twin in the set
}
ratingsPath = Path("data/curation/ratings.jsonl")
extraRecordPaths = [Path("data/processed/userExamples.jsonl")]


def sourceFile(plateId):
    # Each user example image is its own source; CAD plates group by file.
    return plateId if plateId.startswith("userImg_") else plateId.split("_")[0]


def loadRecords(platesPath):
    """Extracted CAD plates plus the user's example images, as one list of records."""
    records = [json.loads(line) for line in Path(platesPath).read_text().splitlines()]
    for extra in extraRecordPaths:
        if extra.exists():
            records += [json.loads(line) for line in extra.read_text().splitlines()]
    return records


def isHoldout(plateId):
    return sourceFile(plateId) in holdoutFiles


def userRatings():
    """plateId -> latest rating (+1 / -1) from the ratings log (UI thumbs up/down)."""
    ratings = {}
    if ratingsPath.exists():
        for line in ratingsPath.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("plateId"):
                ratings[entry["plateId"]] = entry["rating"]
    return ratings


def goldIds():
    ratings = userRatings()
    return (goldPlateIds | {plateId for plateId, rating in ratings.items() if rating > 0}) - {
        plateId for plateId, rating in ratings.items() if rating < 0
    }
