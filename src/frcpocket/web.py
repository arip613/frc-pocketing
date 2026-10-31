"""Local web UI (hand-built front end in webui/): upload a plate STEP, generate candidates,
rate them, export the pocketed STEP. Same pipeline as the Gradio app.

    python -I -m frcpocket.web        then open http://127.0.0.1:7861
"""

import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from frcpocket import cadExport
from frcpocket.plate import PocketParams, isOpening, loadPlateFromStep
from frcpocket.pocketer import Pocketer

webDir = Path(__file__).parent / "webui"
ratingsPath = Path("data/curation/ratings.jsonl")
aluminiumLbPerIn3 = 0.0975  # 6061
densityFactors = {"dense": 0.8, "medium": 1.25, "open": 1.6}
mmPerInch = 25.4

app = FastAPI(title="Plate Pocketer")
plates = {}  # plate id -> {plate, name}
jobs = {}  # job id -> {plateId, params, results}
exports = {}  # export id -> path
_pocketer = None


def pocketer():
    global _pocketer
    if _pocketer is None:
        _pocketer = Pocketer()
    return _pocketer


def coords(polygon):
    return [[round(x, 5), round(y, 5)] for x, y in polygon.exterior.coords]


class Rules(BaseModel):
    plateId: str
    units: str = "in"
    wall: float
    rib: float
    corner: float
    boss: float
    depth: float | None = None  # None = through
    density: str = "medium"
    count: int = 6


class Pick(BaseModel):
    jobId: str
    index: int
    rating: int = 0


@app.post("/api/plate")
async def uploadPlate(file: UploadFile = File(...)):
    suffix = Path(file.filename or "plate.step").suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(400, "That isn't a STEP file (.step or .stp).")
    folder = Path(tempfile.mkdtemp())
    path = folder / Path(file.filename).name
    path.write_bytes(await file.read())
    try:
        plate = loadPlateFromStep(str(path))
    except ValueError as error:
        raise HTTPException(400, str(error))
    plateId = uuid.uuid4().hex[:12]
    plates[plateId] = {"plate": plate, "name": path.name}
    minX, minY, maxX, maxY = plate.outline.bounds
    return {
        "plateId": plateId, "name": path.name,
        "outline": coords(plate.outline), "holes": [coords(f) for f in plate.features],
        "thicknessIn": plate.thicknessIn, "widthIn": maxX - minX, "heightIn": maxY - minY,
        "weightLb": plate.material.area * plate.thicknessIn * aluminiumLbPerIn3,
    }


@app.post("/api/generate")
def generate(rules: Rules):
    entry = plates.get(rules.plateId)
    if entry is None:
        raise HTTPException(404, "Upload the plate again (the server restarted).")
    plate = entry["plate"]
    if min(rules.wall, rules.rib, rules.corner, rules.boss) <= 0:
        raise HTTPException(400, "Wall, rib, corner and boss all have to be above zero.")
    params = PocketParams.fromUnits(rules.units, wall=rules.wall, rib=rules.rib, cornerRadius=rules.corner,
                                    depth=rules.depth, boss=rules.boss,
                                    pocketSizeFactor=densityFactors.get(rules.density, 1.25))
    if params.depthIn is not None and params.depthIn >= plate.thicknessIn:
        raise HTTPException(400, "Blind depth has to be less than the plate thickness.")
    started = time.time()
    results = pocketer().candidates(plate, params, count=max(1, min(8, rules.count)))
    relaxed = results[0].get("relaxed") if results else None
    params = results[0].get("params", params) if results else params  # the settings actually used
    if not results:
        raise HTTPException(422, "Nothing valid fits with these rules. Try a thinner wall or rib, or a smaller corner radius.")
    jobId = uuid.uuid4().hex[:12]
    jobs[jobId] = {"plateId": rules.plateId, "params": params, "results": results}
    depthIn = params.depthIn if params.depthIn is not None else plate.thicknessIn
    materialIn3 = plate.material.area * plate.thicknessIn
    cutouts = [feature for feature in plate.features if isOpening(feature)]
    candidates = []
    for result in results:
        pockets = result["layout"].pockets
        removedIn3 = sum(pocket.area for pocket in pockets) * depthIn
        candidates.append({
            "source": "ML" if result["source"] == "ML generator" else "truss",
            "pockets": [coords(pocket) for pocket in pockets],
            "sound": bool(result.get("sound", True)),
            "looks": result.get("aesthetic"),
            "stiffnessKept": result.get("stiffnessKept"),
            "stressIncrease": result.get("stressIncrease"),
            "removed": removedIn3 / materialIn3,
            "savedLb": removedIn3 * aluminiumLbPerIn3,
            "minWallIn": min((plate.outline.exterior.distance(p) for p in pockets), default=None),
            "minCutoutWallIn": min((f.distance(p) for f in cutouts for p in pockets), default=None),
        })
    inner = plate.outline.buffer(-params.wallIn)
    wallLine = [coords(piece) for piece in getattr(inner, "geoms", [inner]) if not piece.is_empty]
    return {"jobId": jobId, "seconds": time.time() - started, "candidates": candidates, "wallLine": wallLine, "relaxed": relaxed,
            "params": {"wallIn": params.wallIn, "ribIn": params.ribIn, "cornerIn": params.cornerRadiusIn, "bossIn": params.boss}}


@app.post("/api/rate")
def rate(pick: Pick):
    job = jobs.get(pick.jobId)
    if job is None or not 0 <= pick.index < len(job["results"]):
        raise HTTPException(404, "Generate again first.")
    entry, result = plates[job["plateId"]], job["results"][pick.index]
    plate, params = entry["plate"], job["params"]
    plateHash = hashlib.sha1(json.dumps([list(plate.outline.exterior.coords),
                                         [list(f.exterior.coords) for f in plate.features]]).encode()).hexdigest()[:12]
    record = {
        "kind": "uiCandidate", "rating": 1 if pick.rating > 0 else -1, "rater": "user",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plateHash": plateHash, "stepName": entry["name"], "source": result["source"],
        "params": vars(params), "thicknessIn": plate.thicknessIn,
        "outline": list(plate.outline.exterior.coords),
        "holes": [list(feature.exterior.coords) for feature in plate.features],
        "pockets": [list(pocket.exterior.coords) for pocket in result["layout"].pockets],
        "metrics": {key: result[key] for key in ("aesthetic", "removed", "stiffnessKept", "stressIncrease", "utility")},
    }
    ratingsPath.parent.mkdir(parents=True, exist_ok=True)
    with open(ratingsPath, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"ok": True}


@app.post("/api/export")
def export(pick: Pick):
    job = jobs.get(pick.jobId)
    if job is None or not 0 <= pick.index < len(job["results"]):
        raise HTTPException(404, "Generate again first.")
    entry, result = plates[job["plateId"]], job["results"][pick.index]
    plate, params = entry["plate"], job["params"]
    exported = cadExport.buildPocketedSolid(plate, params, result["layout"])
    checks = [{"rule": check.rule, "passed": bool(check.passed)}
              for check in exported.report2d.results + exported.report3d.results]
    if not (exported.report2d.passed and exported.report3d.passed):
        return {"ok": False, "checks": checks}
    name = Path(entry["name"]).stem + f"_pocketed_{chr(65 + pick.index)}.step"
    outPath = Path(tempfile.mkdtemp()) / name
    cadExport.exportStep(exported, outPath)
    exportId = uuid.uuid4().hex[:12]
    exports[exportId] = outPath
    return {"ok": True, "checks": checks, "url": f"/api/download/{exportId}", "name": name}


@app.get("/api/download/{exportId}")
def download(exportId: str):
    path = exports.get(exportId)
    if path is None:
        raise HTTPException(404, "That export is gone, export again.")
    return FileResponse(path, filename=path.name, media_type="application/step")


app.mount("/", StaticFiles(directory=webDir, html=True), name="ui")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7861)


if __name__ == "__main__":
    main()
