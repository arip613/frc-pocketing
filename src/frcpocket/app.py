"""Local UI: upload a flat plate STEP, generate pocketing candidates, rate them, export STEP.

    python -I -m frcpocket.app        then open http://127.0.0.1:7860
"""

import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frcpocket import cadExport, render
from frcpocket.plate import PocketParams, loadPlateFromStep
from frcpocket.pocketer import Pocketer

ratingsPath = Path("data/curation/ratings.jsonl")
unitDefaults = {
    "in": {"wall": 0.1875, "rib": 0.1875, "corner": 0.0625, "depth": 0.125, "boss": 0.1875},
    "mm": {"wall": 4.8, "rib": 4.8, "corner": 1.6, "depth": 3.2, "boss": 4.8},
}
pocketer = None
sessions = {}  # session key -> plate, params, candidates (CAD objects stay server-side)


def getPocketer():
    global pocketer
    if pocketer is None:
        pocketer = Pocketer()
    return pocketer


def previewImage(plate, result, index):
    figure, axis = plt.subplots(figsize=(5, 4), dpi=110)
    render.drawPlate(
        axis,
        list(plate.outline.exterior.coords),
        [list(feature.exterior.coords) for feature in plate.features],
        [list(pocket.exterior.coords) for pocket in result["layout"].pockets],
    )
    path = Path(tempfile.mkdtemp()) / f"candidate{index}.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def caption(index, result):
    if result["stiffnessKept"] is None:
        structure = "structure n/a (FEA failed)"
    else:
        structure = f"{result['stiffnessKept']:.0%} stiffness kept, stress x{result['stressIncrease']:.1f}"
    warning = "" if result.get("sound", True) else " ⚠ structurally weak"
    return f"#{index + 1} {result['source']}{warning}: looks {result['aesthetic']:.2f}, {result['removed']:.0%} removed, {structure}"


densityFactors = {"Dense (more ribs)": 0.8, "Medium": 1.25, "Open (fewer ribs)": 1.6}


def buildParams(units, wall, rib, corner, depthMode, depth, density="Medium", boss=None):
    return PocketParams.fromUnits(units, wall=wall, rib=rib, cornerRadius=corner,
                                  depth=None if depthMode == "Through" else depth, boss=boss,
                                  pocketSizeFactor=densityFactors[density])


def generate(stepFile, units, wall, rib, corner, boss, depthMode, depth, density, count):
    if stepFile is None:
        raise gr.Error("Upload a STEP file first.")
    started = time.time()
    try:
        plate = loadPlateFromStep(stepFile)
    except ValueError as error:
        raise gr.Error(str(error))
    if not boss or boss <= 0:
        raise gr.Error("Boss ring thickness must be above zero.")
    params = buildParams(units, wall, rib, corner, depthMode, depth, density, boss)
    if params.depthIn is not None and params.depthIn >= plate.thicknessIn:
        raise gr.Error("Blind depth must be less than the plate thickness (use Through instead).")
    results = getPocketer().candidates(plate, params, count=int(count))
    if not results:
        raise gr.Error("No valid pocketing fits this plate with these settings. Try a smaller wall, rib, or corner radius.")
    gallery = [(previewImage(plate, result, index), caption(index, result)) for index, result in enumerate(results)]
    scale = 25.4 if units == "mm" else 1.0
    status = (f"Plate: {plate.outline.area * scale**2:.1f} {units}² outline, {plate.thicknessIn * scale:.3g} {units} thick, "
              f"{len(plate.features)} holes/features. {len(results)} valid candidates in {time.time() - started:.0f}s. "
              "Stiffness and stress come from a 2D FEA relative to the unpocketed plate (comparative, not certified).")
    key = uuid.uuid4().hex
    sessions[key] = {"plate": plate, "params": params, "results": results, "stepName": Path(stepFile).name}
    return gallery, key, None, status, None


def selectCandidate(evt: gr.SelectData):
    return evt.index


def rate(key, selected, rating):
    state = sessions.get(key)
    if not state or selected is None:
        raise gr.Error("Generate candidates and click one first.")
    plate, params, result = state["plate"], state["params"], state["results"][selected]
    plateHash = hashlib.sha1(json.dumps([list(plate.outline.exterior.coords),
                                         [list(f.exterior.coords) for f in plate.features]]).encode()).hexdigest()[:12]
    entry = {
        "kind": "uiCandidate", "rating": rating, "rater": "user",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plateHash": plateHash, "stepName": state["stepName"], "source": result["source"],
        "params": vars(params), "thicknessIn": plate.thicknessIn,
        "outline": list(plate.outline.exterior.coords),
        "holes": [list(feature.exterior.coords) for feature in plate.features],
        "pockets": [list(pocket.exterior.coords) for pocket in result["layout"].pockets],
        "metrics": {key: result[key] for key in ("aesthetic", "removed", "stiffnessKept", "stressIncrease", "utility")},
    }
    ratingsPath.parent.mkdir(parents=True, exist_ok=True)
    with open(ratingsPath, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return f"Saved {'👍' if rating > 0 else '👎'} for candidate #{selected + 1}. Ratings retrain the aesthetic scorer: `python -I -m frcpocket.scorer`."


def export(key, selected):
    state = sessions.get(key)
    if not state or selected is None:
        raise gr.Error("Generate candidates and click one first.")
    plate, params, result = state["plate"], state["params"], state["results"][selected]
    exported = cadExport.buildPocketedSolid(plate, params, result["layout"])
    checks = exported.report2d.results + exported.report3d.results
    summary = "\n".join(f"{'✅' if check.passed else '❌'} {check.rule}" for check in checks)
    if not (exported.report2d.passed and exported.report3d.passed):
        return None, "Export blocked, the exact geometry failed a check:\n" + summary
    outPath = Path(tempfile.mkdtemp()) / (Path(state["stepName"]).stem + f"_pocketed_{selected + 1}.step")
    cadExport.exportStep(exported, outPath)
    removedPercent = exported.removedVolumeMm3 / plate.cad.solid.Volume()
    return str(outPath), f"Exported candidate #{selected + 1} ({removedPercent:.0%} of the volume removed). All checks:\n" + summary


def switchUnits(units):
    values = unitDefaults[units]
    return values["wall"], values["rib"], values["corner"], values["boss"], values["depth"]


def buildApp():
    with gr.Blocks(title="FRC Plate Pocketing") as app:
        gr.Markdown("## FRC plate pocketing\nUpload a flat plate STEP, set the rules, pick a candidate, download the pocketed STEP.")
        state = gr.State(None)
        selected = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                stepFile = gr.File(label="Plate STEP", file_types=[".step", ".stp"], type="filepath")
                units = gr.Radio(["in", "mm"], value="in", label="Units")
                wall = gr.Number(value=0.1875, label="Outer wall thickness")
                rib = gr.Number(value=0.1875, label="Rib thickness")
                corner = gr.Number(value=0.0625, label="Pocket corner radius (tool radius)")
                boss = gr.Number(value=0.1875, label="Boss ring around holes (default: the wall; real plates often use about half)")
                depthMode = gr.Radio(["Through", "Blind"], value="Through", label="Pocket depth")
                depth = gr.Number(value=0.125, label="Blind depth (leaves a floor)")
                density = gr.Radio(list(densityFactors), value="Medium", label="Pocket density")
                count = gr.Slider(2, 8, value=6, step=1, label="Candidates")
                generateButton = gr.Button("Generate candidates", variant="primary")
            with gr.Column(scale=2):
                status = gr.Markdown()
                gallery = gr.Gallery(label="Candidates (click one to select)", columns=3, height=520, allow_preview=True)
                with gr.Row():
                    upButton = gr.Button("👍 Looks good")
                    downButton = gr.Button("👎 Looks bad")
                    exportButton = gr.Button("Export selected to STEP", variant="primary")
                feedback = gr.Markdown()
                download = gr.File(label="Pocketed STEP")
        units.change(switchUnits, units, [wall, rib, corner, boss, depth])
        generateButton.click(generate, [stepFile, units, wall, rib, corner, boss, depthMode, depth, density, count],
                             [gallery, state, selected, status, download])
        gallery.select(selectCandidate, None, selected)
        upButton.click(lambda s, i: rate(s, i, 1), [state, selected], feedback)
        downButton.click(lambda s, i: rate(s, i, -1), [state, selected], feedback)
        exportButton.click(export, [state, selected], [download, feedback])
    return app


def main():
    buildApp().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
