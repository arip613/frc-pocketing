"""End to end: STEP in -> baseline layout -> exact pockets cut from the original -> STEP out."""

import cadquery as cq
import pytest

from frcpocket import cadExport, testPlates
from frcpocket.baseline import trussBaseline
from frcpocket.plate import PocketParams, loadPlateFromStep

params = PocketParams(wallIn=0.1875, ribIn=0.125, cornerRadiusIn=0.0625)


@pytest.fixture(scope="module")
def platePaths(tmp_path_factory):
    return testPlates.writeTestPlates(tmp_path_factory.mktemp("plates"))


@pytest.mark.parametrize("name", list(testPlates.testPlateBuilders))
def test_through_pockets_export_single_valid_solid(platePaths, tmp_path, name):
    plate = loadPlateFromStep(platePaths[name])
    layout = trussBaseline(plate, params)
    result = cadExport.buildPocketedSolid(plate, params, layout)
    assert result.report2d.passed, result.report2d.summary()
    assert result.report3d.passed, result.report3d.summary()
    outPath = cadExport.exportStep(result, tmp_path / f"{name}.step")
    reloaded = cq.importers.importStep(str(outPath)).val()
    assert len(reloaded.Solids()) == 1
    assert reloaded.isValid()
    assert reloaded.Volume() == pytest.approx(plate.cad.solid.Volume() - result.removedVolumeMm3, rel=1e-4)


def test_blind_pockets_leave_a_floor(platePaths, tmp_path):
    plate = loadPlateFromStep(platePaths["bearingPlate"])  # 0.25 in thick
    blind = PocketParams(wallIn=0.1875, ribIn=0.125, cornerRadiusIn=0.0625, depthIn=0.15)
    layout = trussBaseline(plate, blind)
    result = cadExport.buildPocketedSolid(plate, blind, layout)
    assert result.report2d.passed and result.report3d.passed, result.report3d.summary()
    pocketArea = sum(face.Area() for face in result.pocketFaces)
    assert result.removedVolumeMm3 == pytest.approx(pocketArea * 0.15 * 25.4)
    # The floor is still there: the plate's footprint at the bottom face is unchanged.
    assert plate.cad.solid.Volume() - result.solid.Volume() < pocketArea * plate.cad.thicknessMm


def test_millimetre_parameters_convert():
    metric = PocketParams.fromUnits("mm", wall=4.7625, rib=3.175, cornerRadius=1.5875, depth=3.81)
    assert metric.wallIn == pytest.approx(0.1875)
    assert metric.ribIn == pytest.approx(0.125)
    assert metric.cornerRadiusIn == pytest.approx(0.0625)
    assert metric.depthIn == pytest.approx(0.15)


def test_refuses_to_export_invalid_design(platePaths, tmp_path):
    plate = loadPlateFromStep(platePaths["gusset"])
    layout = trussBaseline(plate, params)
    layout.ribs = []  # no ribs: every boss floats in one giant pocket
    from frcpocket.layout import buildPockets

    layout.pockets = buildPockets(plate, params, layout.graph, layout.ribs)
    result = cadExport.buildPocketedSolid(plate, params, layout)
    assert not (result.report2d.passed and result.report3d.passed)
    with pytest.raises(ValueError, match="refusing to export"):
        cadExport.exportStep(result, tmp_path / "bad.step")
