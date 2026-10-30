"""Round-trip a known pocketed plate through STEP/DXF and check the extractor reads it back."""

import math

import cadquery as cq
import ezdxf
import pytest
from shapely.geometry import Point, Polygon

from frcpocket import extract, geom2d

plateLength = 6.0
plateWidth = 3.0
plateThickness = 0.25
holeDiameter = 0.201
wallWidth = 0.25
ribWidth = 0.1875
cornerRadius = 0.125


def filletedRect(cx, cy, width, height, radius):
    return (
        cq.Workplane("XY").center(cx, cy).rect(width, height).extrude(1).edges("|Z").fillet(radius)
        .faces("<Z").wires().val()
    )


def knownPlate():
    """6x3 in plate, 4 corner bolt holes, 2 filleted rectangular pockets with a centre rib."""
    plate = cq.Workplane("XY").rect(plateLength, plateWidth).extrude(plateThickness)
    holeOffsets = [(sx * 2.6, sy * 1.1) for sx in (-1, 1) for sy in (-1, 1)]
    plate = plate.faces(">Z").workplane().pushPoints(holeOffsets).hole(holeDiameter)
    pocketHeight = plateWidth - 2 * wallWidth
    # Pockets stop short of the corner holes in x; the rib sits at x=0.
    pocketLength = 2.0
    for cx in (-(ribWidth / 2 + pocketLength / 2), ribWidth / 2 + pocketLength / 2):
        cutter = (
            cq.Workplane("XY").center(cx, 0).rect(pocketLength, pocketHeight).extrude(plateThickness)
            .edges("|Z").fillet(cornerRadius)
        )
        plate = plate.cut(cutter)
    return plate


def test_classifyLoop_basics():
    circle = Point(0, 0).buffer(0.5, 64)
    assert geom2d.classifyLoop({"nLines": 0, "arcRadii": [0.5]}, circle) == "hole"
    hexagon = Polygon([(0.29 * math.cos(math.radians(a)), 0.29 * math.sin(math.radians(a))) for a in range(0, 360, 60)])
    assert geom2d.classifyLoop({"nLines": 6, "arcRadii": []}, hexagon) == "hole"
    triangle = Polygon([(0, 0), (2, 0), (1, 1.5)]).buffer(-0.1).buffer(0.1, 16)
    assert geom2d.classifyLoop({"nLines": 3, "arcRadii": [0.1, 0.1, 0.1]}, triangle) == "pocket"
    sharpWindow = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert geom2d.classifyLoop({"nLines": 4, "arcRadii": []}, sharpWindow) == "hole"


def test_step_roundtrip(tmp_path):
    # Build in inches, export in mm the way real CAD does.
    stepPath = tmp_path / "plate.step"
    cq.exporters.export(knownPlate().val().scale(25.4), str(stepPath))
    records = extract.platesFromStep(stepPath)
    assert len(records) == 1
    record = records[0]
    stats = record["stats"]
    assert record["thicknessIn"] == pytest.approx(plateThickness, abs=1e-3)
    assert stats["holeCount"] == 4
    assert stats["pocketCount"] == 2
    assert min(stats["wallIn"]) == pytest.approx(wallWidth, abs=0.01)
    assert min(stats["ribIn"]) == pytest.approx(ribWidth, abs=0.01)
    radii = [radius for meta in record["pocketMeta"] for radius in meta["arcRadiiIn"]]
    assert min(radii) == pytest.approx(cornerRadius, abs=1e-3)
    assert stats["symmetry"] > 0.95


def test_dxf_roundtrip(tmp_path):
    dxfPath = tmp_path / "plate.dxf"
    document = ezdxf.new()
    document.header["$INSUNITS"] = 1
    modelSpace = document.modelspace()
    modelSpace.add_lwpolyline([(0, 0), (plateLength, 0), (plateLength, plateWidth), (0, plateWidth)], close=True)
    modelSpace.add_circle((0.4, 0.4), holeDiameter / 2)
    # One filleted square pocket drawn with bulged polyline corners.
    bulge = math.tan(math.radians(90) / 4)
    x0, y0, size, radius = 2.0, 0.75, 1.5, cornerRadius
    modelSpace.add_lwpolyline([
        (x0 + radius, y0, 0), (x0 + size - radius, y0, bulge), (x0 + size, y0 + radius, 0),
        (x0 + size, y0 + size - radius, bulge), (x0 + size - radius, y0 + size, 0),
        (x0 + radius, y0 + size, bulge), (x0, y0 + size - radius, 0), (x0, y0 + radius, bulge),
    ], format="xyb", close=True)
    document.saveas(dxfPath)
    records = extract.platesFromDxf(dxfPath)
    assert len(records) == 1
    stats = records[0]["stats"]
    assert stats["holeCount"] == 1
    assert stats["pocketCount"] == 1
    assert min(stats["wallIn"]) == pytest.approx(0.75, abs=0.01)
