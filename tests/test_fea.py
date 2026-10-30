import pytest
from shapely.geometry import LineString, Point, box
from skfem import Basis, ElementTriP1, ElementVector, asm

from frcpocket import fea
from frcpocket.baseline import trussBaseline
from frcpocket.plate import Plate, PocketParams


def test_cantilever_matches_beam_theory():
    length, depth, thickness, force = 10.0, 1.0, 0.25, 100.0
    plate = Plate(outline=box(0, 0, length, depth), features=[], thicknessIn=thickness)
    case = fea.LoadCase("tip", [LineString([(0, 0), (0, depth)])],
                        [(LineString([(length, 0), (length, depth)]), (0.0, -force), "edge")])
    mesh = fea.meshPolygon(plate.material, 0.05)
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    stiffness = asm(fea.planeStressForm(thickness)[0], basis)
    result = fea.solveCase(mesh, basis, stiffness, case, thickness, 0.0125)
    inertia = thickness * depth**3 / 12
    shearModulus = fea.youngsModulusPsi / (2 * (1 + fea.poissonRatio))
    expected = force * length**3 / (3 * fea.youngsModulusPsi * inertia) + force * length / (5 / 6 * shearModulus * thickness * depth)
    assert result.maxDisplacementIn == pytest.approx(expected, rel=0.02)


def bearingBracket():
    bolts = [Point(0.4, 0.4).buffer(0.1005, quad_segs=16), Point(0.4, 2.6).buffer(0.1005, quad_segs=16)]
    bearing = Point(4.5, 1.5).buffer(0.5625, quad_segs=32)
    return Plate(outline=box(0, 0, 6, 3), features=bolts + [bearing], thicknessIn=0.25)


def test_unpocketed_plate_keeps_full_stiffness():
    plate = bearingBracket()
    report = fea.analyze(plate, [], PocketParams())
    assert report.stiffnessKept == pytest.approx(1.0)
    assert report.stressIncrease == pytest.approx(1.0)
    assert [case.name for case in report.cases] == ["bearings 0 deg", "bearings 60 deg", "bearings 120 deg"]


def test_pockets_cost_stiffness_but_less_than_mass():
    plate = bearingBracket()
    params = PocketParams()
    layout = trussBaseline(plate, params)
    report = fea.analyze(plate, layout.pockets, params)
    assert 0.2 < report.removedFraction < 0.7
    assert 0.0 < report.stiffnessKept < 1.0
    assert report.stressIncrease > 1.0
