"""Each hard rule must catch a design built to break it, and only that rule."""

import pytest
from shapely.geometry import Point, Polygon, box

from frcpocket import validator
from frcpocket.baseline import trussBaseline
from frcpocket.plate import Plate, PocketParams

params = PocketParams(wallIn=0.1875, ribIn=0.125, cornerRadiusIn=0.0625)
radius = params.cornerRadiusIn


def roundedBox(x0, y0, x1, y1, r=radius):
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def platePlain():
    """6 x 3 plate with a bolt hole near each corner and one bearing in the middle."""
    holes = [Point(x, y).buffer(0.1005, quad_segs=16) for x in (0.4, 5.6) for y in (0.4, 2.6)]
    holes.append(Point(3.0, 1.5).buffer(0.5625, quad_segs=16))
    return Plate(outline=box(0, 0, 6, 3), features=holes, thicknessIn=0.25)


def failedRules(pockets, plate=None):
    report = validator.validateLayout(plate or platePlain(), params, pockets)
    return {result.rule for result in report.failures}


goodLeft = roundedBox(0.75, 0.25, 2.2, 2.75)
goodRight = roundedBox(3.8, 0.25, 5.25, 2.75)


def test_good_design_passes():
    assert failedRules([goodLeft, goodRight]) == set()


def test_baseline_passes():
    plate = platePlain()
    layout = trussBaseline(plate, params)
    assert layout.pockets
    assert validator.validateLayout(plate, params, layout.pockets).passed


def test_wall_too_thin():
    # A 0.1 in wall is also a neck thinner than the 0.125 in rib, so both rules fire.
    assert failedRules([roundedBox(0.75, 0.1, 2.2, 2.75), goodRight]) == {"1 wall", "3 no thin necks"}


def test_wall_between_rib_and_wall_thickness():
    # 0.15 in: thicker than a rib but thinner than the required wall.
    assert failedRules([roundedBox(0.75, 0.15, 2.2, 2.75), goodRight]) == {"1 wall"}


def test_slightly_small_fillet_caught():
    smallFillet = roundedBox(0.75, 0.25, 2.2, 2.75, r=0.03)
    assert failedRules([smallFillet, goodRight]) == {"4 corner radius"}


def test_hole_inside_pocket():
    plate = platePlain()
    plate.features.append(Point(1.5, 1.5).buffer(0.1, quad_segs=16))
    assert "2 bosses" in failedRules([goodLeft, goodRight], plate)


def test_floating_boss():
    # A ring-shaped pocket around the bearing cuts its boss off from everything else.
    ring = Point(3.0, 1.5).buffer(1.15, quad_segs=32).difference(Point(3.0, 1.5).buffer(0.75, quad_segs=32))
    assert "2 connected" in failedRules([ring])


def test_rib_too_thin():
    pocketA = roundedBox(0.75, 0.25, 1.7, 2.75)
    pocketB = roundedBox(1.75, 0.25, 2.2, 2.75)  # 0.05 in gap
    assert "3 ribs" in failedRules([pocketA, pocketB, goodRight])


def test_thin_neck_inside_one_pocket():
    # A C-shaped pocket leaves a 0.06 in wide peninsula of material inside it.
    outer = roundedBox(0.75, 0.25, 2.2, 2.75)
    peninsula = box(0.6, 1.47, 1.8, 1.53)
    cShape = outer.difference(peninsula.buffer(0.0)).buffer(-radius, quad_segs=16).buffer(radius, quad_segs=16)
    assert "3 no thin necks" in failedRules([cShape, goodRight])


def test_sharp_corners():
    assert failedRules([box(0.75, 0.25, 2.2, 2.75), goodRight]) == {"4 corner radius"}


def test_tiny_pocket():
    tiny = Point(1.5, 1.5).buffer(0.12, quad_segs=16)
    assert failedRules([tiny, goodRight]) == {"4 pocket size"}


@pytest.mark.parametrize("wall", [0.125, 0.25])
def test_wall_parameter_respected(wall):
    custom = PocketParams(wallIn=wall, ribIn=0.125, cornerRadiusIn=0.0625)
    plate = platePlain()
    layout = trussBaseline(plate, custom)
    report = validator.validateLayout(plate, custom, layout.pockets)
    assert report.passed, report.summary()


def test_boss_ring_is_its_own_setting():
    # A pocket 0.15 in from a hole: too close for the default ring (the 0.1875 in wall),
    # fine once the ring is set to 0.125 in, as on real plates with thin bosses.
    from shapely.geometry import Point, box

    from frcpocket.plate import Plate, PocketParams
    from frcpocket.validator import validateLayout

    hole = Point(3, 2).buffer(0.1, quad_segs=16)
    plate = Plate(outline=box(0, 0, 6, 4), features=[hole], thicknessIn=0.25)
    pocket = box(3.25, 1.0, 5.0, 3.0).buffer(-0.1).buffer(0.1)
    default = validateLayout(plate, PocketParams(), [pocket])
    thinRing = validateLayout(plate, PocketParams(bossIn=0.125), [pocket])
    assert not [r for r in default.results if r.rule == "2 bosses"][0].passed
    assert thinRing.passed, thinRing.summary()
