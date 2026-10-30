"""Scorer pieces that don't need trained weights."""

import numpy as np
from shapely.geometry import Point, Polygon, box

from frcpocket.designFeatures import sharpestCorner
from frcpocket.scorer import PlateSet, pairsOf


def test_sharpest_corner_sees_through_fillets_and_bosses():
    assert abs(sharpestCorner(box(0, 0, 1, 1).buffer(-0.15).buffer(0.15)) - 90) < 1
    wedge = Polygon([(0, 0), (4, 0), (0, 1)]).buffer(-0.1).buffer(0.1)
    assert abs(sharpestCorner(wedge) - 14.0) < 1
    cutByBoss = Polygon([(0, 0), (4, -1), (4, 1)]).difference(Point(0, 0).buffer(0.8))
    assert sharpestCorner(cutByBoss) < 35
    assert sharpestCorner(Point(0, 0).buffer(1)) == 180.0


def test_graded_pairs_follow_grades():
    graded = PlateSet("p", np.zeros((3, 1)), np.zeros((3, 1, 2, 2)), np.array([True, False, False]),
                      ["a", "b", "c"], weight=2.0, grades=np.array([5.0, 3.0, 2.5]))
    binary = PlateSet("q", np.zeros((2, 1)), np.zeros((2, 1, 2, 2)), np.array([True, False]), ["x", "y"], weight=1.0)
    pairs = pairsOf([graded, binary])
    gradedPairs = {(better, worse): weight for plate, better, worse, weight in pairs if plate == 0}
    assert gradedPairs == {(0, 1): 2.0, (0, 2): 2.5}  # 3.0 vs 2.5 is under a grade apart
    assert [(better, worse) for plate, better, worse, _ in pairs if plate == 1] == [(0, 1)]
