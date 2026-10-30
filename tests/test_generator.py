"""Generator decoding guarantees that hold even for an untrained model."""

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

from frcpocket import geom2d, validator
from frcpocket.generator import RibGenerator, RibGnn, connectFloatingFeatures, decode, encodeGraph, mirrorMap
from frcpocket.layout import candidateGraph
from frcpocket.plate import Plate, PocketParams

params = PocketParams()


def symmetricPlate():
    holes = [Point(x, y).buffer(0.1005, quad_segs=16) for x in (0.5, 7.5) for y in (0.5, 3.5)]
    holes += [Point(x, 2.0).buffer(0.5625, quad_segs=32) for x in (2.0, 6.0)]
    return Plate(outline=box(0, 0, 8, 4), features=holes, thicknessIn=0.25)


def test_encoding_shapes():
    plate = symmetricPlate()
    graph = candidateGraph(plate, params)
    tensors = encodeGraph(plate, params, graph)
    keep, width = RibGnn()(tensors)
    assert keep.shape == (len(graph.edges),)
    assert width.shape == (len(graph.edges), 3)


def test_symmetric_plate_gives_symmetric_layout():
    torch.manual_seed(0)
    plate = symmetricPlate()
    generator = RibGenerator(model=RibGnn())
    graph, tensors, edgeMirror = generator.prepare(plate, params)
    assert edgeMirror, "a mirror-symmetric plate should map most ribs onto mirror partners"
    layout = generator.propose(plate, params, count=1, prepared=(graph, tensors, edgeMirror))[0]
    pockets = unary_union(layout.pockets)
    assert geom2d.symmetryScore(plate.outline, layout.pockets) > 0.95 or pockets.is_empty


def test_no_feature_left_floating():
    plate = symmetricPlate()
    graph = candidateGraph(plate, params)
    nothing = torch.full((len(graph.edges),), -10.0)  # model wants no ribs at all
    layout = decode(plate, params, graph, nothing, torch.zeros(len(graph.edges), 3), greedy=True)
    report = validator.validateLayout(plate, params, layout.pockets)
    connected = [result for result in report.results if result.rule == "2 connected"][0]
    assert connected.passed, report.summary()


def test_proposals_are_mostly_valid():
    torch.manual_seed(1)
    plate = symmetricPlate()
    layouts = RibGenerator(model=RibGnn()).propose(plate, params, count=6, seed=3)
    valid = [layout for layout in layouts if validator.validateLayout(plate, params, layout.pockets).passed]
    assert len(valid) >= 4


def test_rib_style_caps_ribs_per_hole():
    # A model that wants every rib, decoded with at most 3 ribs per small hole and 40 deg
    # between ribs: no bolt hole ends up with more (connecting a floating hole may add one).
    plate = symmetricPlate()
    styled = PocketParams(minRibAngleDegrees=40.0, maxHoleRibs=3)
    generator = RibGenerator(model=RibGnn())
    graph, _, edgeMirror = generator.prepare(plate, styled)
    everything = torch.full((len(graph.edges),), 10.0)
    layout = decode(plate, styled, graph, everything, torch.zeros(len(graph.edges), 3), greedy=True, edgeMirror=edgeMirror)
    counts = {}
    for i, j in layout.ribs:
        counts[i] = counts.get(i, 0) + 1
        counts[j] = counts.get(j, 0) + 1
    bolts = [k for k, node in enumerate(graph.nodes) if node.kind == "feature" and node.featureKind == "bolt"]
    assert max(counts.get(k, 0) for k in bolts) <= 4
    assert validator.validateLayout(plate, styled, layout.pockets).passed


def test_fill_keeps_big_slots_but_fills_slivers_and_wall_strips():
    from frcpocket.generator import fillJunkPockets
    from frcpocket.layout import Layout, candidateGraph

    plate = Plate(outline=box(0, 0, 12, 6), features=[Point(6, 3).buffer(0.1005, quad_segs=16)], thicknessIn=0.25)
    cells = [box(x, 1.0, x + 1.6, 2.6) for x in (0.6, 2.6, 4.6)]  # ordinary pockets
    bigSlot = box(7.0, 1.0, 11.4, 1.6)  # long and big: stays
    smallSliver = box(7.0, 2.0, 9.0, 2.3)  # long and tiny: filled
    wallStrip = box(0.4, 5.25, 4.4, 5.6)  # narrow, along the wall: filled
    layout = Layout(graph=candidateGraph(plate, params), ribs=[], pockets=cells + [bigSlot, smallSliver, wallStrip])
    kept = fillJunkPockets(plate, params, layout).pockets
    assert any(pocket.equals(bigSlot) for pocket in kept)
    assert not any(pocket.equals(smallSliver) for pocket in kept)
    assert not any(pocket.equals(wallStrip) for pocket in kept)
    assert sum(any(pocket.equals(cell) for pocket in kept) for cell in cells) == 3


def test_big_gaps_get_a_lattice_and_dense_plates_do_not():
    from frcpocket.layout import defaultGraph, isLatticeEdge

    sparse = Plate(outline=box(0, 0, 14, 7), features=[Point(x, y).buffer(0.1005, quad_segs=16) for x in (0.5, 13.5) for y in (0.5, 6.5)],
                   thicknessIn=0.25)
    graph = defaultGraph(sparse, params)
    assert sum(node.kind == "interior" for node in graph.nodes) >= 6
    lattice = [edge for edge in graph.edges if isLatticeEdge(graph, edge)]
    layout = decode(sparse, params, graph, torch.full((len(graph.edges),), -10.0), torch.zeros(len(graph.edges), 3), greedy=True)
    kept = set(layout.ribs)
    assert lattice and sum(edge in kept for edge in lattice) >= 0.8 * len(lattice)  # the pattern survives decoding
    assert validator.validateLayout(sparse, params, layout.pockets).passed
    # A hole every inch, like the example plates: no gap for a lattice.
    dense = Plate(outline=box(0, 0, 6, 4), features=[Point(x + 0.5, y + 0.5).buffer(0.1005, quad_segs=16)
                                                    for x in range(6) for y in range(4)], thicknessIn=0.25)
    assert not any(node.kind == "interior" for node in defaultGraph(dense, params).nodes)
