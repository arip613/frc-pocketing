"""Polish pass guarantees on synthetic plates: valid output, symmetry, blobs opened up,
facing bosses linked."""

from shapely.geometry import Point, box

from frcpocket import geom2d, polish, validator
from frcpocket.generator import fillJunkPockets
from frcpocket.layout import candidateGraph, defaultGraph, makeLayout
from frcpocket.plate import Plate, PocketParams, featureCenter

params = PocketParams()


def symmetricPlate():
    holes = [Point(x, y).buffer(0.1005, quad_segs=16) for x in (0.5, 7.5) for y in (0.5, 3.5)]
    holes += [Point(x, 2.0).buffer(0.5625, quad_segs=32) for x in (2.0, 6.0)]
    return Plate(outline=box(0, 0, 8, 4), features=holes, thicknessIn=0.25)


def twoBorePlate():
    holes = [Point(x, 1.75).buffer(0.5, quad_segs=32) for x in (2.0, 5.0)]
    holes += [Point(x, y).buffer(0.1, quad_segs=16) for x in (0.4, 6.6) for y in (0.4, 3.1)]
    return Plate(outline=box(0, 0, 7, 3.5), features=holes, thicknessIn=0.25)


def isValid(plate, layout):
    return bool(layout.pockets) and validator.validateLayout(plate, params, layout.pockets).passed


def wallSpokes(graph):
    """Only bore-to-wall ribs: the two bores face each other across one open pocket."""
    return [edge for edge in graph.edges
            if {graph.nodes[i].kind for i in edge} == {"feature", "anchor"}
            and any(graph.nodes[i].featureKind == "bearing" for i in edge)]


def test_facing_bosses_get_linked():
    plate = twoBorePlate()
    graph = defaultGraph(plate, params)
    layout = makeLayout(plate, params, graph, wallSpokes(graph))
    assert isValid(plate, layout)
    result = polish.polishLayout(plate, params, layout)
    assert isValid(plate, result.layout), result.log
    bores = [featureCenter(plate.features[k]) for k in (0, 1)]
    linked = [rib for rib in result.layout.ribs
              if all(min(Point(result.layout.graph.nodes[i].point).distance(bore) for bore in bores) < 1e-6 for i in rib)]
    assert linked, result.log


def test_symmetric_plate_gives_symmetric_design():
    plate = symmetricPlate()
    graph = defaultGraph(plate, params)
    # Lopsided on purpose: a full triangulation on the left, bore spokes on the right.
    ribs = [edge for k, edge in enumerate(graph.edges)
            if graph.isDelaunay[k] and max(graph.nodes[edge[0]].x, graph.nodes[edge[1]].x) < 3.5]
    ribs += [edge for edge in wallSpokes(graph) if min(graph.nodes[edge[0]].x, graph.nodes[edge[1]].x) > 4.0]
    layout = makeLayout(plate, params, graph, ribs)
    assert isValid(plate, layout)
    result = polish.polishLayout(plate, params, layout)
    assert isValid(plate, result.layout), result.log
    shortfall = polish.Polisher(plate, params, result.layout).symmetryShortfall(result.layout.pockets)
    assert shortfall < 0.03, (shortfall, result.log)
    assert geom2d.symmetryScore(plate.outline, result.layout.pockets) > 0.95


def test_large_solid_region_gets_pocketed():
    plate = Plate(outline=box(0, 0, 8, 4), features=[Point(x, y).buffer(0.1, quad_segs=16) for x in (0.5, 7.5) for y in (0.5, 3.5)],
                  thicknessIn=0.25)
    graph = candidateGraph(plate, params)
    # A tangle of ribs at both ends leaves only crumbs there, which are left solid: two blobs.
    ribs = [edge for k, edge in enumerate(graph.edges) if graph.isDelaunay[k]
            and (max(graph.nodes[edge[0]].x, graph.nodes[edge[1]].x) < 3.2
                 or min(graph.nodes[edge[0]].x, graph.nodes[edge[1]].x) > 4.8)]
    layout = fillJunkPockets(plate, params, makeLayout(plate, params, graph, ribs))
    assert isValid(plate, layout)
    checker = polish.Polisher(plate, params, layout)
    before = checker.blobArea(polish.Design(layout.graph, layout.ribs, layout.ribWidths), layout)
    assert before > 0.2 * plate.materialArea
    result = polish.polishLayout(plate, params, layout)
    assert isValid(plate, result.layout), result.log
    after = checker.blobArea(polish.Design(result.layout.graph, result.layout.ribs, result.layout.ribWidths), result.layout)
    assert after < 0.25 * before, (before, after, result.log)
    # Pocketed with pockets near the plate's usual size, not one giant one.
    assert max(pocket.area for pocket in result.layout.pockets) < 0.15 * plate.materialArea


def test_validator_holds_and_failed_steps_revert():
    plate = symmetricPlate()
    graph = defaultGraph(plate, params)
    layout = makeLayout(plate, params, graph, wallSpokes(graph))
    assert isValid(plate, layout)
    result = polish.polishLayout(plate, params, layout)
    assert isValid(plate, result.layout), result.log
    # A structural floor that rejects every change: nothing may be kept.
    refused = polish.polishLayout(plate, params, layout, structuralCheck=lambda pockets: False)
    assert not refused.changed and refused.layout is layout
    assert all("reverted" in line for line in refused.log)
