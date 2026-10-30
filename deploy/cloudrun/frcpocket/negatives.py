"""Deliberately bad (or merely mediocre) layouts on a given plate, for scorer training.

Every negative shares its plate's outline and features with the positive it is
compared against, so the only thing the scorer can learn from is the pocket pattern.
These mirror the "what bad pocketing looks like" list in docs/designNotes.md.
"""

import math

import numpy as np
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

from frcpocket import geom2d
from frcpocket.baseline import trussBaseline
from frcpocket.layout import arcSegments, buildPockets, finishRegions, makeLayout, polygons
from frcpocket.projection import dropCrossings


def pocketRegion(plate, params):
    inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
    bosses = [feature.buffer(params.boss, quad_segs=arcSegments) for feature in plate.features]
    return inner.difference(unary_union(bosses)) if bosses else inner


def finishPockets(plate, params, shapes):
    """Same soft-fillet finishing as real layouts, so fillets can't tell good from bad."""
    region = pocketRegion(plate, params)
    cut = unary_union(shapes).intersection(region)
    return finishRegions(polygons(cut), params)


def randomRibs(plate, params, graph, rng, keepFraction):
    chosen = [edge for edge in graph.edges if rng.random() < keepFraction]
    chosen = dropCrossings(graph, chosen, {edge: rng.random() for edge in chosen})
    return buildPockets(plate, params, graph, chosen)


def dropRibs(plate, params, layout, rng, fraction):
    kept = [edge for edge in layout.ribs if rng.random() > fraction]
    return buildPockets(plate, params, layout.graph, kept, layout.ribWidths)


def addCrossingRibs(plate, params, layout, rng, count):
    extras = [edge for edge in layout.graph.edges if edge not in layout.ribs]
    rng.shuffle(extras)
    return buildPockets(plate, params, layout.graph, list(layout.ribs) + extras[:count], layout.ribWidths)


def asymmetrize(plate, params, layout, rng):
    centre = plate.outline.centroid
    angle = math.radians(geom2d.principalAngle(plate.outline))
    axis = np.array([math.cos(angle), math.sin(angle)])
    side = rng.choice([-1, 1])
    kept = []
    for edge in layout.ribs:
        middle = np.array(layout.graph.segment(edge).interpolate(0.5, normalized=True).coords[0]) - np.array(centre.coords[0])
        if np.sign(middle @ axis) == side and rng.random() < 0.6:
            continue
        kept.append(edge)
    return buildPockets(plate, params, layout.graph, kept, layout.ribWidths)


def saltAndPepper(plate, params, rng, count):
    region = pocketRegion(plate, params)
    minx, miny, maxx, maxy = region.bounds
    holes = []
    for _ in range(count * 4):
        if len(holes) >= count:
            break
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        radius = rng.uniform(params.minPocketWidth / 2, max(params.minPocketWidth, 0.06 * math.sqrt(plate.outline.area)))
        disk = point.buffer(radius, quad_segs=8)
        if region.contains(disk) and all(disk.distance(other) >= params.ribIn for other in holes):
            holes.append(affinity.scale(disk, xfact=rng.uniform(0.6, 1.6), origin=point))
    return finishPockets(plate, params, holes)


def gridWindows(plate, params, rng):
    region = pocketRegion(plate, params)
    minx, miny, maxx, maxy = region.bounds
    columns, rows = rng.integers(2, 5), rng.integers(1, 4)
    cellWidth, cellHeight = (maxx - minx) / columns, (maxy - miny) / rows
    rib = params.ribIn * rng.uniform(1.0, 2.0)
    windows = [
        box(minx + c * cellWidth + rib / 2, miny + r * cellHeight + rib / 2,
            minx + (c + 1) * cellWidth - rib / 2, miny + (r + 1) * cellHeight - rib / 2)
        for c in range(columns) for r in range(rows)
    ]
    return finishPockets(plate, params, windows)


def gutted(plate, params, graph, rng, keepFraction=0.0):
    """Wall and bosses with only the ribs needed to stop holes floating: the plate is gutted."""
    import torch

    from frcpocket.generator import connectFloatingFeatures

    chosen = [edge for edge in graph.edges if rng.random() < keepFraction]
    probabilities = torch.full((len(graph.edges),), 0.5).numpy()
    chosen = connectFloatingFeatures(plate, params, graph, chosen, probabilities)
    return buildPockets(plate, params, graph, sorted(chosen))


def ladder(plate, params, graph, rng):
    """Parallel ribs across the plate that leave long slots, the gusset failure mode."""
    angle = math.radians(geom2d.principalAngle(plate.outline) + (0 if rng.random() < 0.5 else 90))
    axis = np.array([math.cos(angle), math.sin(angle)])
    chosen = []
    for edge in graph.edges:
        a, b = graph.nodes[edge[0]], graph.nodes[edge[1]]
        direction = np.array(b.point) - np.array(a.point)
        length = float(np.hypot(*direction))
        if length > 0 and abs(direction @ axis) / length > 0.97:
            chosen.append(edge)
    return buildPockets(plate, params, graph, dropCrossings(graph, chosen, {edge: rng.random() for edge in chosen}))


def negativeSet(plate, params, graph, projected, rng, perKind=3):
    """Named lists of pocket polygons, all on this plate."""
    negatives = []
    for _ in range(perKind):
        negatives.append(("randomSparse", randomRibs(plate, params, graph, rng, rng.uniform(0.15, 0.4))))
        negatives.append(("randomDense", randomRibs(plate, params, graph, rng, rng.uniform(0.6, 0.95))))
        negatives.append(("saltAndPepper", saltAndPepper(plate, params, rng, int(rng.integers(6, 30)))))
        negatives.append(("gridWindows", gridWindows(plate, params, rng)))
        negatives.append(("gutted", gutted(plate, params, graph, rng, keepFraction=rng.uniform(0.0, 0.08))))
        negatives.append(("ladder", ladder(plate, params, graph, rng)))
        if projected is not None and projected.ribs:
            negatives.append(("dropRibs", dropRibs(plate, params, projected, rng, rng.uniform(0.3, 0.7))))
            negatives.append(("crossingRibs", addCrossingRibs(plate, params, projected, rng, int(rng.integers(3, 10)))))
            negatives.append(("asymmetric", asymmetrize(plate, params, projected, rng)))
    negatives.append(("triangulation", trussBaseline(plate, params).pockets))
    everything = [edge for edge, planar in zip(graph.edges, graph.isDelaunay) if planar]
    negatives.append(("allCandidates", buildPockets(plate, params, graph, everything)))
    return [(name, pockets) for name, pockets in negatives if pockets]
