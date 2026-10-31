"""Relaxed truss engine: rib networks with freely placed junctions.

The ML generator picks ribs between fixed points (hole centres and set wall points), so
designs whose ribs meet between holes, like most of the plates the user kept, can't be
drawn. Here the junctions are free: holes stay fixed, wall points slide along the wall
centre line, and interior junctions are spread through the pocketable area and relaxed
(Lloyd iterations) until every node holds an even share of it. A Delaunay triangulation of
the relaxed points is an even truss: triangles near one size, corners near 60 degrees.
Holes on the outer wall are part of the wall and get no node of their own.

    TrussEngine().propose(plate, params, count=4)  # same interface as RibGenerator.propose

Pockets are built by layout.makeLayout from the rib segments, like every other design.
"""

import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from frcpocket import designFeatures
from frcpocket.layout import CandidateGraph, Node, arcSegments, makeLayout, polygons, wallCenterline
from frcpocket.plate import featureCenter

spacingFactors = (0.8, 1.0, 1.25, 1.6)  # node spacing in target pocket widths, one design each
lloydIterations = 25
sampleCount = 6000  # area samples for the relaxation
wallReachFactor = 1.0  # a hole whose boss is within this many rib widths of the wall band is on the wall
narrowFactor = 1.15  # on a narrow stretch, wall points are this many local widths apart (equilateral zigzag)


def triangleSide(area):
    """Side of the equilateral triangle with this area."""
    return math.sqrt(4 * area / math.sqrt(3))


class TrussEngine:
    """Proposes relaxed-truss layouts; drop-in for RibGenerator in Pocketer.generators."""

    def propose(self, plate, params, count=4, seed=0, **_):
        # Pocketer asks once per rib style; a different seed per style gives different relaxations.
        seed = seed + int(params.minRibAngleDegrees) * 7 + int(params.maxHoleRibs or 0)
        layouts = []
        for factor in spacingFactors[: max(1, count)]:
            layout = relaxedTruss(plate, params, factor, seed)
            if layout is not None and layout.pockets:
                layouts.append(layout)
        return layouts


def relaxedTruss(plate, params, spacingFactor=1.0, seed=0):
    if not wallCenterline(plate, params):
        return None
    scale = math.sqrt(plate.materialArea)
    target = designFeatures.referencePocketArea(scale) * params.pocketSizeFactor
    spacing = spacingFactor * (triangleSide(target) + params.ribIn)
    inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
    bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    reach = inner.buffer(-wallReachFactor * params.ribIn)
    onWall = [not reach.contains(boss) for boss in bosses]
    region = inner.difference(unary_union(bosses)) if bosses else inner
    if region.is_empty or region.area < 0.5 * target:
        return None

    # Fixed nodes: holes off the wall. Wall nodes: evenly spaced on the wall centre line,
    # leaving out stretches a boss already covers.
    nodes = [Node(*featureCenter(feature).coords[0], kind="feature", featureIndex=index)
             for index, feature in enumerate(plate.features) if not onWall[index]]
    fixedCount = len(nodes)
    # Wall points go on the edge of the pocketable region: the inside of the wall, and the
    # inside of any band that bolt bosses along the wall form (a bolt row then sits in a solid
    # band with the pockets inside it, as on the Spectrum rail).
    edges_ = [part.exterior for part in polygons(region) if part.area > 0.25 * target]
    wallSlots = {}  # node index -> (ring index, position along it)
    for ringIndex, ring in enumerate(edges_):
        # Walk the ring; where the plate is narrow, wall points sit about one local width
        # apart (a zigzag truss across an arm), elsewhere one spacing apart.
        position = 0.0
        while position < ring.length - 1e-9:
            point = ring.interpolate(position)
            width = localWidth(plate, edges_, ring, position)
            step = max(2.5 * params.ribIn, min(spacing, narrowFactor * width))
            wallSlots[len(nodes)] = (ringIndex, position)
            nodes.append(Node(point.x, point.y, kind="anchor"))
            position += step

    # Free junctions: random points in the region, then Lloyd relaxation.
    rng = np.random.default_rng(seed)
    preparedRegion = prep(region)
    minx, miny, maxx, maxy = region.bounds
    want = max(0, int(round(region.area / (math.sqrt(3) / 4 * spacing ** 2) / 2)))
    samples = sampleIn(preparedRegion, region.bounds, sampleCount, rng)
    free = sampleIn(preparedRegion, region.bounds, want, rng) if want else np.zeros((0, 2))
    fixed = np.array([node.point for node in nodes]) if nodes else np.zeros((0, 2))
    for _ in range(lloydIterations):
        if not len(free):
            break
        points = np.vstack([fixed, free])
        owner = cKDTree(points).query(samples)[1]
        moved = free.copy()
        for k in range(len(free)):
            mine = samples[owner == len(fixed) + k]
            if len(mine):
                moved[k] = mine.mean(axis=0)
        free = moved
    # Drop junctions crowding a fixed node (closer than half a spacing).
    if len(free) and len(fixed):
        distance = cKDTree(fixed).query(free)[0]
        free = free[distance > 0.5 * spacing]
    nodes += [Node(x, y, kind="interior") for x, y in free]

    points = np.array([node.point for node in nodes])
    if len(points) < 3:
        return None
    triangulation = Delaunay(points)
    edges = set()
    for simplex in triangulation.simplices:
        for a, b in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])):
            edges.add((int(min(a, b)), int(max(a, b))))
    alongWall = set()
    byRing = {}
    for index, (ringIndex, position) in wallSlots.items():
        byRing.setdefault(ringIndex, []).append((position, index))
    for slots in byRing.values():
        slots.sort()
        for (_, a), (_, b) in zip(slots, slots[1:] + slots[:1]):
            alongWall.add((min(a, b), max(a, b)))
    insideOutline = prep(plate.outline.buffer(1e-3))  # wall points sit on the wall centre line, inside the outline
    preparedHoles = [prep(feature) for feature in plate.features]
    kept = []
    for i, j in sorted(edges):
        segment = LineString([nodes[i].point, nodes[j].point])
        if segment.length < 1e-6 or not insideOutline.contains(segment):
            continue
        if (i, j) in alongWall:
            continue  # between neighbouring wall points: the wall is already solid there
        ends = {nodes[i].featureIndex, nodes[j].featureIndex} - {None}
        if any(index not in ends and preparedHoles[index].intersects(segment) for index in range(len(plate.features))):
            continue
        kept.append((i, j))
    graph = CandidateGraph(nodes=nodes, edges=kept, isDelaunay=[True] * len(kept))
    return makeLayout(plate, params, graph, kept, name=f"truss{spacingFactor:g}")


def sampleIn(prepared, bounds, count, rng, batch=4000):
    minx, miny, maxx, maxy = bounds
    out = []
    while len(out) < count:
        candidates = np.column_stack([rng.uniform(minx, maxx, batch), rng.uniform(miny, maxy, batch)])
        out += [point for point in candidates if prepared.contains(Point(point))]
        if not out and len(out) < count and batch > 10 ** 6:
            break
        batch = min(batch * 2, 200000)
    return np.array(out[:count]) if out else np.zeros((0, 2))


def localWidth(plate, rings, ring, position):
    """Wall-to-wall width straight inward from a point of the wall centre line (infinite
    where the ray finds no other wall)."""
    length = ring.length
    point = ring.interpolate(position % length)
    ahead, behind = ring.interpolate((position + 0.01 * length) % length), ring.interpolate((position - 0.01 * length) % length)
    tangent = np.array([ahead.x - behind.x, ahead.y - behind.y])
    tangent /= max(np.hypot(*tangent), 1e-12)
    reach = math.dist(*np.reshape(plate.outline.bounds, (2, 2)))
    best = math.inf
    for normal in (np.array([-tangent[1], tangent[0]]), np.array([tangent[1], -tangent[0]])):
        if not plate.outline.contains(Point(point.x + 0.02 * normal[0], point.y + 0.02 * normal[1])):
            continue
        ray = LineString([(point.x + 1e-3 * normal[0], point.y + 1e-3 * normal[1]),
                          (point.x + reach * normal[0], point.y + reach * normal[1])])
        for other in rings:
            hits = ray.intersection(other)
            for hit in getattr(hits, "geoms", [hits]):
                if hit.geom_type == "Point" and hit.distance(point) > 1e-2:
                    best = min(best, hit.distance(point))
    return best
