"""Relaxed truss engine: rib networks with freely placed junctions.

The ML generator picks ribs between fixed points (hole centres and set wall points), so
designs whose ribs meet between holes, like most of the plates the user kept, can't be
drawn. Here the junctions are free: holes stay fixed, wall points slide along the wall
centre line, and interior junctions are spread through the pocketable area and relaxed
(Lloyd iterations) until every node holds an even share of it. A Delaunay triangulation of
the relaxed points is an even truss: triangles near one size, corners near 60 degrees.
Holes on the outer wall are part of the wall and get no node of their own.

    TrussEngine().propose(plate, params, count=4)  # same interface as RibGenerator.propose

A second family, alignedTruss, draws what the user's regular kept plates do: rails along
the plate's main direction (through each row of holes, more in between so the gaps match
the kept pocket size) and rungs between them, alternating diagonals (a row of triangles)
or square across (a row of rectangles).

Pockets are built by layout.makeLayout from the rib segments, like every other design.
"""

import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from frcpocket import designFeatures
from frcpocket.layout import CandidateGraph, Node, arcSegments, makeLayout, polygons, wallCenterline
from frcpocket.plate import featureCenter

# One design per (node spacing in lattice widths, sharpest open-junction angle kept): pure
# triangles, a light clean-up, and mostly 3-way junctions, each at a few densities.
# (Looked at on non-held-out kept plates: 0.8 crowds small plates with crumbs, and 80 degrees
# merges too far, into a few big cells.)
variants = ((1.0, 0.0), (1.3, 0.0), (1.6, 0.0), (0.9, 55.0), (1.1, 55.0), (1.35, 55.0))
# Aligned rails and rungs: (rungs, pitch in kept pocket sizes).
alignedVariants = (("diagonal", 0.8), ("diagonal", 1.1), ("diagonal", 1.5), ("square", 0.8), ("square", 1.2))
holeRowTolerance = 0.2  # holes whose centres are this close across the rails share a rail (in)
lloydIterations = 25
sampleCount = 6000  # area samples for the relaxation
wallReachFactor = 1.0  # a hole whose boss is within this many rib widths of the wall band is on the wall
narrowFactor = 1.15  # on a narrow stretch, wall points are this many local widths apart (equilateral zigzag)
# The plates the user kept (data/curation/plateKeeper.json, measured in docs/designNotes.md):
# median pocket area about 0.165 * S^1.05 in^2 with S = sqrt(material area), at the default
# pocket-size factor; open rib junctions mostly 3-way, their smallest angle about 90 degrees.
keptPocketCoefficient, keptPocketExponent, defaultSizeFactor = 0.165, 1.05, 1.25
mergedPerTriangle = 1.6  # removing acute ribs merges triangles: pockets end up about this many lattice triangles


def triangleSide(area):
    """Side of the equilateral triangle with this area."""
    return math.sqrt(4 * area / math.sqrt(3))


class TrussEngine:
    """Proposes relaxed-truss layouts; drop-in for RibGenerator in Pocketer.generators."""

    def propose(self, plate, params, count=4, seed=0, **_):
        # Pocketer asks once per rib style; a different seed per style gives different relaxations.
        seed = seed + int(params.minRibAngleDegrees) * 7 + int(params.maxHoleRibs or 0)
        layouts = []
        for factor, junctionDegrees in variants:
            layout = relaxedTruss(plate, params, factor, seed, junctionDegrees)
            if layout is not None and layout.pockets:
                layouts.append(layout)
        for pattern, pitch in alignedVariants:
            layout = alignedTruss(plate, params, pattern, pitch)
            if layout is not None and layout.pockets:
                layouts.append(layout)
        return layouts


def relaxedTruss(plate, params, spacingFactor=1.0, seed=0, junctionDegrees=0.0):
    if not wallCenterline(plate, params):
        return None
    scale = math.sqrt(plate.materialArea)
    target = keptPocketCoefficient * scale ** keptPocketExponent * params.pocketSizeFactor / defaultSizeFactor
    spacing = spacingFactor * (triangleSide(target / mergedPerTriangle) + params.ribIn)
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
    if junctionDegrees:
        kept = openJunctions(nodes, kept, junctionDegrees)
        nodes = straightenBends(nodes, kept)
    graph = CandidateGraph(nodes=nodes, edges=kept, isDelaunay=[True] * len(kept))
    return makeLayout(plate, params, graph, kept, name=f"truss{spacingFactor:g}-{junctionDegrees:g}")


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


def openJunctions(nodes, edges, minDegrees):
    """Remove ribs so no two ribs meet at an open (interior) junction at a sharp angle.

    At each sharp pair the longer rib goes (merging the two pockets beside it), unless that
    would leave its far end with fewer than two ribs; repeat until no sharp pair is left.
    Holes and wall points keep their fans: ribs converge on holes in the plates the user kept."""
    edges = set(edges)
    points = np.array([node.point for node in nodes])
    limit = math.radians(minDegrees)

    def neighbours(index):
        return [j if i == index else i for i, j in edges if index in (i, j)]

    for _ in range(4 * len(edges)):
        worst = None
        for index, node in enumerate(nodes):
            if node.kind != "interior":
                continue
            around = neighbours(index)
            if len(around) < 2:
                continue
            angles = sorted((math.atan2(*(points[j] - points[index])[::-1]) % (2 * math.pi), j) for j in around)
            for (a, ja), (b, jb) in zip(angles, angles[1:] + [(angles[0][0] + 2 * math.pi, angles[0][1])]):
                gap = b - a
                if gap < limit and (worst is None or gap < worst[0]):
                    worst = (gap, index, ja, jb)
        if worst is None:
            break
        _, index, ja, jb = worst
        options = sorted((ja, jb), key=lambda j: -np.hypot(*(points[j] - points[index])))
        for far in options:
            if len(neighbours(far)) > 2 or nodes[far].kind != "interior":
                edges.discard((min(index, far), max(index, far)))
                break
        else:
            edges.discard((min(index, options[0]), max(index, options[0])))
    return sorted(edges)


def straightenBends(nodes, edges):
    """An open junction left with two ribs is a bend: move it onto the line between its two
    neighbours, so the two ribs become one straight rib."""
    nodes = list(nodes)
    around = {}
    for i, j in edges:
        around.setdefault(i, []).append(j)
        around.setdefault(j, []).append(i)
    for index, others in around.items():
        if nodes[index].kind != "interior" or len(others) != 2:
            continue
        a, b = (np.array(nodes[k].point) for k in others)
        here = np.array(nodes[index].point)
        direction = b - a
        t = float(np.clip((here - a) @ direction / max(direction @ direction, 1e-12), 0.2, 0.8))
        x, y = a + t * direction
        nodes[index] = Node(float(x), float(y), kind="interior")
    return nodes


def keptPocketArea(plate, params):
    scale = math.sqrt(plate.materialArea)
    return keptPocketCoefficient * scale ** keptPocketExponent * params.pocketSizeFactor / defaultSizeFactor


def mainAngle(plate):
    """Direction of the plate's longest straight run: the long side of its smallest enclosing rectangle."""
    corners = list(plate.outline.minimum_rotated_rectangle.exterior.coords)
    sides = [(math.dist(a, b), math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) for a, b in zip(corners, corners[1:])]
    return max(sides)[1]


def alignedTruss(plate, params, pattern="diagonal", pitch=1.0):
    """Rails along the plate's main direction and rungs between them (see the module notes)."""
    if not wallCenterline(plate, params):
        return None
    target = pitch * keptPocketArea(plate, params)
    inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
    bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    region = inner.difference(unary_union(bosses)) if bosses else inner
    if region.is_empty:
        return None
    angle, centre = mainAngle(plate), plate.outline.centroid
    local = lambda geometry: affinity.rotate(geometry, -angle, origin=centre)
    world = lambda geometry: affinity.rotate(geometry, angle, origin=centre)
    regionLocal = local(region)
    minu, minv, maxu, maxv = regionLocal.bounds
    # A triangle of base p and height h (diagonal rungs) or a p x h rectangle (square rungs)
    # holds the target area with p = h.
    height = math.sqrt((2 if pattern == "diagonal" else 1) * target) + params.ribIn
    # Rails: through each row of two or more holes off the wall, then evenly between.
    reach = inner.buffer(-params.ribIn)
    centres = [local(featureCenter(feature)) for feature, boss in zip(plate.features, bosses) if reach.contains(boss)]
    rows = []
    for v in sorted(point.y for point in centres):
        if rows and v - rows[-1][-1] < holeRowTolerance:
            rows[-1].append(v)
        else:
            rows.append([v])
    rails = [float(np.mean(row)) for row in rows if len(row) >= 2]
    fixed = sorted({minv} | set(rails) | {maxv})
    filled = []
    for a, b in zip(fixed, fixed[1:]):
        count = max(1, int(round((b - a) / height)))
        filled += [a + k * (b - a) / count for k in range(count + 1)]
    levels = sorted(set(round(v, 6) for v in filled))
    span = (minu - height, maxu + height)
    lines = [LineString([(span[0], v), (span[1], v)]) for v in levels[1:-1]]
    # Rungs between neighbouring levels, on a pitch centred on the plate (symmetry).
    centreU = local(centre).x
    for a, b in zip(levels, levels[1:]):
        gap = b - a
        step = (2 * target / gap if pattern == "diagonal" else target / gap) + params.ribIn
        first = centreU - math.ceil((centreU - span[0]) / step) * step
        u = first
        while u <= span[1]:
            if pattern == "diagonal":
                lines.append(LineString([(u, a), (u + step / 2, b)]))
                lines.append(LineString([(u + step / 2, b), (u + step, a)]))
            else:
                lines.append(LineString([(u, a), (u, b)]))
            u += step
    # Into the plate, cut at holes, and noded where ribs cross, so ribs share junctions.
    clipArea = inner.difference(unary_union(plate.features)) if plate.features else inner
    clipped = world(unary_union(lines)).intersection(clipArea)
    pieces = [geometry for geometry in getattr(clipped, "geoms", [clipped]) if geometry.geom_type == "LineString" and geometry.length > 1e-6]
    noded = unary_union(pieces)
    segments = []
    for line in getattr(noded, "geoms", [noded]):
        coords = list(line.coords)
        segments += [(coords[k], coords[k + 1]) for k in range(len(coords) - 1)]
    nodes, index = [], {}
    boundary = clipArea.boundary

    def nodeAt(point):
        key = (round(point[0], 5), round(point[1], 5))
        if key not in index:
            onSolid = boundary.distance(Point(point)) < 1e-4  # ends on the wall or a boss: tied to solid
            index[key] = len(nodes)
            nodes.append(Node(point[0], point[1], kind="anchor" if onSolid else "interior"))
        return index[key]

    edges = sorted({tuple(sorted((nodeAt(a), nodeAt(b)))) for a, b in segments if math.dist(a, b) > 1e-4} - {(i, i) for i in range(len(index))})
    edges = [edge for edge in edges if edge[0] != edge[1]]
    if not edges:
        return None
    graph = CandidateGraph(nodes=nodes, edges=edges, isDelaunay=[True] * len(edges))
    return makeLayout(plate, params, graph, edges, name=f"aligned-{pattern}{pitch:g}")
