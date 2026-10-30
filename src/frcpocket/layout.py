"""Rib-graph layouts: a pocketing design is a set of straight ribs between nodes.

Nodes are feature centres (holes, bores, cutouts) and anchor points on the centre line
of the perimeter wall. The material kept is the perimeter wall, a boss ring around
every feature, and the chosen ribs. Everything else inside the plate is a pocket
candidate. Pockets are then *opened* by the tool radius: only what a round end mill of
that radius can reach is cut, which gives every pocket corner at least that radius
and drops slivers too narrow to machine. Pockets below the minimum area are skipped.

Each pocket's corners are filleted by a radius that scales with the pocket (soft corners,
like the WCP / Team 221 plates), never smaller than the tool radius.

This 2D version uses shapely polygons and is fast enough for the training loop.
cadExport rebuilds the same layout with exact OpenCascade geometry for the STEP output.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay
from shapely import affinity
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from frcpocket import geom2d
from frcpocket.plate import featureCenter, featureKind

arcSegments = 16  # shapely quad_segs: points per quarter circle
cornerTurnDegrees = 25  # outline corners sharper than this get an anchor node
featureNeighbors = 5  # direct hole-to-hole candidates per feature
minAnchorSpacingIn = 0.5
maxAnchorSpacingIn = 1.5
anchorsAcross = 7  # roughly this many anchors per sqrt(plate area) of wall


@dataclass
class Node:
    x: float
    y: float
    kind: str  # "feature" or "anchor"
    featureIndex: int | None = None
    featureKind: str | None = None

    @property
    def point(self):
        return (self.x, self.y)


@dataclass
class CandidateGraph:
    nodes: list
    edges: list  # (i, j) with i < j
    isDelaunay: list  # per edge: True when it came from the planar triangulation
    junctionStep: float = 0.0  # spacing of the free-junction lattice (0 = no lattice)

    def segment(self, edge):
        i, j = edge
        return LineString([self.nodes[i].point, self.nodes[j].point])


@dataclass
class Layout:
    graph: CandidateGraph
    ribs: list  # subset of graph.edges
    pockets: list = field(default_factory=list)  # shapely Polygons, inches
    name: str = ""
    ribWidths: dict = field(default_factory=dict)  # edge -> width (in); missing = params.ribIn

    def ribWidth(self, edge, params):
        return max(self.ribWidths.get(edge, params.ribIn), params.ribIn)


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]


def defaultAnchorSpacing(plate):
    """Real ribs meet the wall about every inch; this keeps candidates that dense."""
    return float(np.clip(math.sqrt(plate.outline.area) / anchorsAcross, minAnchorSpacingIn, maxAnchorSpacingIn))


def wallCenterline(plate, params):
    centre = plate.outline.buffer(-params.wallIn / 2, quad_segs=arcSegments)
    return [part.exterior for part in polygons(centre)]


def anchorPoints(plate, params, spacing, features=None):
    """Anchors on the wall centre line: opposite each feature, at outline corners, and evenly spaced."""
    rings = wallCenterline(plate, params)
    if not rings:
        return []
    anchors = []
    coords = np.asarray(plate.outline.exterior.coords)[:-1]
    for index in range(len(coords)):
        before, here, after = coords[index - 1], coords[index], coords[(index + 1) % len(coords)]
        a = math.atan2(here[1] - before[1], here[0] - before[0])
        b = math.atan2(after[1] - here[1], after[0] - here[0])
        if abs((b - a + math.pi) % (2 * math.pi) - math.pi) > math.radians(cornerTurnDegrees):
            ring = min(rings, key=lambda candidate: candidate.distance(Point(here)))
            anchors.append(ring.interpolate(ring.project(Point(here))))
    # The wall point nearest each feature: "hole straight out to the wall" ribs are the
    # most common rib in real plates. These take priority over the evenly spaced ones.
    for feature in features or []:
        centre = featureCenter(feature)
        ring = min(rings, key=lambda candidate: candidate.distance(centre))
        anchors.insert(0, ring.interpolate(ring.project(centre)))
    for ring in rings:
        count = max(3, int(ring.length / spacing))
        anchors.extend(ring.interpolate(k * ring.length / count) for k in range(count))
    kept = []
    for anchor in anchors:
        if all(anchor.distance(other) >= spacing * 0.45 for other in kept):
            kept.append(anchor)
    return kept


def interiorPoints(plate, params, spacing, bosses):
    """Hex-lattice junction points in the open area, aligned to the plate's principal axis.

    Centring and aligning the lattice on the plate keeps symmetric plates symmetric.
    """
    from shapely import affinity

    region = plate.outline.buffer(-(params.wallIn + params.ribIn / 2), quad_segs=4)
    if bosses:
        region = region.difference(unary_union(bosses).buffer(params.ribIn / 2, quad_segs=4))
    if region.is_empty:
        return []
    centre = plate.outline.centroid
    angle = math.degrees(math.atan2(*reversed(principalAxis(plate))))
    aligned = affinity.rotate(region, -angle, origin=centre)
    minx, miny, maxx, maxy = aligned.bounds
    rowStep = spacing * math.sqrt(3) / 2
    rows = int((maxy - miny) / rowStep) + 2
    columns = int((maxx - minx) / spacing) + 2
    prepared = prep(region)
    points = []
    for row in range(-rows, rows + 1):
        offset = spacing / 2 if row % 2 else 0.0
        for column in range(-columns, columns + 1):
            local = Point(centre.x + column * spacing + offset, centre.y + row * rowStep)
            world = affinity.rotate(local, angle, origin=centre)
            if prepared.contains(world):
                points.append(world)
    return points


def principalAxis(plate):
    coords = np.asarray(plate.outline.exterior.coords)[:-1]
    centred = coords - coords.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    return vectors[:, -1]


def candidateGraph(plate, params, anchorSpacing=None, extraNeighbors=4, interiorSpacing=None, withInterior=True):
    """Delaunay edges between features, wall anchors, and interior junctions, plus k-nearest extras."""
    spacing = anchorSpacing or defaultAnchorSpacing(plate)
    nodes = [
        Node(*featureCenter(feature).coords[0], kind="feature", featureIndex=index, featureKind=featureKind(feature))
        for index, feature in enumerate(plate.features)
    ]
    bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    unionBoss = unary_union(bosses) if bosses else None
    for anchor in anchorPoints(plate, params, spacing, plate.features):
        # A feature whose boss already covers this stretch of wall needs no anchor there.
        if unionBoss is not None and unionBoss.contains(anchor):
            continue
        nodes.append(Node(anchor.x, anchor.y, kind="anchor"))
    if withInterior:
        for point in interiorPoints(plate, params, interiorSpacing or spacing * 0.9, bosses):
            nodes.append(Node(point.x, point.y, kind="interior"))
    if len(nodes) < 3:
        return CandidateGraph(nodes=nodes, edges=[(0, 1)] if len(nodes) == 2 else [], isDelaunay=[True] * (len(nodes) == 2))

    points = np.array([node.point for node in nodes])
    triangulation = Delaunay(points)
    delaunayEdges = set()
    for simplex in triangulation.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((int(simplex[a]), int(simplex[b])))
            delaunayEdges.add((i, j))
    extraEdges = set()
    for i in range(len(nodes)):
        distances = np.hypot(*(points - points[i]).T)
        for j in np.argsort(distances)[1:extraNeighbors + 1]:
            edge = tuple(sorted((i, int(j))))
            if edge not in delaunayEdges:
                extraEdges.add(edge)
    # Straight trusses across the plate: the triangulation of the holes alone, and of the
    # holes plus wall anchors (no interior junctions), so a gusset can be braced leg to leg.
    featureAndWall = [index for index, node in enumerate(nodes) if node.kind != "interior"]
    for subset in ([index for index, node in enumerate(nodes) if node.kind == "feature"], featureAndWall):
        if len(subset) >= 3:
            try:
                sub = Delaunay(points[subset])
            except Exception:
                continue
            for simplex in sub.simplices:
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    edge = tuple(sorted((subset[int(simplex[a])], subset[int(simplex[b])])))
                    if edge not in delaunayEdges:
                        extraEdges.add(edge)
    # Hole-to-hole ribs are what real plates are built from (bearing to bearing, bearing
    # to bolt), so each feature also gets its nearest features directly, whatever sits
    # in between, and a rib straight out to its nearest wall anchor.
    featureNodes = [index for index, node in enumerate(nodes) if node.kind == "feature"]
    anchorNodes = [index for index, node in enumerate(nodes) if node.kind == "anchor"]
    for i in featureNodes:
        others = sorted(featureNodes, key=lambda j: np.hypot(*(points[j] - points[i])))
        for j in others[1:featureNeighbors + 1]:
            edge = tuple(sorted((i, j)))
            if edge not in delaunayEdges:
                extraEdges.add(edge)
        if anchorNodes:
            j = min(anchorNodes, key=lambda j: np.hypot(*(points[j] - points[i])))
            edge = tuple(sorted((i, j)))
            if edge not in delaunayEdges:
                extraEdges.add(edge)

    return filterEdges(plate, params, nodes, points, delaunayEdges, extraEdges)


def filterEdges(plate, params, nodes, points, delaunayEdges, extraEdges):
    """Keep candidate ribs that stay inside the plate, don't just trace the wall, and
    never cross a hole they don't end at."""
    insideOutline = prep(plate.outline.buffer(1e-6))
    preparedFeatures = [prep(feature) for feature in plate.features]
    edges, isDelaunay = [], []
    for edge in sorted(delaunayEdges) + sorted(extraEdges):
        i, j = edge
        segment = LineString([points[i], points[j]])
        if not insideOutline.contains(segment):
            continue  # would leave the plate across a concave notch
        if nodes[i].kind == "anchor" and nodes[j].kind == "anchor":
            if plate.outline.exterior.distance(segment.interpolate(0.5, normalized=True)) < params.wallIn:
                continue  # runs along the wall, which is already solid
        # A rib may clip another boss (it simply merges into it) but never cross a hole.
        endpoints = {nodes[i].featureIndex, nodes[j].featureIndex}
        crossesHole = any(
            index not in endpoints and preparedFeatures[index].intersects(segment) for index in range(len(plate.features))
        )
        if crossesHole:
            continue
        edges.append(edge)
        isDelaunay.append(edge in delaunayEdges)
    return CandidateGraph(nodes=nodes, edges=edges, isDelaunay=isDelaunay)


def ribBand(graph, edge, width):
    return graph.segment(edge).buffer(width / 2, cap_style="flat")


def buildPockets(plate, params, graph, ribs, ribWidths=None):
    ribWidths = ribWidths or {}
    inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
    keep = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    keep += [ribBand(graph, edge, max(ribWidths.get(edge, params.ribIn), params.ribIn)) for edge in ribs]
    raw = inner.difference(unary_union(keep)) if keep else inner
    return finishRegions(polygons(raw), params)


def finishRegions(regions, params):
    """Open each raw region by its own fillet radius, then drop pockets not worth cutting."""
    pockets = []
    for region in regions:
        opened = toolOpening(region, filletRadius(region, params))
        # Opening never grows a region, so pockets stay inside it; intersect to drop float noise.
        pockets += [pocket.buffer(0) for pocket in polygons(opened.intersection(region))]
    pockets = [pocket for pocket in pockets if isWorthCutting(pocket, params)]
    return enforcePocketGaps(pockets, params)


def inscribedRadius(region, iterations=6):
    """Radius of the largest disk inside region (binary search on inward offsets)."""
    low, high = 0.0, math.sqrt(region.area / math.pi)
    for _ in range(iterations):
        middle = (low + high) / 2
        if region.buffer(-middle, quad_segs=4).is_empty:
            high = middle
        else:
            low = middle
    return low


def filletRadius(region, params):
    """Soft-corner radius for one pocket region: a fraction of its size, clamped."""
    style = params.filletFraction * inscribedRadius(region)
    return max(params.cornerRadiusIn, min(params.maxFilletIn, style))


def toolOpening(region, radius):
    return region.buffer(-radius, quad_segs=arcSegments).buffer(radius, quad_segs=arcSegments)


def enforcePocketGaps(pockets, params, maxPasses=4):
    """Keep every pair of pockets at least a rib apart.

    Tool opening can split one region at a narrow neck into two pockets that sit closer
    than a rib. When that happens the smaller pocket is trimmed back to exactly one rib
    from the other and re-opened by the tool, or dropped if nothing cuttable is left.
    """
    gap = params.ribIn
    for _ in range(maxPasses):
        changed = False
        pockets = sorted(pockets, key=lambda pocket: -pocket.area)
        for big in range(len(pockets)):
            for small in range(big + 1, len(pockets)):
                if pockets[big] is None or pockets[small] is None:
                    continue
                if pockets[big].distance(pockets[small]) >= gap - 1e-6:
                    continue
                trimmed = pockets[small].difference(pockets[big].buffer(gap, quad_segs=arcSegments))
                pieces = [piece for piece in polygons(toolOpening(trimmed, params.cornerRadiusIn).intersection(trimmed))
                          if isWorthCutting(piece, params)]
                pockets[small] = max(pieces, key=lambda piece: piece.area) if pieces else None
                changed = True
        pockets = [pocket for pocket in pockets if pocket is not None]
        if not changed:
            break
    return pockets


def isWorthCutting(pocket, params):
    """Big enough in area and wide enough somewhere to be worth the machine time."""
    if pocket.area < params.minPocketArea:
        return False
    return not pocket.buffer(-params.minPocketWidth / 2).is_empty


def pruneDangling(graph, ribs):
    """Drop ribs that dead-end at an interior junction (repeat until none are left)."""
    ribs = list(ribs)
    while True:
        degree = {}
        for i, j in ribs:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        dangling = {node for node, count in degree.items() if count == 1 and graph.nodes[node].kind == "interior"}
        if not dangling:
            return ribs
        ribs = [edge for edge in ribs if not (set(edge) & dangling)]


def makeLayout(plate, params, graph, ribs, name="", ribWidths=None):
    ribs = pruneDangling(graph, ribs)
    ribWidths = dict(ribWidths or {})
    pockets = buildPockets(plate, params, graph, ribs, ribWidths)
    return Layout(graph=graph, ribs=list(ribs), pockets=pockets, name=name, ribWidths=ribWidths)


def removedFraction(plate, pockets, params=None):
    depthFraction = 1.0
    if params is not None and params.depthIn is not None:
        depthFraction = min(1.0, params.depthIn / plate.thicknessIn)
    return sum(pocket.area for pocket in pockets) * depthFraction / plate.materialArea


spokeStepDegrees = 30  # spoke candidates around big bores, every this many degrees
rhythmFactor = 1.5  # wall points between holes are about this many target pocket widths apart
bigBoreDiameterIn = 0.5
junctionClearance = 0.8  # a free junction stays this many lattice steps from any hole or wall point


def triangulationEdges(points, subset):
    edges = set()
    if len(subset) >= 3:
        try:
            triangulation = Delaunay(points[subset])
        except Exception:
            return edges
        for simplex in triangulation.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                edges.add(tuple(sorted((subset[int(simplex[a])], subset[int(simplex[b])]))))
    return edges


def holeGraph(plate, params):
    """Candidate ribs the way the gold plates are drawn: straight lines from hole centres.

    Nodes are only meaningful points: the holes themselves, the wall point straight out
    from each hole that isn't already touching the wall, spoke points around big bores
    (every 30 degrees, where the ray meets the wall), the outline's corners, and rhythm
    points that split long stretches of wall evenly. There is no grid of junctions in open
    space, so every rib runs hole to hole, hole to wall, or bore to rim, and pockets are
    the faces of that network. Nearby wall points are merged order-independently, which
    keeps symmetric plates symmetric.
    """
    from frcpocket import designFeatures

    rings = wallCenterline(plate, params)
    nodes = [
        Node(*featureCenter(feature).coords[0], kind="feature", featureIndex=index, featureKind=featureKind(feature))
        for index, feature in enumerate(plate.features)
    ]
    bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    unionBoss = unary_union(bosses) if bosses else None
    minGap = 2.5 * params.ribIn

    def merged(candidates):
        """Cluster candidates closer than minGap (union-find, so order doesn't matter);
        each cluster becomes one point on the wall, owned by all its members' owners."""
        candidates = [(point, owner) for point, owner in candidates
                      if unionBoss is None or not unionBoss.contains(point)]
        parent = list(range(len(candidates)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                if candidates[a][0].distance(candidates[b][0]) < minGap:
                    parent[find(a)] = find(b)
        clusters = {}
        for index in range(len(candidates)):
            clusters.setdefault(find(index), []).append(index)
        points = []
        for members in clusters.values():
            centre = Point(np.mean([candidates[m][0].x for m in members]), np.mean([candidates[m][0].y for m in members]))
            ring = min(rings, key=lambda candidate: candidate.distance(centre))
            owners = {candidates[m][1] for m in members if candidates[m][1] is not None}
            points.append((ring.interpolate(ring.project(centre)), owners))
        return points

    candidates = []
    if rings:
        coords = np.asarray(plate.outline.exterior.coords)[:-1]
        for index in range(len(coords)):
            before, here, after = coords[index - 1], coords[index], coords[(index + 1) % len(coords)]
            a = math.atan2(here[1] - before[1], here[0] - before[0])
            b = math.atan2(after[1] - here[1], after[0] - here[0])
            if abs((b - a + math.pi) % (2 * math.pi) - math.pi) > math.radians(cornerTurnDegrees):
                ring = min(rings, key=lambda candidate: candidate.distance(Point(here)))
                candidates.append((ring.interpolate(ring.project(Point(here))), None))
        outerRing = plate.outline.exterior
        diagonal = math.dist(*np.reshape(plate.outline.bounds, (2, 2)))
        baseAngle = geom2d.principalAngle(plate.outline)
        for featureIndex, feature in enumerate(plate.features):
            centre = featureCenter(feature)
            ring = min(rings, key=lambda candidate: candidate.distance(centre))
            if outerRing.distance(bosses[featureIndex]) > params.wallIn + params.ribIn:
                candidates.append((ring.interpolate(ring.project(centre)), featureIndex))
            if 2 * math.sqrt(feature.area / math.pi) >= bigBoreDiameterIn:
                for step in range(int(360 / spokeStepDegrees)):
                    angle = math.radians(baseAngle + step * spokeStepDegrees)
                    ray = LineString([centre.coords[0], (centre.x + diagonal * math.cos(angle), centre.y + diagonal * math.sin(angle))])
                    hits = ray.intersection(ring)
                    hitPoints = [geometry for geometry in getattr(hits, "geoms", [hits]) if geometry.geom_type == "Point"]
                    if hitPoints:
                        candidates.append((min(hitPoints, key=lambda point: point.distance(centre)), featureIndex))
    anchors = merged(candidates)
    # Rhythm points: split each long stretch of wall between anchors into even steps near
    # the target pocket width, so truss diagonals can meet the wall between holes.
    step = rhythmFactor * math.sqrt(designFeatures.referencePocketArea(math.sqrt(plate.materialArea)))
    rhythm = []
    for ring in rings:
        positions = sorted(ring.project(point) for point, _ in anchors if ring.distance(point) < 1e-6) or [0.0]
        loop = positions + [positions[0] + ring.length]
        for start, end in zip(loop, loop[1:]):
            count = int(round((end - start) / step))
            rhythm += [(ring.interpolate((start + k * (end - start) / count) % ring.length), None) for k in range(1, count)]
    anchors = merged([(point, owner) for point, owners in anchors for owner in (owners or {None})] + rhythm)

    firstAnchor = len(nodes)
    nodes += [Node(point.x, point.y, kind="anchor") for point, _ in anchors]
    junctionStep = step * math.sqrt(params.pocketSizeFactor)
    junctions = sparseJunctions(plate, params, nodes, junctionStep)
    nodes += [Node(point.x, point.y, kind="interior") for point in junctions]
    if len(nodes) < 3:
        return CandidateGraph(nodes=nodes, edges=[(0, 1)] if len(nodes) == 2 else [], isDelaunay=[True] * (len(nodes) == 2))
    points = np.array([node.point for node in nodes])
    delaunayEdges = triangulationEdges(points, list(range(len(nodes))))
    featureIndices = [index for index, node in enumerate(nodes) if node.kind == "feature"]
    extraEdges = triangulationEdges(points, featureIndices) - delaunayEdges
    for slot, (_, owners) in enumerate(anchors):
        for featureIndex in owners:
            edge = tuple(sorted((featureIndex, firstAnchor + slot)))
            if edge not in delaunayEdges:
                extraEdges.add(edge)
    graph = filterEdges(plate, params, nodes, points, delaunayEdges, extraEdges)
    graph.junctionStep = junctionStep if junctions else 0.0
    return graph


def sparseJunctions(plate, params, nodes, step):
    """Free rib junctions where the plate has no holes to run ribs between.

    Points of a triangular lattice (spacing `step`, centred on the plate and aligned with
    it, so symmetric plates stay symmetric) that are at least junctionClearance steps from
    every hole and wall point, clear of every boss, and inside the wall. On hole-dense
    plates like the example plates almost none survive; on plates with a few holes spread
    far apart they let the decoder cut gold-sized cells instead of giant ones."""
    inner = plate.material.buffer(-(params.wallIn + 0.5 * step))
    if inner.is_empty:
        return []
    centre = plate.outline.centroid
    angle = math.radians(geom2d.principalAngle(plate.outline))
    reach = math.dist(*np.reshape(plate.outline.bounds, (2, 2)))
    taken = np.array([node.point for node in nodes]) if nodes else np.zeros((0, 2))
    bossZone = unary_union([feature.buffer(params.ringFor(feature) + params.ribIn) for feature in plate.features]) if plate.features else None
    rowStep = step * math.sqrt(3) / 2
    points = []
    for row in range(-int(reach / rowStep) - 1, int(reach / rowStep) + 2):
        offset = 0.5 * step if row % 2 else 0.0
        for column in range(-int(reach / step) - 1, int(reach / step) + 2):
            u, v = column * step + offset, row * rowStep
            x = centre.x + u * math.cos(angle) - v * math.sin(angle)
            y = centre.y + u * math.sin(angle) + v * math.cos(angle)
            point = Point(x, y)
            if not inner.contains(point) or (bossZone is not None and bossZone.contains(point)):
                continue
            if len(taken) and np.hypot(*(taken - (x, y)).T).min() < junctionClearance * step:
                continue
            points.append(point)
    return points


def isLatticeEdge(graph, edge):
    """A rib between two neighbouring free junctions of the lattice."""
    i, j = edge
    if not graph.junctionStep or graph.nodes[i].kind != "interior" or graph.nodes[j].kind != "interior":
        return False
    return math.dist(graph.nodes[i].point, graph.nodes[j].point) <= 1.1 * graph.junctionStep


def defaultGraph(plate, params):
    """The candidate graph the ML generator, projection, and scorer data use."""
    return symmetricClosure(plate, holeGraph(plate, params))


symmetryThreshold = 0.95  # material IoU with its image above this counts as symmetric


def mirrorGeometry(geometry, centre, angle, transform):
    """Mirror about the plate's principal x or y axis through centre, or turn 180°."""
    aligned = affinity.rotate(geometry, -angle, origin=centre)
    if transform == "x":
        aligned = affinity.scale(aligned, xfact=-1, yfact=1, origin=centre)
    elif transform == "y":
        aligned = affinity.scale(aligned, xfact=1, yfact=-1, origin=centre)
    else:
        aligned = affinity.rotate(aligned, 180, origin=centre)
    return affinity.rotate(aligned, angle, origin=centre)


def plateSymmetries(plate):
    """The symmetries the plate's material has, from "x", "y" (mirrors) and "rot" (180°)."""
    centre, angle = plate.outline.centroid, geom2d.principalAngle(plate.outline)
    return [transform for transform in ("x", "y", "rot")
            if geom2d.iou(plate.material, mirrorGeometry(plate.material, centre, angle, transform)) >= symmetryThreshold]


def nodeMirrors(plate, nodes, transforms):
    """Per transform: node index -> the node of the same kind at its image, where there is one."""
    centre, angle = plate.outline.centroid, geom2d.principalAngle(plate.outline)
    points = np.array([node.point for node in nodes])
    tolerance = 0.04 * math.sqrt(plate.materialArea)
    maps = []
    for transform in transforms:
        nodeMap = {}
        for index, node in enumerate(nodes):
            image = np.array(mirrorGeometry(Point(node.point), centre, angle, transform).coords[0])
            distances = np.hypot(*(points - image).T)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= tolerance and nodes[nearest].kind == node.kind:
                nodeMap[index] = nearest
        maps.append(nodeMap)
    return maps


def symmetricClosure(plate, graph):
    """Add the symmetry images of every candidate rib.

    A triangulation of a symmetric point set picks diagonals arbitrarily wherever four
    points are cocircular (common on symmetric plates), so a rib's mirror image can be
    missing. Images are added with the original's planar flag.
    """
    transforms = plateSymmetries(plate)
    if not transforms or len(graph.nodes) < 2:
        return graph
    maps = nodeMirrors(plate, graph.nodes, transforms)
    edges, flags = list(graph.edges), list(graph.isDelaunay)
    known = set(edges)
    frontier = list(range(len(edges)))
    while frontier:
        k = frontier.pop()
        i, j = edges[k]
        for nodeMap in maps:
            if i in nodeMap and j in nodeMap and nodeMap[i] != nodeMap[j]:
                image = tuple(sorted((nodeMap[i], nodeMap[j])))
                if image not in known:
                    known.add(image)
                    edges.append(image)
                    flags.append(flags[k])
                    frontier.append(len(edges) - 1)
    return CandidateGraph(nodes=graph.nodes, edges=edges, isDelaunay=flags, junctionStep=graph.junctionStep)
