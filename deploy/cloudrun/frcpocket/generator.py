"""ML rib generator: a small graph neural network that chooses which candidate ribs to keep.

Input: the plate's candidate rib graph (features, wall anchors, interior junctions,
candidate ribs) plus the pocketing parameters. Output: for every candidate rib, a keep
probability and a width class. Decoding turns those into a clean, valid layout:
  1. mirror-average the probabilities on symmetric plates (symmetric output)
  2. sample (or take the most likely ribs; "densified" variants add the next most
     likely triangulation ribs for a stiffer option)
  3. drop crossing ribs and ribs meeting at under 20 degrees, keeping the more confident one
  4. connect any hole whose boss would float, using its most likely rib
  5. split pockets bigger or longer than the gold plates use, along a likely rib that
     halves them evenly
  6. on symmetric plates, add every kept rib's mirror partner
  7. leave crumbs and unsplittable slivers solid
The 2D layout is then validated and cut with exact geometry like any other design.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import Point
from shapely.ops import unary_union
from torch import nn

from frcpocket import designFeatures, geom2d, validator
from frcpocket.layout import (candidateGraph, defaultGraph, inscribedRadius, isLatticeEdge, Layout, makeLayout,
                              mirrorGeometry, nodeMirrors, plateSymmetries, polygons)
from frcpocket.projection import ribWidthClasses

nodeKinds = ("feature", "anchor", "interior")
featureKinds = ("bolt", "bearing", "cutout")
pairKinds = [("feature", "feature"), ("anchor", "feature"), ("feature", "interior"),
             ("anchor", "anchor"), ("anchor", "interior"), ("interior", "interior")]
nodeDim = 11
edgeDim = 18
hiddenDim = 64
densifyFractions = (0.2, 0.4)  # densified variants added to every proposal set
maxPocketShare = 0.15  # largest gold pocket is ~14% of its plate; nothing bigger is decoded
areaCapFactor = 2.5  # and no pocket above 2.5x the gold pocket size fitted for this plate size
maxElongation = 3.0  # gold pockets: median 1.5, 95% under 2.4 (long side / short side)
maxSplitAttempts = 3  # best-scoring splits tried per pocket before giving up on it
minRibAngleDegrees = 20  # ribs sharing an end meet at least this wide (real truss members: 30-60 deg)
crumbFraction = 0.2  # the user's examples keep small pockets near holes; only true crumbs go
sliverElongation = 4.5  # the most elongated gold pocket is 4.3x as long as it is wide
sliverFraction = 1.0  # slivers up to this many typical gold pockets in area get filled
# A long pocket much narrower than the design's own typical pocket, squeezed between the
# wall and a cutout or hole, reads as a leftover: only 9 of 216 gold pockets are long and
# narrow, nearly all on plates where every pocket is. Long fan pockets between ribs stay.
stripElongation = 3.0
stripWidthFraction = 0.65
stripBoundShare = 0.6  # share of the outline that runs along the wall or a boss, not a rib


@dataclass
class GraphTensors:
    nodeFeatures: torch.Tensor  # (nodes, nodeDim)
    edgeFeatures: torch.Tensor  # (edges, edgeDim)
    edgeIndex: torch.Tensor  # (2, edges)


def plateScale(plate):
    return math.sqrt(plate.materialArea)


def encodeGraph(plate, params, graph):
    scale = plateScale(plate)
    centre = np.array(plate.outline.centroid.coords[0])
    angle = math.radians(geom2d.principalAngle(plate.outline))
    axis = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-axis[1], axis[0]])
    ring = plate.outline.exterior
    degree = np.zeros(len(graph.nodes))
    for i, j in graph.edges:
        degree[i] += 1
        degree[j] += 1
    nodeRows = []
    for index, node in enumerate(graph.nodes):
        point = np.array(node.point)
        relative = point - centre
        kindHot = [float(node.kind == kind) for kind in nodeKinds]
        featureHot = [float(node.featureKind == kind) for kind in featureKinds]
        size = 0.0
        if node.featureIndex is not None:
            size = math.sqrt(plate.features[node.featureIndex].area / math.pi) / scale
        nodeRows.append(kindHot + featureHot + [
            size * 5,
            ring.distance(Point(point)) / scale * 5,
            abs(relative @ axis) / scale * 2,
            abs(relative @ across) / scale * 2,
            degree[index] / 10,
        ])
    segments = [graph.segment(edge) for edge in graph.edges]
    crossings = np.zeros(len(graph.edges))
    if len(segments) < 1500:
        from shapely.strtree import STRtree

        tree = STRtree(segments)
        for k, segment in enumerate(segments):
            for other in tree.query(segment):
                if other != k and segment.crosses(segments[other]):
                    crossings[k] += 1
    globalRow = [params.ribIn / scale * 10, params.wallIn / scale * 10, params.cornerRadiusIn / scale * 10]
    edgeRows = []
    for k, ((i, j), planar) in enumerate(zip(graph.edges, graph.isDelaunay)):
        a, b = graph.nodes[i], graph.nodes[j]
        vector = np.array(b.point) - np.array(a.point)
        length = float(np.hypot(*vector))
        direction = vector / max(length, 1e-9)
        pair = tuple(sorted((a.kind, b.kind)))
        pairHot = [float(pair == tuple(sorted(kinds))) for kinds in pairKinds]
        bearings = float(a.featureKind == "bearing") + float(b.featureKind == "bearing")
        middle = segments[k].interpolate(0.5, normalized=True)
        edgeRows.append([
            length / scale * 3,
            math.log(max(length, 1e-3) / params.ribIn) / 3,
            float(planar),
            *pairHot,
            ring.distance(middle) / scale * 5,
            abs(direction @ axis),
            abs(direction @ across),
            crossings[k] / 5,
            bearings / 2,
            *globalRow,
            0.0,  # reserved
        ])
    return GraphTensors(
        nodeFeatures=torch.tensor(nodeRows, dtype=torch.float32),
        edgeFeatures=torch.tensor(edgeRows, dtype=torch.float32).reshape(-1, edgeDim),
        edgeIndex=torch.tensor(graph.edges, dtype=torch.long).T.reshape(2, -1),
    )


class RibGnn(nn.Module):
    def __init__(self, rounds=3):
        super().__init__()
        self.nodeIn = nn.Sequential(nn.Linear(nodeDim, hiddenDim), nn.ReLU(), nn.Linear(hiddenDim, hiddenDim))
        self.edgeIn = nn.Sequential(nn.Linear(edgeDim, hiddenDim), nn.ReLU(), nn.Linear(hiddenDim, hiddenDim))
        self.messages = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * hiddenDim, hiddenDim), nn.ReLU(), nn.Linear(hiddenDim, hiddenDim)) for _ in range(rounds)
        )
        self.keepHead = nn.Sequential(nn.Linear(3 * hiddenDim, hiddenDim), nn.ReLU(), nn.Linear(hiddenDim, 1))
        self.widthHead = nn.Sequential(nn.Linear(3 * hiddenDim, hiddenDim), nn.ReLU(), nn.Linear(hiddenDim, len(ribWidthClasses)))

    def forward(self, tensors):
        nodes = self.nodeIn(tensors.nodeFeatures)
        edges = self.edgeIn(tensors.edgeFeatures)
        source, target = tensors.edgeIndex
        counts = torch.zeros(len(nodes)).index_add_(0, source, torch.ones(len(source))).index_add_(0, target, torch.ones(len(target)))
        counts = counts.clamp(min=1).unsqueeze(-1)
        for message in self.messages:
            # Symmetric in the two endpoints, so a rib's direction doesn't matter.
            update = message(torch.cat([edges, nodes[source] + nodes[target], (nodes[source] - nodes[target]).abs()], dim=-1))
            incoming = torch.zeros_like(nodes).index_add_(0, source, update).index_add_(0, target, update) / counts
            nodes = nodes + incoming
            edges = edges + update
        readout = torch.cat([edges, nodes[source] + nodes[target], (nodes[source] - nodes[target]).abs()], dim=-1)
        return self.keepHead(readout).squeeze(-1), self.widthHead(readout)


def mirrorMap(plate, graph):
    """Edge index -> its orbit under the plate's symmetries, or None if it has none.

    Every symmetry the plate has (mirror about either principal axis, 180° turn) counts,
    so a plate symmetric both ways gets designs symmetric both ways. The orbit is the
    sorted tuple of edge indices the edge maps to, itself included. Edges whose image
    is missing from the graph are left out, so the decoder can rule them out.
    """
    transforms = plateSymmetries(plate)
    if not transforms:
        return None
    edgeLookup = {edge: k for k, edge in enumerate(graph.edges)}
    edgeMaps = []
    for nodeMap in nodeMirrors(plate, graph.nodes, transforms):
        edgeMap = {}
        for k, (i, j) in enumerate(graph.edges):
            if i in nodeMap and j in nodeMap:
                partner = edgeLookup.get(tuple(sorted((nodeMap[i], nodeMap[j]))))
                if partner is not None:
                    edgeMap[k] = partner
        edgeMaps.append(edgeMap)
    orbits = {}
    for k in range(len(graph.edges)):
        orbit, frontier, complete = {k}, [k], True
        while frontier and complete:
            edge = frontier.pop()
            for edgeMap in edgeMaps:
                image = edgeMap.get(edge)
                if image is None:
                    complete = False
                    break
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        if complete:
            orbits[k] = tuple(sorted(orbit))
    return orbits


def orbitOf(k, edgeMirror):
    """Edge indices kept or dropped together with edge k (just k on asymmetric plates)."""
    return edgeMirror.get(k, (k,)) if edgeMirror else (k,)


def connectFloatingFeatures(plate, params, graph, chosen, probabilities, edgeMirror=None):
    """Connect any hole whose boss would float, or any piece of the junction lattice that
    isn't tied to the wall, along its most likely path to the network.

    "Connected" means joined to the wall (through anchors, or a boss that already reaches
    the wall) by kept ribs or overlapping bosses. A floating hole gets the cheapest path
    (cost -log p per rib) to the connected network, so any interior junctions on the
    path have two ribs and survive dead-end pruning. Paths avoid crossing kept ribs where
    they can, and on symmetric plates each path rib comes with its symmetry images (which
    connect the images of the floating hole).
    """
    import heapq

    bosses = [feature.buffer(params.boss, quad_segs=8) for feature in plate.features]
    inner = plate.outline.buffer(-params.wallIn, quad_segs=8)
    wallTouching = {index for index, boss in enumerate(bosses) if not inner.contains(boss)}
    featureNodes = {node.featureIndex: index for index, node in enumerate(graph.nodes) if node.kind == "feature"}
    overlaps = [(featureNodes[a], featureNodes[b]) for a in range(len(bosses)) for b in range(a + 1, len(bosses))
                if a in featureNodes and b in featureNodes and bosses[a].intersects(bosses[b])]
    segments = [graph.segment(edge) for edge in graph.edges]
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    costs = [-math.log(max(float(probability), 1e-6)) for probability in probabilities]
    chosen = set(chosen)

    def orbitEdges(k):
        return [graph.edges[member] for member in orbitOf(k, edgeMirror)]

    def clean(k):
        """The rib's whole orbit fits: no crossing or sharp meeting with kept ribs or itself."""
        members = orbitEdges(k)
        return all(not conflicts(graph, segments, lookup, edge, [other for other in chosen | set(members) if other != edge], params)
                   for edge in members if edge not in chosen)

    def connectedToWall():
        reached = {index for index, node in enumerate(graph.nodes)
                   if node.kind == "anchor" or (node.featureIndex is not None and node.featureIndex in wallTouching)}
        links = {index: set() for index in range(len(graph.nodes))}
        for i, j in chosen:
            links[i].add(j)
            links[j].add(i)
        for a, b in overlaps:
            links[a].add(b)
            links[b].add(a)
        frontier = list(reached)
        while frontier:
            node = frontier.pop()
            for other in links[node]:
                if other not in reached:
                    reached.add(other)
                    frontier.append(other)
        return reached

    def cheapestPath(start, reached, usable):
        adjacency = {index: [] for index in range(len(graph.nodes))}
        for k, (i, j) in enumerate(graph.edges):
            if usable(k):
                adjacency[i].append((j, k))
                adjacency[j].append((i, k))
        distance, previous, heap = {start: 0.0}, {}, [(0.0, start)]
        while heap:
            cost, node = heapq.heappop(heap)
            if node in reached:
                path = []
                while node != start:
                    node, edgeIndex = previous[node]
                    path.append(edgeIndex)
                return path
            if cost > distance.get(node, float("inf")):
                continue
            for other, edgeIndex in adjacency[node]:
                newCost = cost + costs[edgeIndex]
                if newCost < distance.get(other, float("inf")):
                    distance[other], previous[other] = newCost, (node, edgeIndex)
                    heapq.heappush(heap, (newCost, other))
        return None

    for _ in range(len(graph.nodes) + 1):
        reached = connectedToWall()
        floating = [index for index in featureNodes.values() if index not in reached]
        floating += sorted({node for edge in chosen for node in edge
                            if graph.nodes[node].kind == "interior" and node not in reached})
        if not floating or not reached:
            return chosen
        path = cheapestPath(floating[0], reached, lambda k: graph.edges[k] in chosen or clean(k))
        if path is not None:
            for edgeIndex in path:
                chosen.update(orbitEdges(edgeIndex))
            continue
        # No clean path: connect anyway (a floating hole breaks a hard rule; a crossing doesn't).
        path = cheapestPath(floating[0], reached, lambda k: True)
        if path is None:
            return chosen  # no candidate path at all; the validator will reject it
        chosen.update(graph.edges[edgeIndex] for edgeIndex in path)
    return chosen


def tooAcute(graph, edge, other, minAngle=minRibAngleDegrees):
    """Two ribs leaving the same node at less than minAngle degrees leave a sliver."""
    shared = set(edge) & set(other)
    if len(shared) != 1:
        return False
    node = shared.pop()
    origin = np.array(graph.nodes[node].point)
    a = np.array(graph.nodes[edge[0] if edge[1] == node else edge[1]].point) - origin
    b = np.array(graph.nodes[other[0] if other[1] == node else other[1]].point) - origin
    cosine = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return cosine > math.cos(math.radians(minAngle))


def ribCap(graph, node, params):
    """Most ribs a node may carry under params.maxHoleRibs (None = no limit). Free
    junctions carry their whole lattice, so they have no cap."""
    data = graph.nodes[node]
    if params is None or not params.maxHoleRibs or data.kind == "interior":
        return None
    if data.kind == "feature":
        return params.maxHoleRibs if data.featureKind == "bolt" else None
    return max(2, params.maxHoleRibs - 1)


def conflicts(graph, segments, lookup, edge, kept, params=None):
    """True if edge crosses a kept rib, meets one at too sharp an angle, or would give a
    node more ribs than the style allows."""
    k = lookup[edge]
    minAngle = params.minRibAngleDegrees if params is not None else minRibAngleDegrees
    for node in edge:
        cap = ribCap(graph, node, params)
        if cap is not None and sum(1 for other in kept if node in other) >= cap:
            return True
    for other in kept:
        if set(edge) & set(other):
            if tooAcute(graph, edge, other, minAngle):
                return True
        elif segments[k].crosses(segments[lookup[other]]):
            return True
    return False


def decode(plate, params, graph, keepLogits, widthLogits, rng=None, temperature=1.0, greedy=False, edgeMirror=None):
    probabilities = torch.sigmoid(keepLogits / max(temperature, 1e-3)).numpy()
    widths = torch.softmax(widthLogits, dim=-1).numpy()
    if edgeMirror:
        # Average over each symmetry orbit, and give ribs with no complete orbit (e.g. to a
        # wall anchor with no mirror image) almost no chance, so the result stays symmetric.
        averaged = np.full_like(probabilities, 1e-4)
        averagedWidths = widths.copy()
        for k, orbit in edgeMirror.items():
            averaged[k] = probabilities[list(orbit)].mean()
            averagedWidths[k] = widths[list(orbit)].mean(axis=0)
        probabilities, widths = averaged, averagedWidths
    draws = rng.random(len(graph.edges)) if rng is not None else np.full(len(graph.edges), 0.5)
    if edgeMirror:
        for k, orbit in edgeMirror.items():
            draws[k] = draws[orbit[0]]
    keep = probabilities > 0.5 if greedy else draws < probabilities
    segments = [graph.segment(edge) for edge in graph.edges]
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    # Big gaps between holes get one consistent pattern: every lattice rib between free
    # junctions is kept (equal triangles of the target pocket size), and the model's ribs
    # tie that pattern to the holes and the wall. Then whole orbits at a time, most likely
    # first: an orbit whose members would cross a kept rib (or each other) is skipped.
    chosen = [edge for edge in graph.edges if isLatticeEdge(graph, edge)]
    for orbit in sorted({orbitOf(k, edgeMirror) for k in range(len(graph.edges)) if keep[k]},
                        key=lambda orbit: (-probabilities[orbit[0]], orbit)):
        members = [graph.edges[k] for k in orbit]
        trial = []
        for edge in members:
            if conflicts(graph, segments, lookup, edge, chosen + trial, params):
                break
            trial.append(edge)
        else:
            chosen += trial
    chosen = connectFloatingFeatures(plate, params, graph, chosen, probabilities, edgeMirror)
    ribWidths = {edge: params.ribIn * ribWidthClasses[int(np.argmax(widths[lookup[edge]]))] for edge in chosen}
    layout = makeLayout(plate, params, graph, sorted(chosen), name="ml", ribWidths=ribWidths)
    layout = splitOversizedPockets(plate, params, graph, layout, probabilities, segments, widths, edgeMirror)
    layout = mergeSlotPockets(plate, params, graph, layout, probabilities, segments, edgeMirror)
    layout = mirrorClosure(plate, params, graph, layout, probabilities, segments, edgeMirror)
    return fillJunkPockets(plate, params, layout)


def fillJunkPockets(plate, params, layout):
    """Leave solid what the gold plates never cut: crumbs far below the plate's typical
    pocket size, long thin slivers that couldn't be split, and strips much narrower than
    the design's other pockets (e.g. along a cutout).
    Filling a pocket only adds material, so it can't break any rule."""
    scale = math.sqrt(plate.materialArea)
    crumb = crumbFraction * designFeatures.referencePocketArea(scale) * params.pocketSizeFactor
    radii = [inscribedRadius(pocket) for pocket in layout.pockets]
    typical = float(np.median(radii)) if radii else 0.0

    def isStrip(pocket, radius):
        return (elongation(pocket) > stripElongation and radius < stripWidthFraction * typical
                and boundShare(plate, params, pocket) > stripBoundShare)

    def isSliver(pocket):
        # Only small slivers: filling a big slot would leave a large part of the plate solid,
        # which looks worse than the slot (the scorer weighs slots against the rest).
        return elongation(pocket) > sliverElongation and pocket.area < sliverFraction * referenceArea

    referenceArea = designFeatures.referencePocketArea(scale) * params.pocketSizeFactor
    kept = [pocket for pocket, radius in zip(layout.pockets, radii)
            if pocket.area >= crumb and not isSliver(pocket) and not isStrip(pocket, radius)]
    if len(kept) == len(layout.pockets):
        return layout
    return Layout(graph=layout.graph, ribs=layout.ribs, pockets=kept, name=layout.name, ribWidths=layout.ribWidths)


def boundShare(plate, params, pocket, samples=48):
    """Share of a pocket's outline that runs along the wall or a feature's boss (not a rib)."""
    ring = pocket.exterior
    wallReach, bossReach = 1.25 * params.wallIn, 1.25 * params.boss
    near = 0
    for k in range(samples):
        point = ring.interpolate(k / samples, normalized=True)
        if plate.outline.exterior.distance(point) <= wallReach or any(
                feature.distance(point) <= bossReach for feature in plate.features):
            near += 1
    return near / samples


def mirrorClosure(plate, params, graph, layout, probabilities, segments, edgeMirror):
    """On symmetric plates, keep every rib's symmetry images too (repairs and splits are
    added one at a time and can drift). Orbits that would cross are dropped, weaker first,
    and floating holes are reconnected along whole orbits."""
    if not edgeMirror:
        return layout
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    orbits = {orbitOf(lookup[edge], edgeMirror) for edge in layout.ribs}

    def pick(orbits):
        kept = []
        for orbit in sorted(orbits, key=lambda orbit: (-max(probabilities[k] for k in orbit), orbit)):
            trial = []
            for k in orbit:
                if conflicts(graph, segments, lookup, graph.edges[k], kept + trial, params):
                    break
                trial.append(graph.edges[k])
            else:
                kept += trial
        return kept

    kept = pick(orbits)
    # Dropping a crossing orbit can strand a hole; reconnect it, then close that path too.
    repaired = connectFloatingFeatures(plate, params, graph, kept, probabilities, edgeMirror)
    if set(repaired) != set(kept):
        kept = pick(orbits | {orbitOf(lookup[edge], edgeMirror) for edge in repaired})
    if set(kept) == set(layout.ribs):
        return layout
    ribWidths = {}
    for edge in kept:
        orbit = orbitOf(lookup[edge], edgeMirror)
        known = [layout.ribWidths[graph.edges[k]] for k in orbit if graph.edges[k] in layout.ribWidths]
        ribWidths[edge] = max(known) if known else params.ribIn
    closed = makeLayout(plate, params, graph, sorted(kept), name="ml", ribWidths=ribWidths)
    if (validator.validateLayout(plate, params, closed.pockets, checkCorners=False).passed
            or not validator.validateLayout(plate, params, layout.pockets, checkCorners=False).passed):
        return closed
    return layout  # closing broke validity; keep the valid (slightly asymmetric) design


def elongation(pocket):
    rectangle = pocket.minimum_rotated_rectangle
    xs, ys = rectangle.exterior.coords.xy
    sides = sorted(math.dist((xs[k], ys[k]), (xs[k + 1], ys[k + 1])) for k in range(2))
    return sides[1] / max(sides[0], 1e-9)


def mergeSlotPockets(plate, params, graph, layout, probabilities, segments, edgeMirror=None, maxMerges=20):
    """Widen slots by dropping a rib beside them: the opposite of splitting.

    A pocket longer than maxElongation that splitting couldn't fix (a fan of long thin
    pockets, say) loses the least likely rib along its side, merging it with its
    neighbour, when that leaves the design valid, the merged pocket less slot-like, and
    no pocket past the size limit. Mirror images go with it on symmetric plates.
    Otherwise these slots would be filled in solid at the end."""
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    scale = math.sqrt(plate.materialArea)
    areaCap = min(maxPocketShare * plate.materialArea,
                  areaCapFactor * designFeatures.referencePocketArea(scale) * params.pocketSizeFactor)
    stuck = []
    for _ in range(maxMerges):
        slots = [pocket for pocket in layout.pockets if elongation(pocket) > maxElongation
                 and not any(pocket.equals_exact(done, 1e-6) for done in stuck)]
        if not slots:
            return layout
        slot = max(slots, key=elongation)
        beside = sorted((edge for edge in layout.ribs if segments[lookup[edge]].distance(slot) < params.ribIn),
                        key=lambda edge: probabilities[lookup[edge]])
        merged = None
        for edge in beside[:maxSplitAttempts + 1]:
            dropped = {graph.edges[k] for k in orbitOf(lookup[edge], edgeMirror)}
            candidate = makeLayout(plate, params, graph, sorted(set(layout.ribs) - dropped), name="ml",
                                   ribWidths={e: w for e, w in layout.ribWidths.items() if e not in dropped})
            grown = [pocket for pocket in candidate.pockets if pocket.intersects(slot.representative_point())]
            if not grown or elongation(grown[0]) >= elongation(slot) or grown[0].area > areaCap:
                continue
            if validator.validateLayout(plate, params, candidate.pockets, checkCorners=False).passed:
                merged = candidate
                break
        if merged is None:
            stuck.append(slot)
        else:
            layout = merged
    return layout


def splitNeed(plate, params, pocket):
    """How far a pocket is past the gold limits on size and elongation (>1 means split it)."""
    scale = math.sqrt(plate.materialArea)
    target = designFeatures.referencePocketArea(scale) * params.pocketSizeFactor
    areaCap = min(maxPocketShare * plate.materialArea, areaCapFactor * target)
    if pocket.area < 2.5 * params.minPocketArea:
        return 0.0  # halves wouldn't be worth cutting
    return max(pocket.area / areaCap, elongation(pocket) / maxElongation)


def splitQuality(pocket, bands):
    """How good a cut the bands make: even (smaller/larger piece) and compact pieces.

    0 when they don't actually split the pocket. Pieces longer than the gold limit are
    penalized, so a cut never trades one big pocket for two skinny triangles.
    """
    pieces = sorted(polygons(pocket.difference(unary_union(bands))), key=lambda piece: -piece.area)
    if len(pieces) < 2 or pieces[0].area <= 0:
        return 0.0
    balance = pieces[1].area / pieces[0].area
    worstElongation = max(elongation(piece) for piece in pieces[:2])
    return balance * min(1.0, maxElongation / worstElongation)


def splitOversizedPockets(plate, params, graph, layout, probabilities, segments, widths, edgeMirror=None, maxSplits=40):
    """Keep pockets inside the gold plates' limits on size and elongation.

    The worst pocket is split by a candidate rib that crosses it, scored by the model's keep
    probability times the cut quality (even, compact pieces), so the model still decides
    where ribs go. Single straight ribs come first; a two-rib path through a free junction
    is the fallback. A split is kept only if the design still validates, and mirror
    partners are added on symmetric plates.
    """
    lookup = {edge: k for k, edge in enumerate(graph.edges)}
    unsplittable = []
    for _ in range(maxSplits):
        needs = [(splitNeed(plate, params, pocket), pocket) for pocket in layout.pockets]
        needs = [(need, pocket) for need, pocket in needs
                 if need > 1.0 and not any(pocket.equals_exact(done, 1e-6) for done in unsplittable)]
        if not needs:
            return layout
        _, pocket = max(needs, key=lambda item: item[0])
        kept = set(layout.ribs)
        used = {node for rib in kept for node in rib}

        def attached(node):
            # A rib ending at an unused interior junction would be pruned as a dead end.
            return graph.nodes[node].kind != "interior" or node in used

        def crossesKept(k, edge):
            return conflicts(graph, segments, lookup, edge, kept, params)

        def band(k):
            return segments[k].buffer(params.ribIn / 2, cap_style="flat")

        options = []
        for k, edge in enumerate(graph.edges):
            if edge in kept or not (attached(edge[0]) and attached(edge[1])):
                continue
            if segments[k].intersection(pocket).length < 0.3 * segments[k].length or crossesKept(k, edge):
                continue
            quality = splitQuality(pocket, [band(k)])
            if quality > 0.15:
                options.append((float(probabilities[k]) * quality, (edge,)))
        if not options:
            for node, data in enumerate(graph.nodes):
                if data.kind != "interior" or node in used or not pocket.contains(Point(data.point)):
                    continue
                spokes = []
                for k, edge in enumerate(graph.edges):
                    if node not in edge:
                        continue
                    other = edge[0] if edge[1] == node else edge[1]
                    if attached(other) and not crossesKept(k, edge):
                        spokes.append((float(probabilities[k]), k, edge))
                spokes.sort(reverse=True)
                for first in range(len(spokes)):
                    for second in range(first + 1, len(spokes)):
                        a, b = spokes[first], spokes[second]
                        if segments[a[1]].crosses(segments[b[1]]) or tooAcute(graph, a[2], b[2], params.minRibAngleDegrees):
                            continue
                        quality = splitQuality(pocket, [band(a[1]), band(b[1])])
                        if quality > 0.15:
                            options.append(((a[0] + b[0]) / 2 * quality, (a[2], b[2])))
                        break
                    else:
                        continue
                    break
        withImages = []
        for score, added in sorted(options, key=lambda option: -option[0]):
            added = list(added)
            # Keep symmetric plates symmetric: split the image pockets the same way.
            for edge in list(added):
                for k in orbitOf(lookup[edge], edgeMirror):
                    if graph.edges[k] not in kept and graph.edges[k] not in added:
                        added.append(graph.edges[k])
            if not any(conflicts(graph, segments, lookup, edge, [other for other in kept | set(added) if other != edge], params)
                       for edge in added):
                withImages.append(added)
        accepted = None
        for added in withImages[:maxSplitAttempts]:
            ribWidths = dict(layout.ribWidths)
            for edge in added:
                ribWidths[edge] = params.ribIn * ribWidthClasses[int(np.argmax(widths[lookup[edge]]))]
            candidate = makeLayout(plate, params, graph, sorted(kept | set(added)), name="ml", ribWidths=ribWidths)
            # A split must never turn a valid design into an invalid one.
            if validator.validateLayout(plate, params, candidate.pockets, checkCorners=False).passed:
                accepted = candidate
                break
        if accepted is None:
            unsplittable.append(pocket)
        else:
            layout = accepted
    return layout


class RibGenerator:
    """Loads trained weights and proposes layouts for a plate."""

    def __init__(self, path=None, model=None):
        self.model = model or RibGnn()
        if path is not None:
            self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def prepare(self, plate, params):
        graph = defaultGraph(plate, params)
        tensors = encodeGraph(plate, params, graph)
        return graph, tensors, mirrorMap(plate, graph)

    def logits(self, tensors):
        with torch.no_grad():
            return self.model(tensors)

    def propose(self, plate, params, count=8, seed=0, temperatures=(0.6, 1.0, 1.4), prepared=None,
                densities=densifyFractions):
        """Most-likely layout, sampled layouts, and densified variants of the most likely one."""
        graph, tensors, edgeMirror = prepared or self.prepare(plate, params)
        if len(graph.edges) == 0:
            return []
        keepLogits, widthLogits = self.logits(tensors)
        rng = np.random.default_rng(seed)
        layouts = [decode(plate, params, graph, keepLogits, widthLogits, greedy=True, edgeMirror=edgeMirror)]
        for fraction in densities:
            layouts.append(densify(plate, params, graph, keepLogits, widthLogits, edgeMirror, fraction))
        for index in range(max(0, count - 1 - len(densities))):
            temperature = temperatures[index % len(temperatures)]
            layouts.append(decode(plate, params, graph, keepLogits, widthLogits, rng, temperature, edgeMirror=edgeMirror))
        return layouts


def densify(plate, params, graph, keepLogits, widthLogits, edgeMirror, fraction):
    """The most likely layout plus the model's next most likely triangulation ribs.

    Adds the top `fraction` of planar (Delaunay) candidates by keep probability on top
    of everything already above 0.5: a stiffer variant built from the model's own ranking.
    """
    planar = [k for k, isPlanar in enumerate(graph.isDelaunay) if isPlanar]
    if not planar:
        return decode(plate, params, graph, keepLogits, widthLogits, greedy=True, edgeMirror=edgeMirror)
    order = sorted(planar, key=lambda k: -float(keepLogits[k]))
    boosted = keepLogits.clone()
    for k in order[: max(1, int(round(fraction * len(planar))))]:
        boosted[k] = max(float(boosted[k]), 10.0)
    return decode(plate, params, graph, boosted, widthLogits, greedy=True, edgeMirror=edgeMirror)
