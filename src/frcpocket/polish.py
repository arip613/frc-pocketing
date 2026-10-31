"""Deterministic polish pass for ML designs: runs after the generator picks ribs and
before the final validation and export.

The generator's designs are judged by eye against real FRC plates, and four things
stood out (user feedback, comparing real | generated plates):

  symmetry  a mirror-symmetric plate should get a mirror-symmetric rib set
  links     bosses that face each other across a pocket are usually tied by a straight
            hole-to-hole rib (bearing to bearing, bearing to bolt)
  blobs     big solid regions with no pocket ruin the look; they are opened up and
            re-split into pockets of the plate's usual size
  split     pockets much bigger than the design's typical pocket are split by ribs
  huggers   ribs running just beside a wall or big cutout are dropped, so the pocket next
            to them reaches the wall instead of stopping short of it
            (a single rib, or an X of two crossing ribs in a large squarish or
            triangular pocket)

Every rib the pass adds is a straight segment realized in the layout's candidate graph
(snapped to an existing node, or a new wall point, a T-junction on an existing rib, or a
free junction), so pockets are still built by layout.makeLayout and the STEP export
rebuilds them exactly. Each step is checked with the full validator (and, when the
caller passes one, the structural floor); a step that fails either is reverted.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from frcpocket import designFeatures, geom2d, validator
from frcpocket.layout import (CandidateGraph, Layout, Node, arcSegments, filletRadius, makeLayout, mirrorGeometry,
                              plateSymmetries, polygons, toolOpening, wallCenterline)
from frcpocket.plate import featureCenter, isOpening

enabled = True  # the shared generation path (Pocketer, rating sheets) polishes ML designs when on

# Pocket size: "typical" is the design's own median pocket, held near the gold size.
typicalClip = (0.7, 1.3)  # typical pocket area, in gold reference areas for this plate size
splitCapFactor = 1.8  # pockets above this many typical pockets get split
splitElongation = 3.0  # and long pockets above this elongation, if at least a typical pocket big
minPieceFactor = 0.3  # no link leaves a piece below this many typical pockets
splitPieceFactor = 0.45  # and no split or X a piece below this many
crossMinFactor = 3.0  # X-bracing only in pockets at least this many typical pockets big
crossMaxElongation = 2.2  # and only in squarish / triangular ones
blobMinFactor = 0.15  # solid regions at least this many typical pockets big count as blobs
linkMaxGapFactor = 2.2  # facing bosses are linked when their gap is under this many typical pocket widths
linkMinCornerDegrees = 25.0  # ... and the pieces it leaves have no wedge corner sharper than this
linkPocketShare = 0.6  # ... and at least this share of the gap runs through pockets
minMeetDegrees = 30.0  # a new rib meets any rib, or the wall, at least this wide
minMeetAtHoleDegrees = 20.0  # ribs leave a boss rim, so they may fan out more tightly there
maxRounds = 12  # per step
hugShare = 0.6  # a rib hugs a wall or cutout when this share of it runs too close to pocket the gap
hugTries = 4  # fully check only the few drops that gain the most pocket


@dataclass
class PolishResult:
    layout: Layout
    log: list = field(default_factory=list)
    changed: bool = False


class Design:
    """A rib set on its own copy of the candidate graph (polish may add nodes and edges)."""

    def __init__(self, graph, ribs, ribWidths):
        self.graph = CandidateGraph(nodes=list(graph.nodes), edges=list(graph.edges),
                                    isDelaunay=list(graph.isDelaunay), junctionStep=graph.junctionStep)
        self.ribs = sorted(set(ribs))
        self.ribWidths = {edge: width for edge, width in ribWidths.items() if edge in set(self.ribs)}

    def copy(self):
        return Design(self.graph, self.ribs, self.ribWidths)

    def segment(self, edge):
        return self.graph.segment(edge)

    def addEdge(self, i, j, width=None):
        edge = tuple(sorted((i, j)))
        if edge not in set(self.graph.edges):
            self.graph.edges.append(edge)
            self.graph.isDelaunay.append(False)
        if edge not in set(self.ribs):
            self.ribs = sorted(set(self.ribs) | {edge})
        if width is not None:
            self.ribWidths[edge] = width
        return edge

    def removeRib(self, edge):
        self.ribs = [rib for rib in self.ribs if rib != edge]
        self.ribWidths.pop(edge, None)


class Polisher:
    def __init__(self, plate, params, layout, structuralCheck=None):
        self.plate, self.params = plate, params
        self.structuralCheck = structuralCheck
        self.scale = math.sqrt(plate.materialArea)
        reference = designFeatures.referencePocketArea(self.scale) * params.pocketSizeFactor
        areas = [pocket.area for pocket in layout.pockets]
        median = float(np.median(areas)) if areas else reference
        # The design's own median pocket, pulled halfway (in log) toward the gold size.
        self.typical = float(np.clip(math.sqrt(median * reference), typicalClip[0] * reference, typicalClip[1] * reference))
        self.typicalWidth = math.sqrt(self.typical)
        self.minPiece = max(minPieceFactor * self.typical, 2.5 * params.minPocketArea)
        self.minSplitPiece = max(splitPieceFactor * self.typical, 2.5 * params.minPocketArea)
        self.inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
        self.rings = wallCenterline(plate, params)
        self.bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
        self.preparedFeatures = [prep(feature) for feature in plate.features]
        self.insideOutline = prep(plate.outline.buffer(1e-6))
        self.snap = max(0.5 * params.ribIn, 0.012 * self.scale)
        self.nodeSnap = 0.04 * self.scale  # same tolerance the generator's mirror map uses
        self.minMeet = max(minMeetDegrees, params.minRibAngleDegrees)
        self.transforms = plateSymmetries(plate)
        self.centre = plate.outline.centroid
        self.angle = geom2d.principalAngle(plate.outline)
        self.cache = {}
        self.rawCache = {}
        self.strictExport = True  # off only when the input design itself needed gap trimming
        self.orbitCache = {}
        self.linesCache = {}

    # ---------- building and checking -------------------------------------------------

    def key(self, design):
        """The rib set by position (graph copies may number new nodes differently)."""
        nodes = design.graph.nodes

        def at(index):
            return (round(nodes[index].x, 5), round(nodes[index].y, 5))

        return tuple(sorted(tuple(sorted((at(i), at(j)))) + (round(design.ribWidths.get((i, j), self.params.ribIn), 5),)
                            for i, j in design.ribs))

    def rawLayout(self, design):
        """The finished layout without validating it (cheap; for ranking trial moves)."""
        from frcpocket.generator import fillJunkPockets

        key = self.key(design)
        if key in self.cache:
            return self.cache[key][0]
        if key not in self.rawCache:
            layout = makeLayout(self.plate, self.params, design.graph, design.ribs, name="ml", ribWidths=design.ribWidths)
            self.rawCache[key] = fillJunkPockets(self.plate, self.params, layout)
        return self.rawCache[key]

    def build(self, design):
        """Layout and validity, the same way the generator finishes a design."""
        key = self.key(design)
        if key not in self.cache:
            layout = self.rawLayout(design)
            ok = bool(layout.pockets) and validator.validateLayout(self.plate, self.params, layout.pockets).passed
            ok = ok and (not self.strictExport or self.exportMatches(design, layout))
            self.cache[key] = (layout, ok)
        layout, ok = self.cache[key]
        if len(layout.ribs) != len(design.ribs):
            design.ribs = list(layout.ribs)  # dead ends pruned
            design.ribWidths = {edge: width for edge, width in design.ribWidths.items() if edge in set(design.ribs)}
        return layout, ok

    def exportMatches(self, design, layout):
        """The STEP export rebuilds pockets without the 2D gap trimming (layout.enforcePocketGaps),
        so a design is only taken if no trimming was needed: the untrimmed tool-opened pockets
        it keeps are already a rib apart."""
        params = self.params
        keep = list(self.bosses) + [design.segment(rib).buffer(self.width(design, rib) / 2, cap_style="flat") for rib in design.ribs]
        raw = self.inner.difference(unary_union(keep)) if keep else self.inner
        untrimmed = []
        for region in polygons(raw):
            opened = toolOpening(region, filletRadius(region, params))
            for piece in polygons(opened.intersection(region)):
                if any(pocket.contains(piece.representative_point()) for pocket in layout.pockets):
                    untrimmed.append(piece)
        return validator.minPairDistance(untrimmed, 2 * params.ribIn) >= params.ribIn - validator.lengthTolIn

    def width(self, design, edge):
        return self.params.ribWidthFor(design.ribWidths.get(edge))

    # ---------- geometry helpers -------------------------------------------------------

    def mirror(self, geometry, transform):
        return mirrorGeometry(geometry, self.centre, self.angle, transform)

    def orbit(self, segment):
        """The segment and its images under the plate's symmetries (no near-duplicates)."""
        if not self.transforms:
            return [segment]
        key = tuple(np.round(np.ravel(segment.coords), 6))
        if key in self.orbitCache:
            return self.orbitCache[key]
        members = [segment]
        for _ in range(2):
            for transform in self.transforms:
                for member in list(members):
                    image = self.mirror(member, transform)
                    if not any(sameSegment(image, other, self.nodeSnap) for other in members):
                        members.append(image)
        self.orbitCache[key] = members
        return members

    def featureNode(self, design, featureIndex):
        for index, node in enumerate(design.graph.nodes):
            if node.kind == "feature" and node.featureIndex == featureIndex:
                return index
        return None

    def realize(self, design, point):
        """Node index at `point`, creating a wall point, rib T-junction or free junction."""
        point = Point(point)
        for featureIndex, feature in enumerate(self.plate.features):
            if feature.buffer(self.snap).contains(point):
                index = self.featureNode(design, featureIndex)
                if index is not None:
                    return index
        nodes = design.graph.nodes
        used = {node for rib in design.ribs for node in rib}
        best, bestDistance = None, self.snap
        for index, node in enumerate(nodes):
            if node.kind == "feature":
                continue
            distance = math.dist(node.point, point.coords[0])
            limit = self.nodeSnap if (node.kind == "anchor" or index in used) else self.snap
            if distance < min(limit, bestDistance if best is not None else limit):
                best, bestDistance = index, distance
        if best is not None:
            return best
        for ring in self.rings:
            if ring.distance(point) < self.snap:
                onRing = ring.interpolate(ring.project(point))
                nodes.append(Node(onRing.x, onRing.y, kind="anchor"))
                return len(nodes) - 1
        for rib in list(design.ribs):
            segment = design.segment(rib)
            if segment.distance(point) < self.snap:
                t = segment.project(point, normalized=True)
                if 0.05 < t < 0.95:
                    onRib = segment.interpolate(t, normalized=True)
                    nodes.append(Node(onRib.x, onRib.y, kind="interior"))
                    middle = len(nodes) - 1
                    width = design.ribWidths.get(rib)
                    design.removeRib(rib)
                    design.addEdge(rib[0], middle, width)
                    design.addEdge(middle, rib[1], width)
                    return middle
        nodes.append(Node(point.x, point.y, kind="interior"))
        return len(nodes) - 1

    def addSegments(self, design, segments, width=None):
        for segment in segments:
            a, b = segment.coords[0], segment.coords[-1]
            i = self.realize(design, a)
            j = self.realize(design, b)
            if i != j:
                design.addEdge(i, j, width)
        return design

    def ribLines(self, design):
        return [design.segment(rib) for rib in design.ribs]

    def directionsAt(self, design, point, port):
        """Unit directions of the ribs (and wall) already meeting at a port."""
        kind, ref = port
        origin = np.array(point)
        directions = []

        def towards(other):
            vector = np.array(other) - origin
            length = np.hypot(*vector)
            if length > 1e-9:
                directions.append(vector / length)

        if kind in ("feature", "node"):
            for i, j in design.ribs:
                if ref in (i, j):
                    towards(design.graph.nodes[j if i == ref else i].point)
        if kind == "rib":
            segment = design.segment(ref)
            towards(segment.coords[0])
            towards(segment.coords[-1])
        if kind == "wall" or (kind == "node" and design.graph.nodes[ref].kind == "anchor"):
            ring = min(self.rings, key=lambda candidate: candidate.distance(Point(point)))
            position = ring.project(Point(point))
            step = 0.05 * self.typicalWidth
            towards(ring.interpolate((position + step) % ring.length).coords[0])
            towards(ring.interpolate((position - step) % ring.length).coords[0])
        return directions

    def meetsWide(self, design, start, end, port):
        vector = np.array(end) - np.array(start)
        vector = vector / max(np.hypot(*vector), 1e-12)
        degrees = max(minMeetAtHoleDegrees, self.params.minRibAngleDegrees) if port[0] == "feature" else self.minMeet
        limit = math.cos(math.radians(degrees))
        return all(float(vector @ direction) < limit for direction in self.directionsAt(design, start, port))

    def segmentClear(self, design, segment, endFeatures, partners=()):
        """Inside the plate, crossing no hole it doesn't end at and no kept rib (except partners)."""
        if segment.length < 1e-6 or not self.insideOutline.contains(segment):
            return False
        trimmed = trimSegment(segment, min(self.snap, 0.2 * segment.length))
        for index, feature in enumerate(self.plate.features):
            if index in endFeatures:
                continue
            if self.preparedFeatures[index].intersects(trimmed):
                return False
            if feature.distance(trimmed) < 0.5 * self.params.boss:
                return False
        lines, tree = self.ribTree(design)
        for index in tree.query(trimmed):
            line = lines[index]
            if line.intersects(trimmed) and not any(line.equals(partner) for partner in partners):
                return False
        return True

    def ribTree(self, design):
        """Rib centre lines and a search tree over them (cached per rib set)."""
        from shapely.strtree import STRtree

        key = (tuple(design.ribs), len(design.graph.nodes), id(design.graph.nodes))
        if key not in self.linesCache:
            lines = self.ribLines(design)
            self.linesCache[key] = (lines, STRtree(lines))
        return self.linesCache[key]

    def pieceCost(self, pieces):
        """How far pieces are from the typical pocket (size, elongation, wedge corners)."""
        cost = 0.0
        for piece in pieces:
            cost += abs(math.log(piece.area / self.typical))
            cost += 0.6 * max(0.0, elongationOf(piece) - 2.4)
            if designFeatures.sharpestCorner(piece) < 35:
                cost += 0.4
        return cost / max(len(pieces), 1)

    def splitPieces(self, pocket, segments, width):
        bands = unary_union([segment.buffer(width / 2 + 0.5 * self.params.cornerRadiusIn, cap_style="flat") for segment in segments])
        return [piece for piece in polygons(pocket.difference(bands)) if piece.area > 1e-4]

    def reconnect(self, design):
        """Tie any hole left floating back to the network by its shortest candidate path."""
        from frcpocket.generator import connectFloatingFeatures

        lengths = np.array([design.segment(edge).length for edge in design.graph.edges])
        likelihood = np.exp(-lengths / max(self.typicalWidth, 1e-6))  # shorter ribs first
        ribs = connectFloatingFeatures(self.plate, self.params, design.graph, set(design.ribs), likelihood)
        design.ribs = sorted(set(ribs))
        return design

    # ---------- measurements ----------------------------------------------------------

    def symmetryShortfall(self, pockets):
        if not self.transforms or not pockets:
            return 0.0
        union = unary_union(pockets)
        return max(1.0 - geom2d.iou(union, self.mirror(union, transform)) for transform in self.transforms)

    def blobs(self, design, layout):
        """Solid regions inside the wall that no pocket comes near.

        The inner region minus every pocket grown by the rib and every boss ring, opened by
        half the minimum pocket width so that only regions a pocket could fill remain (thick
        ribs and soft corners fall away in the opening). A tangle of ribs with no pockets
        between them counts, and so do thin bands left along a wall or cutout by a rib that
        runs beside it."""
        params = self.params
        taken = [pocket.buffer(params.ribIn, quad_segs=8) for pocket in layout.pockets]
        taken += [feature.buffer(params.ringFor(feature), quad_segs=8) for feature in self.plate.features]
        free = self.inner.difference(unary_union(taken)) if taken else self.inner
        radius = 0.5 * params.minPocketWidth
        opened = free.buffer(-radius, quad_segs=8).buffer(radius, quad_segs=8)
        return [region for region in polygons(opened) if region.area >= blobMinFactor * self.typical]

    def blobArea(self, design, layout):
        return sum(region.area for region in self.blobs(design, layout))

    def overCap(self, pocket):
        cap = min(0.15 * self.plate.materialArea, splitCapFactor * self.typical)
        need = pocket.area / cap
        if pocket.area > self.typical:
            need = max(need, elongationOf(pocket) / splitElongation)
        return need

    def slots(self, layout):
        """Pockets longer than the gold limit (thin wedges and slots)."""
        return sum(1 for pocket in layout.pockets if elongationOf(pocket) > splitElongation)

    def crumbs(self, layout):
        return sum(1 for pocket in layout.pockets if pocket.area < self.minPiece)

    # ---------- acceptance ------------------------------------------------------------

    def keepsPockets(self, before, after, share=0.6):
        """Every pocket of the old design is still mostly pocket (no region left solid)."""
        if not after.pockets:
            return not before.pockets
        union = unary_union(after.pockets)
        return all(pocket.intersection(union).area >= share * pocket.area for pocket in before.pockets)

    def accept(self, design, layout, previous):
        """Valid, and not worse on the things the other steps fix."""
        built, ok = self.build(design)
        if not ok:
            return False
        if self.crumbs(built) > self.crumbs(previous[1]):
            return False
        if not self.keepsPockets(previous[1], built):
            return False
        if self.slots(built) > self.slots(previous[1]):
            return False
        if self.blobArea(design, built) > self.blobArea(*previous) + 0.25 * self.typical:
            return False
        if self.transforms and self.symmetryShortfall(built.pockets) > self.symmetryShortfall(previous[1].pockets) + 0.02:
            return False
        return True

    # ---------- steps -----------------------------------------------------------------

    def symmetrize(self, design, layout):
        """Give every rib its mirror image, or drop ribs whose image can't be added."""
        if not self.transforms:
            return design, layout, ""
        before = self.symmetryShortfall(layout.pockets)
        if before < 0.01:
            return design, layout, ""
        lines = self.ribLines(design)
        covered = unary_union(lines).buffer(self.nodeSnap) if lines else None
        missing = []
        for rib, line in zip(design.ribs, lines):
            images = [image for image in self.orbit(line)[1:]
                      if covered is None or image.difference(covered).length > 0.15 * image.length]
            if images:
                missing.append((rib, images))
        options = []
        if missing:
            added = design.copy()
            for rib, images in missing:
                self.addSegments(added, images, design.ribWidths.get(rib))
            options.append(("added mirror images", added))
            removed = design.copy()
            for rib, _ in missing:
                removed.removeRib(rib)
            options.append(("dropped unmatched ribs", removed))
            # Per rib: add its images where they fit, else drop it.
            mixed = design.copy()
            for rib, images in missing:
                trial = self.addSegments(mixed.copy(), images, design.ribWidths.get(rib))
                if self.build(trial)[1]:
                    mixed = trial
                else:
                    trial = mixed.copy()
                    trial.removeRib(rib)
                    if self.build(trial)[1]:
                        mixed = trial
            options.append(("mixed", mixed))
        best = None
        for name, option in options:
            built, ok = self.build(option)
            if not ok or self.crumbs(built) > self.crumbs(layout) or self.slots(built) > self.slots(layout):
                continue
            shortfall = self.symmetryShortfall(built.pockets)
            if shortfall < before - 0.01 and (best is None or shortfall < best[0]):
                best = (shortfall, name, option, built)
        if best is None:
            return design, layout, ""
        return best[2], best[3], f"symmetry {before:.2f}->{best[0]:.2f} ({best[1]})"

    def linkBosses(self, design, layout):
        """Straight ribs between bosses that face each other across a pocket."""
        plate, params = self.plate, self.params
        notes = []
        pairs = []
        centres = [featureCenter(feature) for feature in plate.features]
        for a in range(len(plate.features)):
            for b in range(a + 1, len(plate.features)):
                segment = LineString([centres[a].coords[0], centres[b].coords[0]])
                gap = segment.difference(self.bosses[a]).difference(self.bosses[b])
                if gap.is_empty or gap.length < 1.5 * params.ribIn:
                    continue
                if gap.length > linkMaxGapFactor * self.typicalWidth:
                    continue
                pairs.append((gap.length, a, b, segment, gap))
        pairs.sort(key=lambda item: item[0])
        done = set()
        for _, a, b, segment, gap in pairs:
            if (a, b) in done:
                continue
            na, nb = self.featureNode(design, a), self.featureNode(design, b)
            if na is None or nb is None or tuple(sorted((na, nb))) in set(design.ribs):
                continue
            pocketUnion = unary_union(layout.pockets) if layout.pockets else None
            if pocketUnion is None or gap.intersection(pocketUnion).length < linkPocketShare * gap.length:
                continue
            if not self.segmentClear(design, segment, {a, b}):
                continue
            if not (self.meetsWide(design, segment.coords[0], segment.coords[-1], ("feature", na))
                    and self.meetsWide(design, segment.coords[-1], segment.coords[0], ("feature", nb))):
                continue
            if params.maxHoleRibs and any(
                    plate.features[f].area < math.pi * 0.225 ** 2 and sum(1 for rib in design.ribs if n in rib) >= params.maxHoleRibs
                    for f, n in ((a, na), (b, nb))):
                continue
            # The pocket it crosses must split into pieces worth cutting.
            crossed = [pocket for pocket in layout.pockets if pocket.intersects(gap)]
            pieces = [piece for pocket in crossed for piece in self.splitPieces(pocket, [segment], params.ribIn)]
            if any(piece.area < self.minPiece or elongationOf(piece) > splitElongation
                   or designFeatures.sharpestCorner(piece) < linkMinCornerDegrees for piece in pieces):
                continue  # would leave a crumb, a slot or a thin wedge
            trial = self.addSegments(design.copy(), self.orbit(segment), params.ribIn)
            built, _ = self.build(trial)
            if self.accept(trial, built, (design, layout)):
                for image in self.orbit(segment):
                    ends = [index for index, centre in enumerate(centres)
                            if centre.distance(Point(image.coords[0])) < self.snap or centre.distance(Point(image.coords[-1])) < self.snap]
                    if len(ends) == 2:
                        done.add(tuple(sorted(ends)))
                design, layout = trial, built
                notes.append(f"{a}-{b}")
        return design, layout, f"linked bosses {', '.join(notes)}" if notes else ""

    def ports(self, design, pocket):
        """Points a splitting rib may end at, around one pocket: (point, (kind, ref))."""
        params = self.params
        found = []
        used = {node for rib in design.ribs for node in rib}
        for featureIndex, boss in enumerate(self.bosses):
            if boss.distance(pocket) < 1.5 * params.ribIn + params.cornerRadiusIn:
                index = self.featureNode(design, featureIndex)
                if index is not None:
                    found.append((design.graph.nodes[index].point, ("feature", index)))
        for index, node in enumerate(design.graph.nodes):
            if node.kind == "anchor" and Point(node.point).distance(pocket) < params.wallIn / 2 + params.ribIn + params.maxFilletIn:
                found.append((node.point, ("node", index)))
            elif node.kind == "interior" and index in used and Point(node.point).distance(pocket) < 2 * params.ribIn + params.maxFilletIn:
                found.append((node.point, ("node", index)))
        step = 0.35 * self.typicalWidth
        for ring in self.rings:
            if ring.distance(pocket) > params.wallIn / 2 + params.ribIn + params.maxFilletIn:
                continue
            count = max(4, int(ring.length / step))
            for k in range(count):
                point = ring.interpolate(k * ring.length / count)
                if point.distance(pocket) < params.wallIn / 2 + 0.5 * params.ribIn + 0.5 * params.maxFilletIn:
                    if all(math.dist(point.coords[0], other) > 0.5 * step for other, _ in found):
                        found.append((point.coords[0], ("wall", None)))
        for rib in design.ribs:
            segment = design.segment(rib)
            if segment.distance(pocket) > self.width(design, rib) / 2 + params.maxFilletIn + params.cornerRadiusIn:
                continue
            for t in (0.33, 0.5, 0.67):
                point = segment.interpolate(t, normalized=True)
                if any(boss.buffer(params.ribIn).contains(point) for boss in self.bosses):
                    continue
                if point.distance(pocket) < self.width(design, rib) / 2 + params.maxFilletIn + params.cornerRadiusIn:
                    found.append((point.coords[0], ("rib", rib)))
        # Mirror wall points too, so a cut and its mirror image can land on the same points.
        for point, port in list(found):
            if port[0] != "wall":
                continue
            for transform in self.transforms:
                image = self.mirror(Point(point), transform)
                ring = min(self.rings, key=lambda candidate: candidate.distance(image))
                image = ring.interpolate(ring.project(image))
                if image.distance(pocket) < params.wallIn / 2 + 0.5 * params.ribIn + 0.5 * params.maxFilletIn and all(
                        math.dist(image.coords[0], other) > self.snap for other, _ in found):
                    found.append((image.coords[0], ("wall", None)))
        return found

    def splitOptions(self, design, pocket, allowCross):
        """Candidate cuts of one pocket, cheapest first: (cost, [segments], label)."""
        params = self.params
        ports = self.ports(design, pocket)
        singles = []
        for x in range(len(ports)):
            for y in range(x + 1, len(ports)):
                (p, portP), (q, portQ) = ports[x], ports[y]
                if portP[0] == "rib" and portQ[0] == "rib" and portP[1] == portQ[1]:
                    continue
                segment = LineString([p, q])
                if segment.length < 2 * params.ribIn:
                    continue
                inside = segment.intersection(pocket).length
                if inside < 0.5 * params.minPocketWidth:
                    continue
                endFeatures = {design.graph.nodes[port[1]].featureIndex for port in (portP, portQ) if port[0] == "feature"}
                outsideBosses = segment
                for featureIndex in endFeatures:
                    outsideBosses = outsideBosses.difference(self.bosses[featureIndex])
                if inside < 0.6 * outsideBosses.length:
                    continue  # runs mostly through material or other pockets
                if not self.segmentClear(design, segment, endFeatures):
                    continue
                if not (self.meetsWide(design, p, q, portP) and self.meetsWide(design, q, p, portQ)):
                    continue
                pieces = self.splitPieces(pocket, self.orbit(segment), params.ribIn)
                if len(pieces) < 2 or min(piece.area for piece in pieces) < self.minSplitPiece:
                    continue
                bonus = 0.12 * sum(port[0] == "feature" for port in (portP, portQ))
                penalty = 0.06 * sum(port[0] in ("rib", "wall") for port in (portP, portQ))
                cost = self.pieceCost(pieces) - bonus + penalty + 0.05 * segment.length / self.typicalWidth
                singles.append((cost, [segment], "rib", (portP, portQ)))
        options = list(singles)
        if allowCross and pocket.area >= crossMinFactor * self.typical and elongationOf(pocket) <= crossMaxElongation:
            ranked = sorted(singles, key=lambda option: option[0])[:25]
            for x in range(len(ranked)):
                for y in range(x + 1, len(ranked)):
                    first, second = ranked[x][1][0], ranked[y][1][0]
                    if not first.crosses(second):
                        continue
                    crossing = first.intersection(second)
                    if crossing.geom_type != "Point" or not pocket.buffer(-params.ribIn).contains(crossing):
                        continue
                    if crossingAngle(first, second) < 50:
                        continue
                    pieces = self.splitPieces(pocket, self.orbit(first) + self.orbit(second), params.ribIn)
                    if len(pieces) < 4 or min(piece.area for piece in pieces) < self.minSplitPiece:
                        continue
                    cost = self.pieceCost(pieces) - 0.1 + 0.05 * (first.length + second.length) / self.typicalWidth
                    options.append((cost, [first, second], "X", None))
        return sorted(options, key=lambda option: option[0])

    def splitLarge(self, design, layout, allowCross=True, region=None, rounds=maxRounds):
        """Split pockets past the typical-size cap (or very long ones) with ribs."""
        notes = []
        stuck = []
        for _ in range(rounds):
            candidates = [pocket for pocket in layout.pockets if self.overCap(pocket) > 1.0
                          and (region is None or pocket.intersects(region))
                          and not any(pocket.equals_exact(done, 1e-6) for done in stuck)]
            if not candidates:
                break
            pocket = max(candidates, key=self.overCap)
            accepted = False
            for cost, segments, label, _ in self.splitOptions(design, pocket, allowCross)[:4]:
                images = [image for segment in segments for image in self.orbit(segment)]
                trial = self.addSegments(design.copy(), images, self.params.ribIn)
                built, _ = self.build(trial)
                if not self.accept(trial, built, (design, layout)):
                    continue
                # The split must actually shrink this pocket.
                remaining = [other for other in built.pockets if other.intersects(pocket.representative_point())]
                if remaining and remaining[0].area > 0.8 * pocket.area:
                    continue
                design, layout, accepted = trial, built, True
                notes.append(f"{label} in {pocket.area:.2f}in2")
                break
            if not accepted:
                stuck.append(pocket)
        return design, layout, f"split {', '.join(notes)}" if notes else ""

    def fillBlobs(self, design, layout):
        """Open up solid blobs: drop or slim a rib beside them, then re-split to the usual size."""
        notes = []
        tried = set()
        for _ in range(maxRounds):
            blobs = [blob for blob in self.blobs(design, layout) if blob.wkb not in tried]
            if not blobs:
                break
            blob = max(blobs, key=lambda region: region.area)
            tried.add(blob.wkb)
            current = self.blobArea(design, layout)
            reach = blob.buffer(2 * self.params.ribIn + self.params.maxFilletIn)
            moves = []
            seen = set()
            insideAll = []
            segments = {rib: design.segment(rib) for rib in design.ribs}
            for rib in design.ribs:
                if not segments[rib].intersects(reach):
                    continue
                orbit = self.orbit(segments[rib])
                group = [other for other in design.ribs if any(sameSegment(segments[other], image, self.nodeSnap) for image in orbit)]
                key = tuple(sorted(group))
                if key in seen:
                    continue
                seen.add(key)
                moves.append(("dropped", group))
                if any(segments[other].intersection(reach).length >= 0.5 * segments[other].length for other in group):
                    insideAll.extend(group)  # with its mirror images, so the plate stays symmetric
                if any(self.width(design, other) > self.params.ribIn * 1.05 for other in group):
                    moves.append(("slimmed", group))
            # Single-rib moves nearest the blob first (a tangle has many; a few are enough).
            centre = blob.representative_point()
            moves = sorted(moves, key=lambda move: min(design.segment(rib).distance(centre) for rib in move[1]))[:10]
            # A tangle of ribs: clear every rib lying in the blob (keeping the ones that hold
            # a hole), then re-split the opened region to the usual pocket size.
            if insideAll:
                featureNodes = {index for index, node in enumerate(design.graph.nodes) if node.kind == "feature"}
                moves.append(("cleared", sorted(set(insideAll))))
                kept = sorted({rib for rib in insideAll if not (set(rib) & featureNodes)})
                if kept:
                    moves.append(("cleared", kept))
            # Cheap pass first (rebuild only), then re-split the few most promising.
            opened = []
            for kind, group in moves:
                trial = design.copy()
                for rib in group:
                    if kind in ("dropped", "cleared"):
                        trial.removeRib(rib)
                    else:
                        trial.ribWidths[rib] = self.params.ribIn
                built, ok = self.build(trial)
                if not ok and kind == "cleared":
                    trial = self.reconnect(trial)
                    built, ok = self.build(trial)
                if not ok:
                    continue
                area = self.blobArea(trial, built)
                if area <= current - 0.4 * blob.area:
                    opened.append((area, len(group), kind, trial, built))
            worstBefore = max([self.overCap(pocket) for pocket in layout.pockets if pocket.intersects(reach)] or [0.0])
            best = None
            for _, _, kind, trial, built in sorted(opened, key=lambda item: item[:2])[:3]:
                trial, built, _ = self.splitLarge(trial, built, region=reach, rounds=2 * maxRounds)
                area = self.blobArea(trial, built)
                if area > current - 0.4 * blob.area:
                    continue
                if self.crumbs(built) > self.crumbs(layout) or not self.build(trial)[1]:
                    continue
                if self.transforms and self.symmetryShortfall(built.pockets) > self.symmetryShortfall(layout.pockets) + 0.02:
                    continue
                worst = max([self.overCap(pocket) for pocket in built.pockets if pocket.intersects(reach)] or [0.0])
                if worst > max(1.0, worstBefore):
                    continue
                if best is None or area < best[0]:
                    best = (area, kind, trial, built)
            if best is not None:
                notes.append(f"{best[1]} rib by {blob.area:.2f}in2 blob")
                design, layout = best[2], best[3]
        return design, layout, f"blobs: {', '.join(notes)}" if notes else ""

    def dropHuggers(self, design, layout):
        """Drop ribs that run just beside the wall or a big cutout.

        Such a rib leaves a band between itself and the wall's (or cutout ring's) edge that
        is too thin to be a pocket of its own, so it stays solid. Dropping the rib lets the
        pocket beside it run up to the edge. Kept only if the plate gains pocket area and
        no pocket ends up oversized, crumbled or asymmetric."""
        params = self.params
        boundaries = [(self.plate.outline.exterior, params.wallIn)]
        boundaries += [(feature.exterior, params.ringFor(feature)) for feature in self.plate.features if isOpening(feature)]
        notes = []
        for _ in range(maxRounds):
            segments = {rib: design.segment(rib) for rib in design.ribs}
            candidates = []
            for rib, segment in segments.items():
                if segment.length < 2 * params.minPocketWidth:
                    continue
                points = [segment.interpolate(t, normalized=True) for t in np.linspace(0, 1, 9)]
                for ring, keep in boundaries:
                    distances = [ring.distance(point) for point in points]
                    band = keep + 0.5 * self.width(design, rib)
                    # along most of the rib the strip to the edge is too thin to pocket, none
                    # of it is far off, and somewhere it sticks out past the ring (else it
                    # costs nothing)
                    close = np.mean([d < band + params.minPocketWidth for d in distances])
                    if (close >= hugShare and max(distances) < band + 2 * params.minPocketWidth
                            and max(distances) > band + validator.lengthTolIn):
                        candidates.append(rib)
                        break
            pocketed = sum(pocket.area for pocket in layout.pockets)
            worstBefore = max([self.overCap(pocket) for pocket in layout.pockets] or [0.0])
            best = None
            seen = set()
            trials = []
            for rib in candidates:
                orbit = self.orbit(segments[rib])
                group = tuple(sorted(other for other in design.ribs
                                     if any(sameSegment(segments[other], image, self.nodeSnap) for image in orbit)))
                if group in seen:
                    continue
                seen.add(group)
                trial = design.copy()
                for other in group:
                    trial.removeRib(other)
                # rank by the pocket gained before paying for validation
                gainGuess = sum(pocket.area for pocket in self.rawLayout(trial).pockets) - pocketed
                if gainGuess >= params.minPocketArea:
                    trials.append((gainGuess, group, trial))
            for _, group, trial in sorted(trials, key=lambda item: -item[0])[:hugTries]:
                built, ok = self.build(trial)
                if not ok:
                    continue
                gain = sum(pocket.area for pocket in built.pockets) - pocketed
                if gain < params.minPocketArea:
                    continue
                if max([self.overCap(pocket) for pocket in built.pockets] or [0.0]) > max(1.0, worstBefore):
                    # the merged pocket came out too big: split it back to the usual size
                    reach = unary_union([segments[other] for other in group]).buffer(self.typicalWidth)
                    trial, built, _ = self.splitLarge(trial, built, region=reach, rounds=2 * maxRounds)
                    if not self.build(trial)[1]:
                        continue
                    gain = sum(pocket.area for pocket in built.pockets) - pocketed
                    if gain < params.minPocketArea:
                        continue
                if self.crumbs(built) > self.crumbs(layout):
                    continue
                if max([self.overCap(pocket) for pocket in built.pockets] or [0.0]) > max(1.0, worstBefore):
                    continue
                if self.transforms and self.symmetryShortfall(built.pockets) > self.symmetryShortfall(layout.pockets) + 0.02:
                    continue
                if best is None or gain > best[0]:
                    best = (gain, trial, built)
            if best is None:
                break
            notes.append(f"{best[0]:.2f}in2")
            design, layout = best[1], best[2]
        return design, layout, f"huggers: dropped {len(notes)} ({', '.join(notes)} more pocket)" if notes else ""

    # ---------- driver ----------------------------------------------------------------

    def run(self, layout, skip=()):
        design = Design(layout.graph, layout.ribs, layout.ribWidths)
        built, ok = self.build(design)
        if not ok:
            # An input that needed 2D gap trimming can't be held to the stricter check.
            self.strictExport, self.cache, self.rawCache = False, {}, {}
            built, ok = self.build(design)
        if not ok:
            # The pass only edits designs that already pass; anything else is left alone.
            return PolishResult(layout=layout, log=["skipped: input design not valid as rebuilt"], changed=False)
        log = []
        steps = [("symmetry", self.symmetrize), ("links", self.linkBosses), ("blobs", self.fillBlobs),
                 ("huggers", self.dropHuggers), ("split", self.splitLarge), ("symmetry", self.symmetrize),
                 # Symmetrizing can drop ribs and re-open big pockets; split those (in orbits).
                 ("split", self.splitLarge)]
        for name, step in steps:
            if name in skip:
                continue
            newDesign, newLayout, note = step(design, built)
            if not note:
                continue
            if not self.build(newDesign)[1]:
                log.append(f"{name}: reverted (validator)")
                continue
            if self.structuralCheck is not None and not self.structuralCheck(newLayout.pockets):
                log.append(f"{name}: reverted (structural floor)")
                continue
            design, built = newDesign, newLayout
            log.append(note)
        changed = self.key(design) != self.key(Design(layout.graph, layout.ribs, layout.ribWidths))
        if not changed:
            return PolishResult(layout=layout, log=log, changed=False)
        polished = Layout(graph=design.graph, ribs=list(design.ribs), pockets=built.pockets,
                          name=(layout.name or "ml") + "+polish", ribWidths=dict(design.ribWidths))
        return PolishResult(layout=polished, log=log, changed=True)


def polishLayout(plate, params, layout, structuralCheck=None, skip=()):
    """Polish one ML design. structuralCheck(pockets) -> bool, when given, is the structural
    floor each step must keep (e.g. FEA soundness); the validator is always applied.
    skip names steps to leave out (e.g. ("blobs",))."""
    if not layout.ribs or not layout.pockets:
        return PolishResult(layout=layout, log=["skipped: no ribs"], changed=False)
    return Polisher(plate, params, layout, structuralCheck).run(layout, skip=skip)


# ---------- small geometry helpers ------------------------------------------------------

def sameSegment(a, b, tolerance):
    a0, a1 = a.coords[0], a.coords[-1]
    b0, b1 = b.coords[0], b.coords[-1]
    return ((math.dist(a0, b0) < tolerance and math.dist(a1, b1) < tolerance)
            or (math.dist(a0, b1) < tolerance and math.dist(a1, b0) < tolerance))


def trimSegment(segment, amount):
    length = segment.length
    if length <= 2 * amount:
        return segment.interpolate(0.5, normalized=True)
    start = segment.interpolate(amount)
    end = segment.interpolate(length - amount)
    return LineString([start.coords[0], end.coords[0]])


def elongationOf(pocket):
    rectangle = pocket.minimum_rotated_rectangle
    xs, ys = rectangle.exterior.coords.xy
    sides = sorted(math.dist((xs[k], ys[k]), (xs[k + 1], ys[k + 1])) for k in range(2))
    return sides[1] / max(sides[0], 1e-9)


def crossingAngle(first, second):
    a = np.subtract(first.coords[-1], first.coords[0])
    b = np.subtract(second.coords[-1], second.coords[0])
    cosine = abs(float(a @ b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return math.degrees(math.acos(min(1.0, cosine)))
