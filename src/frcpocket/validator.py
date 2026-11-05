"""Geometric validator: every design must pass all five hard rules or it is rejected.

It checks only the final geometry (plate + pockets), never how a layout was produced,
so ML output gets no special treatment.

  1. wall     every pocket stays >= wall thickness from the plate perimeter, and from
              the edge of every big non-round cutout (window, slot, notch)
  2. bosses   every feature keeps a ring >= the boss setting wide (the wall thickness
              unless set separately), and the material
              is one connected piece, so no boss floats and no feature sits in a pocket
  3. ribs     material between pockets is >= rib thickness everywhere
  4. corners  every pocket is cuttable by a tool of the corner radius (so each corner
              radius >= tool radius), and no pocket is below the minimum size
  5. solid    (3D, after cutting) one valid solid; volume removed matches the pockets;
              the original solid is unchanged around every feature and along the wall

validateLayout() covers 1–4 in 2D; validateSolid() covers 5 on the exported solid.
"""

from dataclasses import dataclass, field

import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree

from frcpocket.layout import arcSegments, isWorthCutting, polygons
from frcpocket.plate import isOpening

lengthTolIn = 1e-3  # 0.001 in: well under machining tolerance, above float/arc noise
areaTolIn2 = 2e-4


@dataclass
class RuleResult:
    rule: str
    passed: bool
    measured: float | None = None
    required: float | None = None
    detail: str = ""


@dataclass
class ValidationReport:
    results: list = field(default_factory=list)

    @property
    def passed(self):
        return all(result.passed for result in self.results)

    @property
    def failures(self):
        return [result for result in self.results if not result.passed]

    def add(self, *args, **kwargs):
        self.results.append(RuleResult(*args, **kwargs))

    def summary(self):
        lines = []
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            numbers = ""
            if result.measured is not None:
                numbers = f" measured {result.measured:.4f}"
                if result.required is not None:
                    numbers += f" / required {result.required:.4f}"
            lines.append(f"{status} {result.rule}{numbers}{' - ' + result.detail if result.detail else ''}")
        return "\n".join(lines)


def minPairDistance(pockets, searchRadius):
    """Smallest gap between two different pockets (inf if none are near each other)."""
    if len(pockets) < 2:
        return float("inf")
    tree = STRtree(pockets)
    best = float("inf")
    for index, pocket in enumerate(pockets):
        for other in tree.query(pocket.buffer(searchRadius)):
            if other > index:
                best = min(best, pocket.distance(pockets[other]))
    return best


def thinMaterialNearPockets(plate, pockets, width):
    """Area of material narrower than `width` that pocketing created (not already in the plate)."""
    def thinPart(region):
        radius = width / 2 - lengthTolIn
        return region.difference(region.buffer(-radius, quad_segs=arcSegments).buffer(radius, quad_segs=arcSegments))

    original = plate.material
    pocketed = original.difference(unary_union(pockets))
    created = thinPart(pocketed).difference(thinPart(original).buffer(lengthTolIn * 5))
    nearPockets = created.intersection(unary_union(pockets).buffer(lengthTolIn * 5))
    # Ignore arc-approximation slivers where big fillets blend into bosses (each ~1e-4 in^2);
    # a real neck is a strip at least a few rib-widths long, far above this.
    minPieceArea = width**2 / 8
    return sum(piece.area for piece in polygons(nearPockets) if piece.area > minPieceArea)


def validateLayout(plate, params, pockets, checkCorners=True):
    """Rules 1-4 in 2D. checkCorners=False skips rule 4, for the decoder's intermediate
    checks: pockets built by layout.finishRegions are opened at the tool radius or more,
    so they satisfy it by construction, and every finished design gets the full check."""
    report = ValidationReport()
    if not pockets:
        report.add("pockets", True, detail="no pockets (nothing to cut)")
    outlineRing = plate.outline.exterior
    features = plate.features

    # Rule 1: perimeter wall.
    outside = [pocket for pocket in pockets if not plate.outline.buffer(lengthTolIn).contains(pocket)]
    minWall = min((outlineRing.distance(pocket) for pocket in pockets), default=float("inf"))
    report.add("1 wall", not outside and minWall >= params.wallIn - lengthTolIn, minWall, params.wallIn,
               f"{len(outside)} pocket(s) leave the plate" if outside else "")

    # Big cutouts are walls too (rule 1); every other feature keeps its boss ring (rule 2).
    minBoss = minCutoutWall = float("inf")
    if features and pockets:
        featureTree = STRtree(features)
        reach = 2 * max(params.boss, params.wallIn)
        for pocket in pockets:
            for index in featureTree.query(pocket.buffer(reach)):
                distance = pocket.distance(features[index])
                if isOpening(features[index]):
                    minCutoutWall = min(minCutoutWall, distance)
                else:
                    minBoss = min(minBoss, distance)
    cutoutWall = max(params.boss, params.wallIn)
    report.add("1 wall at cutouts", minCutoutWall >= cutoutWall - lengthTolIn, minCutoutWall, cutoutWall)

    # Rule 2: bosses around every feature, connected material.
    material = plate.material.difference(unary_union(pockets)) if pockets else plate.material
    pieces = polygons(material)
    floating = len([piece for piece in pieces if piece.area > areaTolIn2])
    report.add("2 bosses", minBoss >= params.boss - lengthTolIn, minBoss, params.boss)
    report.add("2 connected", floating == 1, floating, 1,
               "" if floating == 1 else "material splits into separate pieces (a floating boss or island)")

    # Rule 3: rib thickness between pockets, and no thin necks created anywhere.
    # Pocket-to-pocket gaps must reach the rib width. Anywhere else (bosses, walls, a
    # peninsula inside one pocket) the floor is the smallest of rib, wall and boss, since
    # bosses only owe the boss width and walls the wall thickness.
    minRib = minPairDistance(pockets, params.ribIn * 2)
    neckWidth = min(params.ribIn, params.wallIn, params.boss)
    thinArea = thinMaterialNearPockets(plate, pockets, neckWidth) if pockets else 0.0
    report.add("3 ribs", minRib >= params.ribIn - lengthTolIn, minRib, params.ribIn)
    report.add("3 no thin necks", thinArea <= areaTolIn2 * max(1, len(pockets)), thinArea, 0.0,
               "" if thinArea <= areaTolIn2 * max(1, len(pockets)) else "material thinner than the rib next to a pocket")

    if not checkCorners:
        return report
    # Rule 4: every pocket is a union of tool-radius disks (corner radius >= tool radius).
    # The worst-case distance between the pocket and what the tool can actually cut is
    # ~0.41 r at a sharp 90° corner, and 0 when every corner is filleted at >= r.
    radius = params.cornerRadiusIn - lengthTolIn
    worstResidual, tooSmall = 0.0, 0
    for pocket in pockets:
        opened = pocket.buffer(-radius, quad_segs=arcSegments).buffer(radius, quad_segs=arcSegments)
        if opened.is_empty:
            residual = float("inf")
        else:
            residual = shapely.hausdorff_distance(pocket.boundary, opened.boundary, densify=0.1)
        worstResidual = max(worstResidual, residual)
        if not isWorthCutting(pocket, params):
            tooSmall += 1
    report.add("4 corner radius", worstResidual <= lengthTolIn, worstResidual, lengthTolIn,
               "a pocket has a corner tighter than the tool radius" if worstResidual > lengthTolIn else "")
    report.add("4 pocket size", tooSmall == 0, tooSmall, 0,
               f"{tooSmall} pocket(s) too small to cut" if tooSmall else "")
    return report


def validateSolid(plate, params, resultSolid, expectedRemovedMm3, relTol=1e-3):
    """Rule 5 on the exported solid (millimetres, world coordinates)."""
    import cadquery as cq

    report = ValidationReport()
    solids = resultSolid.Solids() if hasattr(resultSolid, "Solids") else [resultSolid]
    report.add("5 single solid", len(solids) == 1, len(solids), 1)
    report.add("5 valid B-rep", resultSolid.isValid(), None, None)
    originalVolume = plate.cad.solid.Volume()
    removed = originalVolume - resultSolid.Volume()
    report.add("5 volume removed", abs(removed - expectedRemovedMm3) <= relTol * originalVolume,
               removed, expectedRemovedMm3)

    # Around every feature and along the wall the solid must be untouched.
    scale = 25.4
    keepZone = unary_union(
        [feature.buffer(params.ringFor(feature) / 2, quad_segs=arcSegments) for feature in plate.features]
        + [plate.outline.difference(plate.outline.buffer(-params.wallIn / 2, quad_segs=arcSegments))]
    )
    thicknessMm = plate.cad.thicknessMm
    zoneSolids = []
    for piece in polygons(keepZone):
        exterior = [cq.Vector(x * scale, y * scale, 0) for x, y in piece.exterior.coords[:-1]]
        wires = [cq.Wire.makePolygon(exterior, close=True)]
        for ring in piece.interiors:
            wires.append(cq.Wire.makePolygon([cq.Vector(x * scale, y * scale, 0) for x, y in ring.coords[:-1]], close=True))
        face = cq.Face.makeFromWires(wires[0], wires[1:])
        bb = plate.cad.solid.BoundingBox()
        height = max(thicknessMm * 4, bb.DiagonalLength)
        zoneSolids.append(cq.Solid.extrudeLinear(face, cq.Vector(0, 0, height)).moved(cq.Location(cq.Vector(0, 0, -height / 2))))
    zone = cq.Compound.makeCompound(zoneSolids).transformShape(plate.cad.plane.rG)
    before = plate.cad.solid.intersect(zone).Volume()
    after = resultSolid.intersect(zone).Volume()
    report.add("5 features unchanged", abs(before - after) <= relTol * max(before, 1.0), abs(before - after), 0.0,
               "" if abs(before - after) <= relTol * max(before, 1.0) else "pocketing changed material around a feature or the wall")
    return report
