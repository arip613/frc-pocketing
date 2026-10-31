"""The plate being pocketed: its 2D outline and keep-out features (inches) plus, when it
came from a STEP file, the original solid and plate frame needed to cut it in 3D.

Every inner loop of the input is a feature to preserve. That includes holes, slots,
and cutouts, plus counterbore footprints and anything that rises above the plate face.
"""

import math
from dataclasses import dataclass, field
from functools import cached_property

from shapely.geometry import Polygon
from shapely.ops import unary_union

from frcpocket import geom2d

mmPerInch = 25.4
boltMaxDiameterIn = 0.45  # bolts and counterbores; the smallest FRC bearing bores are ~0.5 in and up
# Big non-round cutouts (windows, slots, notches) are edges of the plate, not holes: their
# pockets keep the full wall, never just the boss ring. Hex bores (roundness 0.91) stay holes.
openingMinAreaIn2 = 0.5
openingMaxRoundness = 0.8


def isOpening(feature):
    return feature.area > openingMinAreaIn2 and 4 * math.pi * feature.area / feature.length**2 < openingMaxRoundness


@dataclass
class PocketParams:
    """User-facing pocketing parameters, always stored in inches."""

    wallIn: float = 0.1875
    ribIn: float = 0.125
    cornerRadiusIn: float = 0.0625  # tool radius; every pocket corner gets at least this
    depthIn: float | None = None  # None = through; otherwise blind depth from the top face
    bossIn: float | None = None  # ring around each feature; None = the wall thickness
    minPocketAreaIn2: float | None = None
    minPocketWidthIn: float | None = None  # skip pockets that can't fit a circle this wide
    # Soft corners like the gold plates: each pocket's fillet is this fraction of its own
    # inscribed radius, never below the tool radius and never above maxFilletIn. Measured on
    # the gold pockets: median 0.32 on the user's examples, 0.38 on the CAD gold plates.
    filletFraction: float = 0.33
    maxFilletIn: float = 0.25
    # Tight corners: every pocket corner is just the machining minimum (1.3x the tool radius),
    # so ribs keep one width right up to where they meet instead of flaring into webs. The user
    # and their CAD experts want uniform ribs; turn off for the soft gold-plate corners above.
    tightCorners: bool = True
    # Rib layout style for the ML decoder: ribs sharing an end meet at least this wide, and
    # a small hole carries at most this many ribs (0 = no limit), a wall point one fewer.
    # Gold plates: the tightest rib-rib angle at a node is 45 deg at the 10th percentile and
    # 104 at the median; holes carry 2 ribs on average and rarely more than 4.
    minRibAngleDegrees: float = 20.0
    maxHoleRibs: int = 0
    # One rib width per plate: the user and their CAD experts don't like variable-width ribs.
    uniformRibs: bool = True
    # Density: multiplies the gold-fitted typical pocket size the ML decodes toward
    # (below 1 = more ribs and smaller pockets, above 1 = fewer, bigger pockets).
    pocketSizeFactor: float = 1.0

    @classmethod
    def fromUnits(cls, units, wall, rib, cornerRadius, depth=None, boss=None, **extra):
        scale = 1 / mmPerInch if units == "mm" else 1.0
        return cls(
            wallIn=wall * scale,
            ribIn=rib * scale,
            cornerRadiusIn=cornerRadius * scale,
            depthIn=None if depth is None else depth * scale,
            bossIn=None if boss is None else boss * scale,
            **extra,
        )

    @property
    def boss(self):
        """Ring width kept around every hole. Its own setting: real FRC plates often use rings
        about half the wall (0.12 in rings with 0.25 in walls on the WCP and Team 221 plates)."""
        return self.bossIn if self.bossIn is not None else self.wallIn

    def ribWidthFor(self, stored):
        """Width a rib is cut at: the rib setting, or (only with uniformRibs off) its own wider width."""
        return self.ribIn if self.uniformRibs or stored is None else max(stored, self.ribIn)

    def ringFor(self, feature):
        """Material kept between a pocket and this feature: the boss ring around holes, and
        at least the wall along big cutouts, whose edges are walls."""
        return max(self.boss, self.wallIn) if isOpening(feature) else self.boss

    @property
    def minPocketArea(self):
        if self.minPocketAreaIn2 is not None:
            return self.minPocketAreaIn2
        return max(0.1, math.pi * (2 * self.cornerRadiusIn) ** 2)

    @property
    def minPocketWidth(self):
        if self.minPocketWidthIn is not None:
            return max(self.minPocketWidthIn, 2 * self.cornerRadiusIn)
        return max(0.3, 4 * self.cornerRadiusIn)


@dataclass
class PlateCad:
    solid: object  # cadquery Solid in its original (world) coordinates, millimetres
    plane: object  # cadquery Plane: origin on the bottom face, normal through the plate
    thicknessMm: float


@dataclass
class Plate:
    outline: Polygon  # exterior only
    features: list  # keep-out polygons
    thicknessIn: float
    name: str = ""
    cad: PlateCad | None = field(default=None, repr=False)

    # Cached: the decoder asks for these thousands of times per design. Plates are never
    # changed after they're built.
    @cached_property
    def material(self):
        return self.outline.difference(unary_union(self.features)) if self.features else self.outline

    @cached_property
    def materialArea(self):
        return self.material.area


def featureKind(polygon):
    """'bolt', 'bearing' (round bores and hex/square shaft bores), or 'cutout'."""
    if geom2d.circularity(polygon) > 0.95:
        diameter = 2 * math.sqrt(polygon.area / math.pi)
        return "bolt" if diameter <= boltMaxDiameterIn else "bearing"
    if geom2d.isShaftBore(polygon):
        return "bearing"
    return "cutout"


def featureCenter(polygon):
    centroid = polygon.centroid
    return centroid if polygon.contains(centroid) else polygon.representative_point()


def plateFromRecord(record, defaultThicknessIn=0.25):
    """A dataset record as an unpocketed input plate (its pockets are dropped)."""
    outline = geom2d.toPolygon(record["outline"])
    features = [geom2d.toPolygon(hole) for hole in record["holes"]]
    return Plate(
        outline=Polygon(outline.exterior),
        features=[feature for feature in features if feature.area > 0],
        thicknessIn=record.get("thicknessIn") or defaultThicknessIn,
        name=record.get("id", ""),
    )


def _loopsToPolygons(loops, scale):
    return [geom2d.toPolygon([(x * scale, y * scale) for x, y in loop["coords"]]) for loop in loops]


def _raisedFootprints(localSolid, thicknessMm, scale):
    """Footprints of material that sits above the plate's top face (bosses, standoffs)."""
    import cadquery as cq

    from frcpocket import cadio

    bb = localSolid.BoundingBox()
    if bb.zmax <= thicknessMm * 1.01 + 0.01:
        return []
    pad = 10.0
    above = cq.Solid.makeBox(
        bb.xlen + 2 * pad, bb.ylen + 2 * pad, bb.zmax - thicknessMm + pad,
        pnt=cq.Vector(bb.xmin - pad, bb.ymin - pad, thicknessMm + 0.02),
    )
    footprints = []
    for face in localSolid.intersect(above).Faces():
        if face.geomType() == "PLANE" and abs(face.normalAt().z) > 0.999:
            points = [(x * scale, y * scale) for x, y in cadio.wireLoop(face.outerWire())["coords"]]
            if len(points) >= 3:
                footprints.append(geom2d.toPolygon(points))
    return footprints


def loadPlateFromStep(path):
    """Read a STEP file holding one flat plate. Raises ValueError if none is found."""
    from frcpocket import cadio

    scale = 1 / mmPerInch
    candidates = []
    for solid in cadio.importStepSolids(path):
        frame = cadio.plateFrame(solid)
        if frame is None:
            continue
        plateData = cadio.extractPlate(solid)
        if plateData is None:
            continue
        candidates.append((solid, frame, plateData))
    if not candidates:
        raise ValueError("No flat plate found in the STEP file (need one prismatic plate solid).")
    solid, (plane, thicknessMm), plateData = max(candidates, key=lambda item: item[0].Volume())

    outline = geom2d.toPolygon([(x * scale, y * scale) for x, y in plateData["outer"]["coords"]])
    loops = plateData["through"] + plateData["topLoops"] + plateData["bottomLoops"]
    features = _loopsToPolygons(loops, scale)
    if plateData["hasRaisedFeatures"]:
        features += _raisedFootprints(cadio.toPlateCoords(solid, plane), thicknessMm, scale)
    merged = unary_union([feature for feature in features if feature.area > 0])
    mergedFeatures = list(getattr(merged, "geoms", [merged])) if not merged.is_empty else []
    return Plate(
        outline=Polygon(outline.exterior),
        features=[Polygon(feature.exterior) for feature in mergedFeatures],
        thicknessIn=thicknessMm * scale,
        name=str(path),
        cad=PlateCad(solid=solid, plane=plane, thicknessMm=thicknessMm),
    )
