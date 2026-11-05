"""Cut a layout's pockets out of the original solid with exact geometry, then export STEP.

The 2D layout (shapely) decides *which* pockets exist. Here the same construction is
repeated with exact OpenCascade geometry in the plate's own frame:
  inner region = the plate's mid-section outline offset inward by the wall (true offset)
  keep         = feature loops offset outward by the boss, plus straight rib rectangles
  pockets      = inner - keep, then opened by the tool radius (exact arcs of radius r)
Each pocket is extruded (through, or blind from the top face) and subtracted from the
untouched original solid, so every original feature stays exactly as modelled.
"""

from dataclasses import dataclass

import cadquery as cq
from shapely.geometry import Point

from frcpocket import cadio, geom2d, validator
from frcpocket.layout import filletRadius

mmPerInch = 25.4
extrudeMarginMm = 1.0
minExactOverlap = 0.98  # IoU between each exact pocket and its 2D twin


@dataclass
class ExportResult:
    solid: object  # cadquery Solid, world coordinates, millimetres
    pocketFaces: list  # exact pocket faces in the plate frame (z = 0), millimetres
    removedVolumeMm3: float
    report2d: object  # ValidationReport on the exact pockets (discretized)
    report3d: object  # ValidationReport for rule 5


def offsetWire(wire, distance, grow):
    """Offset a closed planar wire by |distance|, outward if grow else inward. Returns faces."""
    faces = []
    for signed in (distance, -distance):
        try:
            wires = wire.offset2D(signed, kind="arc")
        except Exception:
            continue
        candidates = [upFace(cq.Face.makeFromWires(result)) for result in wires]
        if not candidates:
            if not grow:
                return []  # inward offset swallowed the whole region
            continue
        original = cq.Face.makeFromWires(wire).Area()
        total = sum(face.Area() for face in candidates)
        if (total > original) == grow:
            return candidates
    return []


def upFace(face):
    """The same face lying on z = 0 with its normal along +z.

    2D offsets can hand back faces pointing either way, and at the height of the face
    the loop originally came from (a counterbore loop on the top face, say).
    """
    if face.normalAt().z < 0:
        face = cq.Shape.cast(face.wrapped.Reversed())
    height = face.Center().z
    return face.translate(cq.Vector(0, 0, -height)) if abs(height) > 1e-9 else face


def prism(face, heightMm=1.0):
    return cq.Solid.extrudeLinear(upFace(face), cq.Vector(0, 0, heightMm))


def fuseSolids(solids):
    if not solids:
        return None
    return solids[0].fuse(*solids[1:]).clean() if len(solids) > 1 else solids[0]


def bottomFaces(solid):
    """Planar faces lying on z = 0: the 2D regions of a prism-based boolean result."""
    return [face for face in solid.Faces() if face.geomType() == "PLANE" and abs(face.Center().z) < 1e-6]


def ribFace(start, end, widthMm):
    sx, sy = start
    ex, ey = end
    length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
    if length < 1e-9:
        return None
    nx, ny = -(ey - sy) / length * widthMm / 2, (ex - sx) / length * widthMm / 2
    corners = [(sx + nx, sy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (sx - nx, sy - ny)]
    return cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(x, y, 0) for x, y in corners], close=True))


def openFace(face, radiusMm):
    """Morphological opening of a planar face by a disk: what an end mill of radius r can cut."""
    opened = []
    for eroded in offsetWire(face.outerWire(), radiusMm, grow=False):
        opened.extend(offsetWire(eroded.outerWire(), radiusMm, grow=True))
    islands = [prism(cq.Face.makeFromWires(wire)) for wire in face.innerWires()]
    if islands and opened:
        cut = fuseSolids([prism(piece) for piece in opened]).cut(fuseSolids(islands))
        opened = bottomFaces(cut)
    return [upFace(piece) for piece in opened]


sectionInsetMm = 0.05  # slice this far inside the top and bottom faces


def featureWires(local, thicknessMm):
    """Every keep-out loop, taken from fresh slices near the bottom, middle, and top.

    Slicing (rather than reading the solid's own top/bottom faces) gives geometry with
    no inherited placement, which 2D offsets need, and catches counterbores and
    countersinks from either side. Loops are projected onto exactly z = 0.
    """
    midFace = cadio.midSectionFace(local, thicknessMm)
    midShift = cq.Vector(0, 0, -midFace.Center().z)
    wires = [wire.translate(midShift) for wire in midFace.innerWires()]
    for z in (sectionInsetMm, thicknessMm - sectionInsetMm):
        face = cadio.sectionFace(local, z)
        if face is not None:
            shift = cq.Vector(0, 0, -face.Center().z)
            wires += [wire.translate(shift) for wire in face.innerWires()]
    return midFace.outerWire().translate(midShift), wires


def polygonOfFace(face):
    outer = cadio.wireLoop(face.outerWire())["coords"] / mmPerInch
    holes = [cadio.wireLoop(wire)["coords"] / mmPerInch for wire in face.innerWires()]
    return geom2d.toPolygon(outer) if not holes else geom2d.Polygon(outer, holes).buffer(0)


def exactPocketFaces(plate, params, layout):
    cad = plate.cad
    thicknessMm = cad.thicknessMm
    local = cadio.toPlateCoords(cad.solid, cad.plane)
    outerWire, holeWires = featureWires(local, thicknessMm)
    # Booleans run on thin prisms: solid booleans don't care which way a face points,
    # and stacked concentric bosses (hole + counterbore) merge correctly.
    inner = fuseSolids([prism(face) for face in offsetWire(outerWire, params.wallIn * mmPerInch, grow=False)])
    if inner is None:
        return []
    rings = [params.ringFor(geom2d.toPolygon(cadio.wireLoop(wire)["coords"] / mmPerInch)) for wire in holeWires]
    keep = [face for wire, ring in zip(holeWires, rings) for face in offsetWire(wire, ring * mmPerInch, grow=True)]
    for edge in layout.ribs:
        i, j = edge
        start = (layout.graph.nodes[i].x * mmPerInch, layout.graph.nodes[i].y * mmPerInch)
        end = (layout.graph.nodes[j].x * mmPerInch, layout.graph.nodes[j].y * mmPerInch)
        face = ribFace(start, end, layout.ribWidth(edge, params) * mmPerInch)
        if face is not None:
            keep.append(face)
    raw = inner.cut(fuseSolids([prism(face) for face in keep])) if keep else inner
    pockets = []
    for face in bottomFaces(raw):
        radius = filletRadius(polygonOfFace(face), params)  # same per-pocket fillet as the 2D layout
        pockets.extend(openFace(face, radius * mmPerInch))
    # Keep exactly the pockets the 2D layout decided on (same construction, so they match).
    chosen = []
    for face in pockets:
        centre = polygonOfFace(face).representative_point()
        if any(pocket.contains(centre) for pocket in layout.pockets):
            chosen.append(face)
    return chosen


def cutPockets(plate, params, pocketFaces):
    cad = plate.cad
    thicknessMm = cad.thicknessMm
    if params.depthIn is None:
        bottom, depth = -extrudeMarginMm, thicknessMm + 2 * extrudeMarginMm
    else:
        depthMm = min(params.depthIn * mmPerInch, thicknessMm)
        bottom, depth = thicknessMm - depthMm, depthMm + extrudeMarginMm
    tools = [prism(face, depth).moved(cq.Location(cq.Vector(0, 0, bottom))) for face in pocketFaces]
    if not tools:
        return cad.solid, 0.0
    cutter = cq.Compound.makeCompound(tools).transformShape(cad.plane.rG)
    result = cad.solid.cut(cutter).clean()
    solids = result.Solids()
    if len(solids) == 1:
        result = solids[0]
    effectiveDepth = thicknessMm if params.depthIn is None else min(params.depthIn * mmPerInch, thicknessMm)
    removed = sum(face.Area() for face in pocketFaces) * effectiveDepth
    return result, removed


def buildPocketedSolid(plate, params, layout):
    if plate.cad is None:
        raise ValueError("This plate has no source solid; load it with loadPlateFromStep.")
    pocketFaces = exactPocketFaces(plate, params, layout)
    exactPolygons = [polygonOfFace(face) for face in pocketFaces]
    report2d = validator.validateLayout(plate, params, exactPolygons)
    worstOverlap = min(
        (max(pocket.intersection(exact).area / pocket.union(exact).area for pocket in layout.pockets)
         for exact in exactPolygons),
        default=1.0,
    )
    matched = len(pocketFaces) == len(layout.pockets) and worstOverlap >= minExactOverlap
    report2d.add("exact match", matched, worstOverlap, minExactOverlap,
                 "" if matched else "exact geometry differs from the validated 2D layout")
    solid, removed = cutPockets(plate, params, pocketFaces)
    report3d = validator.validateSolid(plate, params, solid, removed)
    return ExportResult(solid=solid, pocketFaces=pocketFaces, removedVolumeMm3=removed, report2d=report2d, report3d=report3d)


def exportStep(result, path, allowInvalid=False):
    """Write the STEP only for a design that passed every rule (unless explicitly overridden)."""
    if not allowInvalid and not (result.report2d.passed and result.report3d.passed):
        failed = [check.rule for check in result.report2d.failures + result.report3d.failures]
        raise ValueError(f"refusing to export a design that fails validation: {failed}")
    cq.exporters.export(result.solid, str(path))
    return path
