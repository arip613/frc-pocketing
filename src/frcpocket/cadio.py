"""STEP import, flat-plate detection, and conversion of plate faces to 2D loops.

A "plate" is a prismatic solid: two large parallel planar faces a constant thickness
apart, with every other face running straight through between them (holes, pockets,
the outer profile). We find the plate frame, move the solid so the plate lies on XY
with its bottom at z=0, and read loops (closed boundaries) off the mid-thickness
cross-section and the top/bottom faces.
"""

import math

import cadquery as cq
import numpy as np
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.TopAbs import TopAbs_REVERSED

mmPerInch = 25.4
minThicknessMm = 0.9
maxThicknessMm = 20.0
minAspect = 6.0  # smallest in-plane extent / thickness
maxHeightRatio = 3.0  # bosses/standoffs may rise above the base plate, up to 3x its thickness
prismVolumeTolerance = 0.2  # a pure prism's volume matches midArea * thickness within 20%
arcSegmentLengthMm = 0.5


def importStepSolids(path):
    shape = cq.importers.importStep(str(path)).val()
    if isinstance(shape, cq.Solid):
        return [shape]
    return list(shape.Solids())


def planarFaces(solid):
    return [face for face in solid.Faces() if face.geomType() == "PLANE"]


def plateFrame(solid):
    """Return (plane, thickness) if the solid is a flat plate, else None.

    plane is a cq.Plane whose origin lies on the bottom face and whose normal points
    through the plate.
    """
    faces = planarFaces(solid)
    if len(faces) < 2:
        return None
    largest = max(faces, key=lambda face: face.Area())
    normal = largest.normalAt().normalized()
    origin = largest.Center()
    best = None
    for face in faces:
        if face is largest:
            continue
        otherNormal = face.normalAt().normalized()
        if otherNormal.dot(normal) > -0.999:
            continue
        distance = abs((face.Center() - origin).dot(normal))
        if distance < 1e-6:
            continue
        if best is None or face.Area() > best[0].Area():
            best = (face, distance)
    if best is None:
        return None
    thickness = best[1]
    if not (minThicknessMm <= thickness <= maxThicknessMm):
        return None
    # Pick an in-plane x direction from the longest straight outer-profile edge so
    # orientation is repeatable; fall back to any perpendicular vector.
    lineEdges = [edge for edge in largest.outerWire().Edges() if edge.geomType() == "LINE"]
    if lineEdges:
        longest = max(lineEdges, key=lambda edge: edge.Length())
        xDir = (longest.endPoint() - longest.startPoint()).normalized()
    else:
        xDir = normal.cross(cq.Vector(0, 0, 1) if abs(normal.z) < 0.9 else cq.Vector(1, 0, 0)).normalized()
    # The largest face's normal points out of the solid, so the plate body lies along -normal.
    plane = cq.Plane(origin=origin - normal * thickness, xDir=xDir, normal=normal)
    return plane, thickness


def toPlateCoords(solid, plane):
    return solid.transformShape(plane.fG)


def wireLoop(wire):
    """Discretize a closed wire into an (N, 2) array plus edge metadata."""
    points = []
    nLines = 0
    arcRadii = []
    explorer = BRepTools_WireExplorer(wire.wrapped)
    while explorer.More():
        occEdge = explorer.Current()
        edge = cq.Edge(occEdge)
        kind = edge.geomType()
        if kind == "LINE":
            nLines += 1
            count = 2
        else:
            if kind == "CIRCLE":
                arcRadii.append(edge.radius())
            count = max(4, int(math.ceil(edge.Length() / arcSegmentLengthMm)) + 1)
        edgePoints = [edge.positionAt(t) for t in np.linspace(0.0, 1.0, count)]
        if occEdge.Orientation() == TopAbs_REVERSED:
            edgePoints.reverse()
        for point in edgePoints:
            xy = (point.x, point.y)
            if not points or math.dist(points[-1], xy) > 1e-6:
                points.append(xy)
        explorer.Next()
    if len(points) > 2 and math.dist(points[0], points[-1]) < 1e-4:
        points.pop()
    return {"coords": np.array(points), "nLines": nLines, "arcRadii": arcRadii}


def faceLoops(face):
    outer = wireLoop(face.outerWire())
    inner = [wireLoop(wire) for wire in face.innerWires()]
    return outer, inner


def midSectionFace(localSolid, thickness):
    """Cross-section of the plate at half thickness, as the planar face of a thin slice."""
    return sectionFace(localSolid, thickness / 2, max(thickness * 0.02, 0.01))


def sectionFace(localSolid, z, sliceHeight=0.01):
    """Cross-section at height z as a freshly built face (no inherited transforms)."""
    bb = localSolid.BoundingBox()
    pad = 10.0
    slab = cq.Solid.makeBox(
        bb.xlen + 2 * pad, bb.ylen + 2 * pad, sliceHeight,
        pnt=cq.Vector(bb.xmin - pad, bb.ymin - pad, z - sliceHeight / 2),
    )
    thinSlice = localSolid.intersect(slab)
    candidates = [
        face for face in thinSlice.Faces()
        if face.geomType() == "PLANE" and abs(face.normalAt().z) > 0.999
    ]
    if not candidates:
        return None
    # Several faces if the plate is in disconnected pieces; keep the biggest one.
    return max(candidates, key=lambda face: face.Area())


def zFaces(localSolid, z):
    return [
        face for face in planarFaces(localSolid)
        if abs(face.normalAt().z) > 0.999 and abs(face.Center().z - z) < 1e-3
    ]


def extractPlate(solid):
    """Return a dict describing the plate in millimetres, or None if not a plate."""
    frame = plateFrame(solid)
    if frame is None:
        return None
    plane, thickness = frame
    local = toPlateCoords(solid, plane)
    bb = local.BoundingBox()
    if bb.zlen > thickness * maxHeightRatio + 0.05:
        return None  # a tube wall or bent flange, not a plate
    if min(bb.xlen, bb.ylen) < minAspect * thickness:
        return None
    midFace = midSectionFace(local, thickness)
    if midFace is None:
        return None
    prismVolume = midFace.Area() * thickness
    isPrism = bb.zlen <= thickness * 1.05 + 0.05
    if isPrism and abs(local.Volume() - prismVolume) > prismVolumeTolerance * local.Volume():
        return None
    if not isPrism and not (0.8 * prismVolume <= local.Volume() <= 2.0 * prismVolume):
        return None
    outer, through = faceLoops(midFace)
    topLoops = [loop for face in zFaces(local, thickness) for loop in faceLoops(face)[1]]
    bottomLoops = [loop for face in zFaces(local, 0.0) for loop in faceLoops(face)[1]]
    return {
        "thicknessMm": thickness,
        "outer": outer,
        "through": through,
        "topLoops": topLoops,
        "bottomLoops": bottomLoops,
        "volumeMm3": local.Volume(),
        "hasRaisedFeatures": not isPrism,
    }
