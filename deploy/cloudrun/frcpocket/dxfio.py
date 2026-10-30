"""Read 2D plate profiles (router / waterjet exports) from DXF files."""

import math

import ezdxf
from ezdxf import path as dxfPath
from ezdxf.math import bulge_to_arc
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

flattenDistance = 0.005  # drawing units; fine enough for both inch and mm drawings
insUnitsToInch = {1: 1.0, 2: 12.0, 4: 1 / 25.4, 5: 1 / 2.54, 6: 1000 / 25.4}
curveTypes = ("LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE")


def readDxfLoops(path):
    """Return (loops, unitScale) where each loop dict holds coords in drawing units."""
    document = ezdxf.readfile(str(path))
    units = document.header.get("$INSUNITS", 0)
    modelSpace = document.modelspace()
    lines = []
    arcs = []  # (midpoint, radius) of every arc so we can attach radii to loops later
    for entity in modelSpace.query(" ".join(curveTypes)):
        kind = entity.dxftype()
        if kind == "ARC":
            midAngle = math.radians((entity.dxf.start_angle + (entity.dxf.end_angle - entity.dxf.start_angle) % 360 / 2))
            center = entity.dxf.center
            arcs.append(((center.x + entity.dxf.radius * math.cos(midAngle), center.y + entity.dxf.radius * math.sin(midAngle)), entity.dxf.radius))
        elif kind == "LWPOLYLINE":
            points = list(entity.get_points("xyb"))
            closedPoints = points + ([points[0]] if entity.closed else [])
            for (x0, y0, bulge), (x1, y1, _) in zip(closedPoints, closedPoints[1:]):
                if bulge:
                    center, startAngle, endAngle, radius = bulge_to_arc((x0, y0), (x1, y1), bulge)
                    sweep = (endAngle - startAngle) % math.tau
                    midAngle = startAngle + sweep / 2
                    arcs.append(((center.x + radius * math.cos(midAngle), center.y + radius * math.sin(midAngle)), radius))
        try:
            entityPath = dxfPath.make_path(entity)
        except (TypeError, ValueError):
            continue
        points = [(vertex.x, vertex.y) for vertex in entityPath.flattening(distance=flattenDistance)]
        if len(points) >= 2:
            lines.append(LineString(points))
    return lines, arcs, units


def dxfParts(path):
    """Split a DXF into separate parts: [(outerPolygon, [innerPolygons]), ...] in inches."""
    lines, arcs, units = readDxfLoops(path)
    if not lines:
        return [], []
    merged = unary_union(lines)  # nodes the linework so touching segments join up
    faces = [face for face in polygonize(merged) if face.area > 0]
    if not faces:
        return [], []
    scale = insUnitsToInch.get(units)
    if scale is None:
        # Unitless drawing: FRC plates are a few inches to a few feet across.
        largest = max(face.bounds[2] - face.bounds[0] for face in faces)
        scale = 1 / 25.4 if largest > 60 else 1.0
    faces.sort(key=lambda face: -face.area)
    parts = []
    for face in faces:
        exteriorOnly = Polygon(face.exterior)
        if any(Polygon(outer.exterior).contains(exteriorOnly.representative_point()) for outer, _ in parts):
            continue  # this face is a hole (or an island inside a hole) of a bigger part
        inner = [Polygon(ring) for ring in face.interiors]
        parts.append((face, inner))
    return [(scalePolygon(outer, scale), [scalePolygon(hole, scale) for hole in inner]) for outer, inner in parts], [
        ((x * scale, y * scale), radius * scale) for (x, y), radius in arcs
    ]


def scalePolygon(polygon, scale):
    from shapely import affinity

    return affinity.scale(Polygon(polygon.exterior, [ring for ring in polygon.interiors]), xfact=scale, yfact=scale, origin=(0, 0))
