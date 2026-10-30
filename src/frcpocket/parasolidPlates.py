"""Flat plates straight from Parasolid text files (.x_t), for CAD only published that way.

SDS ships its newer swerve modules (MK4c, MK4n, MK5) only as Parasolid. parasolid-kit
(optional dependency) parses the B-rep; converting a whole assembly to OCCT fails on any
one unsupported surface (a wheel, a bolt), so plates are read from the B-rep directly:
a plate is a large planar face with an anti-parallel planar face a plate thickness
away whose outline matches it (assembly exports can fuse several parts into one solid). Loops on that face become the outline
and the cut-outs, sampled the same way cadio.wireLoop samples STEP wires, and written to
the same raw-plate records extract.py caches.

    python -I -m frcpocket.parasolidPlates data/raw/vendors/SDS_parasolid --cache data/processed_wcp/cache
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from frcpocket.cadio import arcSegmentLengthMm, maxThicknessMm, minThicknessMm
from frcpocket.extract import cacheVersion, cachePathFor

mmPerMetre = 1000.0
mmPerInch = 25.4
parallelDot = 0.999
minFaceAreaMm2 = 300.0
minOutlineOverlap = 0.9


def vec(v):
    return np.array([v.x, v.y, v.z])


class BodyGeometry:
    def __init__(self, model):
        self.model = model
        self.curves = {curve.id: curve for curve in model.curves}
        self.surfaces = {surface.id: surface for surface in model.surfaces}
        self.vertices = {vertex.id: vertex for vertex in model.vertices}
        self.points = {point.id: point for point in model.points}
        self.edges = {edge.id: edge for edge in model.edges}
        self.halfEdges = {halfEdge.id: halfEdge for halfEdge in model.half_edges}
        self.loops = {loop.id: loop for loop in model.loops}
        self.faces = {face.id: face for face in model.faces}
        self.shells = {shell.id: shell for shell in model.shells}
        self.regions = {region.id: region for region in model.regions}

    def vertexMm(self, vertexId):
        return vec(self.points[self.vertices[vertexId].point].position) * mmPerMetre

    def regionFaces(self, region):
        faceIds = []
        for shellId in region.shells:
            shell = self.shells[shellId]
            faceIds += list(shell.front_faces) + list(shell.back_faces)
        return [self.faces[faceId] for faceId in dict.fromkeys(faceIds)]

    def faceVertices(self, faces):
        vertexIds = set()
        for face in faces:
            for loopId in face.loops:
                for halfEdgeId in self.loops[loopId].half_edges:
                    vertexId = self.halfEdges[halfEdgeId].vertex
                    if vertexId is not None:
                        vertexIds.add(vertexId)
        return vertexIds

    def plane(self, face):
        surface = self.surfaces.get(face.surface)
        if surface is None or surface.kind.value != "plane":
            return None
        normal = vec(surface.definition.normal)
        normal /= np.linalg.norm(normal)
        flip = (surface.sense.value == "-") != (face.sense.value == "-")
        return vec(surface.definition.point) * mmPerMetre, -normal if flip else normal

    def halfEdgeEnds(self, halfEdge):
        edge = self.edges[halfEdge.edge]
        start, end = edge.start_vertex, edge.end_vertex
        return (start, end) if halfEdge.sense.value == "+" else (end, start)

    def edgePoints(self, halfEdge):
        """3D points (mm) along one half-edge in loop order, plus its kind and radius."""
        edge = self.edges[halfEdge.edge]
        startId, endId = self.halfEdgeEnds(halfEdge)
        start = self.vertexMm(startId) if startId is not None else None
        end = self.vertexMm(endId) if endId is not None else None
        curve = self.curves.get(edge.curve)
        kind = curve.kind.value if curve is not None else "none"
        definition = curve.definition if curve is not None else None
        sense = curve.sense.value if curve is not None else "+"
        if kind == "trimmed":
            basis = self.curves.get(definition.basis_curve)
            if basis is not None and basis.kind.value in ("line", "circle"):
                kind, definition = basis.kind.value, basis.definition
                sense = "+" if (basis.sense.value == "+") == (sense == "+") else "-"
        if kind == "circle":
            centre = vec(definition.center) * mmPerMetre
            normal = vec(definition.normal)
            normal /= np.linalg.norm(normal)
            xAxis = vec(definition.x_axis)
            xAxis /= np.linalg.norm(xAxis)
            yAxis = np.cross(normal, xAxis)
            radius = definition.radius * mmPerMetre
            counterClockwise = (sense == "+") == (halfEdge.sense.value == "+")

            def angle(point):
                offset = point - centre
                return math.atan2(offset @ yAxis, offset @ xAxis)

            if start is None or end is None or np.linalg.norm(start - end) < 1e-6:
                a0 = angle(start) if start is not None else 0.0
                sweep = 2 * math.pi
            else:
                a0, a1 = angle(start), angle(end)
                sweep = (a1 - a0) % (2 * math.pi) if counterClockwise else -((a0 - a1) % (2 * math.pi))
                sweep = abs(sweep)
            direction = 1 if counterClockwise else -1
            count = max(4, int(math.ceil(radius * sweep / arcSegmentLengthMm)) + 1)
            angles = a0 + direction * np.linspace(0.0, sweep, count)
            points = [centre + radius * (math.cos(a) * xAxis + math.sin(a) * yAxis) for a in angles]
            return points, "circle", radius
        if start is None:
            return [], kind, None
        return [start, end], ("line" if kind == "line" else kind), None

    def loopPoints(self, loop):
        """Loop as 3D points in traversal order with STEP-style line/arc metadata."""
        points, nLines, arcRadii = [], 0, []
        for halfEdgeId in loop.half_edges:
            halfEdge = self.halfEdges[halfEdgeId]
            if halfEdge.edge is None:
                continue
            edgePoints, kind, radius = self.edgePoints(halfEdge)
            if kind == "line":
                nLines += 1
            elif kind == "circle":
                arcRadii.append(radius)
            for point in edgePoints:
                if not points or np.linalg.norm(points[-1] - point) > 1e-6:
                    points.append(point)
        if len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < 1e-4:
            points.pop()
        return points, nLines, arcRadii


def toPlane(points, origin, xAxis, yAxis):
    return [((p - origin) @ xAxis, (p - origin) @ yAxis) for p in points]


def platesFromModel(model):
    geometry = BodyGeometry(model)
    plates = []
    # SolidWorks assembly exports can fuse several parts into one solid, so plates are
    # found as face pairs rather than whole bodies: a large planar face with an
    # anti-parallel planar face a plate thickness away whose outline matches it.
    planar = []
    for body in model.bodies:
        if body.kind.value != "solid":
            continue
        for regionId in body.regions:
            region = geometry.regions[regionId]
            if region.kind.value != "solid":
                continue
            for face in geometry.regionFaces(region):
                frame = geometry.plane(face)
                if frame is None:
                    continue
                loops = [geometry.loopPoints(geometry.loops[loopId]) for loopId in face.loops]
                loops = [loop for loop in loops if len(loop[0]) >= 3]
                if not loops:
                    continue
                origin, normal = frame
                helper = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
                xAxis = np.cross(normal, helper)
                xAxis /= np.linalg.norm(xAxis)
                yAxis = np.cross(normal, xAxis)
                polygons = [Polygon(toPlane(loop[0], origin, xAxis, yAxis)) for loop in loops]
                if not all(polygon.is_valid for polygon in polygons):
                    continue
                outer = int(np.argmax([polygon.area for polygon in polygons]))
                if polygons[outer].area < minFaceAreaMm2:
                    continue
                planar.append({"id": face.id, "origin": origin, "normal": normal, "loops": loops, "outer": outer,
                               "outerPoints": loops[outer][0]})
    used = set()
    for top in sorted(planar, key=lambda item: -len(item["outerPoints"])):
        if top["id"] in used:
            continue
        normal = -top["normal"]
        helper = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
        xAxis = np.cross(normal, helper)
        xAxis /= np.linalg.norm(xAxis)
        yAxis = np.cross(normal, xAxis)
        topPolygon = Polygon(toPlane(top["outerPoints"], top["origin"], xAxis, yAxis))
        best = None
        for other in planar:
            if other["id"] in used or other is top or other["normal"] @ top["normal"] > -parallelDot:
                continue
            thickness = (top["origin"] - other["origin"]) @ top["normal"]
            if not (minThicknessMm <= thickness <= maxThicknessMm):
                continue
            otherPolygon = Polygon(toPlane(other["outerPoints"], top["origin"], xAxis, yAxis))
            union = topPolygon.union(otherPolygon).area
            overlap = topPolygon.intersection(otherPolygon).area / union if union else 0
            if overlap > minOutlineOverlap and (best is None or overlap > best[1]):
                best = (other, overlap, thickness)
        if best is None:
            continue
        bottom, _, thickness = best
        used.update((top["id"], bottom["id"]))
        # Plate coordinates: bottom face at z=0, x along the outline's longest straight edge.
        origin = top["origin"] - top["normal"] * thickness
        outerPoints = top["outerPoints"]
        segments = [(outerPoints[i], outerPoints[(i + 1) % len(outerPoints)]) for i in range(len(outerPoints))]
        a, b = max(segments, key=lambda seg: np.linalg.norm(seg[1] - seg[0]))
        xAxis = (b - a) - ((b - a) @ normal) * normal
        xAxis /= np.linalg.norm(xAxis)
        yAxis = np.cross(normal, xAxis)

        def record(loop):
            points, nLines, arcRadii = loop
            return {"coords": [[x / mmPerInch, y / mmPerInch] for x, y in toPlane(points, origin, xAxis, yAxis)],
                    "nLines": nLines, "arcRadii": [radius / mmPerInch for radius in arcRadii]}

        topInner = [record(loop) for i, loop in enumerate(top["loops"]) if i != top["outer"]]
        bottomInner = [record(loop) for i, loop in enumerate(bottom["loops"]) if i != bottom["outer"]]
        bottomPolygons = [Polygon(loop["coords"]).buffer(0.01) for loop in bottomInner]
        through = [loop for loop in topInner
                   if any(poly.contains(Polygon(loop["coords"]).representative_point()) for poly in bottomPolygons)]
        blind = [loop for loop in topInner if loop not in through]
        blind += [loop for loop in bottomInner
                  if not any(Polygon(t["coords"]).buffer(0.01).contains(Polygon(loop["coords"]).representative_point())
                             for t in through)]
        plates.append({
            "solidIndex": len(plates),
            "thicknessIn": thickness / mmPerInch,
            "hasRaisedFeatures": False,
            "outline": record(top["loops"][top["outer"]])["coords"],
            "through": through,
            "faceLoops": blind,
        })
    return plates


def main():
    import parasolid_kit

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--cache", default="data/processed_wcp/cache")
    args = parser.parse_args()
    cacheDir = Path(args.cache)
    cacheDir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.folder).glob("*.x_t")):
        cachePath = cachePathFor(path, cacheDir)
        try:
            parsed = parasolid_kit.read_brep(str(path))
            plates = platesFromModel(parsed.brep)
            cachePath.write_text(json.dumps({"ok": True, "version": cacheVersion, "plates": plates}))
            print(f"{path.name}: {len(parsed.brep.bodies)} bodies, {len(plates)} plates", flush=True)
        except Exception as error:
            cachePath.write_text(json.dumps({"ok": False, "version": cacheVersion, "error": f"{type(error).__name__}: {error}"[:2000]}))
            print(f"{path.name}: FAILED {type(error).__name__}: {str(error)[:200]}", flush=True)


if __name__ == "__main__":
    main()
