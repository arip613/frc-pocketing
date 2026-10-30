"""2D plane-stress FEA of a pocketed plate (scikit-fem, linear triangles, inches/lbf/psi).

Load cases follow how FRC plates are used. Bolt holes are bolted to the robot (fixed).
Bearing bores carry a shaft load, pushed against one side of the bore, from several
directions. Plates without bearings are loaded like a bracket: holes at one end of
the long axis fixed, the other end loaded. Results are relative to the same plate with
no pockets: stiffness kept, peak stress increase, and mass removed.

These are comparative numbers for ranking layouts (CD's advice: FEA is good for
relative decisions), not a certified strength analysis.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
import triangle
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, MeshTri, asm, condense, solve
from skfem.helpers import ddot, eye, sym_grad, trace

from frcpocket.layout import polygons
from frcpocket.plate import featureCenter, featureKind

youngsModulusPsi = 10.0e6  # 6061-T6 aluminium
poissonRatio = 0.33
loadLbf = 100.0
loadDirectionsDegrees = (0.0, 60.0, 120.0)
stressPercentile = 99.0  # peak stress ignoring single-element singularities


@dataclass
class LoadCase:
    name: str
    fixedRegions: list  # shapely geometries; mesh boundary nodes on them are clamped
    loads: list  # (geometry, (fx, fy), style) with style "bearing" or "edge"


@dataclass
class CaseResult:
    name: str
    compliance: float  # f . u (in*lbf); lower is stiffer
    maxDisplacementIn: float
    peakStressPsi: float  # percentile von Mises


@dataclass
class StructuralReport:
    cases: list = field(default_factory=list)
    baselineCases: list = field(default_factory=list)  # same plate, no pockets
    removedFraction: float = 0.0

    @property
    def stiffnessKept(self):
        """Worst-case ratio of pocketed to solid stiffness (1.0 = as stiff as solid)."""
        return min(base.compliance / case.compliance for case, base in zip(self.cases, self.baselineCases))

    @property
    def stressIncrease(self):
        return max(case.peakStressPsi / base.peakStressPsi for case, base in zip(self.cases, self.baselineCases))

    @property
    def maxDisplacementIn(self):
        return max(case.maxDisplacementIn for case in self.cases)

    @property
    def peakStressPsi(self):
        return max(case.peakStressPsi for case in self.cases)

    @property
    def specificStiffness(self):
        """Stiffness kept per unit of mass kept; above 1 means the pockets cost less stiffness than mass."""
        return self.stiffnessKept / max(1e-9, 1 - self.removedFraction)


maxSteinerPoints = 200000  # hard cap so a bad input can't run the mesher out of memory


def ringSegments(ring, maxLength):
    """Boundary points no farther apart than maxLength, with near-duplicates removed."""
    coords = np.asarray(ring.simplify(maxLength / 20).coords)[:-1]
    keep = [coords[0]]
    for point in coords[1:]:
        if np.hypot(*(point - keep[-1])) > maxLength / 10:
            keep.append(point)
    if len(keep) > 2 and np.hypot(*(keep[0] - keep[-1])) <= maxLength / 10:
        keep.pop()
    coords = np.array(keep)
    points = []
    for start, end in zip(coords, np.roll(coords, -1, axis=0)):
        steps = max(1, int(math.ceil(np.hypot(*(end - start)) / maxLength)))
        points.extend(start + (end - start) * k / steps for k in range(steps))
    return np.array(points)


snapGrid = 1e-6  # inches; coordinates are snapped so touching loops share vertices


def meshPolygon(region, meshSize):
    """Quality triangle mesh of a polygon with holes (Shewchuk's Triangle).

    Triangle can crash on degenerate input, so the boundary is snapped to a grid, made
    valid, and assembled with one shared vertex per location and no repeated or
    zero-length segments.
    """
    region = shapely.set_precision(shapely.make_valid(region), snapGrid)
    vertexIndex, vertices, segments, holes = {}, [], set(), []

    def indexOf(point):
        key = (round(float(point[0]) / snapGrid), round(float(point[1]) / snapGrid))
        if key not in vertexIndex:
            vertexIndex[key] = len(vertices)
            vertices.append((key[0] * snapGrid, key[1] * snapGrid))
        return vertexIndex[key]

    for polygon in polygons(region):
        for ringIndex, ring in enumerate([polygon.exterior, *polygon.interiors]):
            points = ringSegments(ring, meshSize)
            if len(points) < 3:
                continue
            indices = [indexOf(point) for point in points]
            for a, b in zip(indices, indices[1:] + indices[:1]):
                if a != b:
                    segments.add((min(a, b), max(a, b)))
            if ringIndex > 0:
                holes.append(Polygon(ring).representative_point().coords[0])
    if len(vertices) < 3 or len(segments) < 3:
        raise ValueError("region too small to mesh")
    data = {"vertices": np.array(vertices), "segments": np.array(sorted(segments))}
    if holes:
        data["holes"] = np.array(holes)
    maxArea = meshSize**2 * math.sqrt(3) / 4 * 1.2
    result = triangle.triangulate(data, f"pq28a{maxArea:.8f}S{maxSteinerPoints}")
    return MeshTri(np.ascontiguousarray(result["vertices"].T), np.ascontiguousarray(result["triangles"].T))


def planeStressForm(thickness):
    mu = youngsModulusPsi / (2 * (1 + poissonRatio))
    lam = youngsModulusPsi * poissonRatio / (1 - poissonRatio**2)  # plane-stress lambda*

    @BilinearForm
    def stiffness(u, v, w):
        def stress(strain):
            return 2 * mu * strain + lam * eye(trace(strain), 2)

        return thickness * ddot(stress(sym_grad(u)), sym_grad(v))

    return stiffness, mu, lam


def nodesNear(mesh, geometry, tolerance):
    boundary = mesh.boundary_nodes()
    points = mesh.p[:, boundary].T
    keep = [index for index, point in zip(boundary, points) if geometry.distance(Point(point)) <= tolerance]
    return np.array(keep, dtype=int)


def nodalLoads(mesh, geometry, force, style, tolerance):
    nodes = nodesNear(mesh, geometry, tolerance)
    if len(nodes) == 0:
        return nodes, np.zeros(0)
    force = np.asarray(force, dtype=float)
    if style == "bearing":
        centre = np.array(geometry.centroid.coords[0]) if geometry.geom_type != "Point" else np.array(geometry.coords[0])
        direction = force / max(np.linalg.norm(force), 1e-12)
        offsets = mesh.p[:, nodes].T - centre
        weights = np.maximum(0.0, offsets @ direction / np.maximum(np.linalg.norm(offsets, axis=1), 1e-12))
    else:
        weights = np.ones(len(nodes))
    if weights.sum() <= 0:
        weights = np.ones(len(nodes))
    return nodes, weights / weights.sum()


def solveCase(mesh, basis, stiffnessMatrix, case, thickness, tolerance):
    nDofs = basis.N
    loadVector = np.zeros(nDofs)
    for geometry, force, style in case.loads:
        nodes, weights = nodalLoads(mesh, geometry, force, style, tolerance)
        for node, weight in zip(nodes, weights):
            loadVector[basis.nodal_dofs[0, node]] += force[0] * weight
            loadVector[basis.nodal_dofs[1, node]] += force[1] * weight
    fixedNodes = np.unique(np.concatenate([nodesNear(mesh, region, tolerance) for region in case.fixedRegions] or [np.zeros(0, int)]))
    if len(fixedNodes) < 2:
        raise ValueError(f"load case {case.name}: fewer than two supported nodes")
    fixedDofs = basis.nodal_dofs[:, fixedNodes].ravel()
    displacement = solve(*condense(stiffnessMatrix, loadVector, D=fixedDofs))
    compliance = float(loadVector @ displacement)
    nodalDisplacement = displacement[basis.nodal_dofs]
    maxDisplacement = float(np.max(np.hypot(*nodalDisplacement)))
    peakStress = vonMisesPercentile(mesh, basis, displacement)
    return CaseResult(case.name, compliance, maxDisplacement, peakStress)


def vonMisesPercentile(mesh, basis, displacement):
    mu = youngsModulusPsi / (2 * (1 + poissonRatio))
    lam = youngsModulusPsi * poissonRatio / (1 - poissonRatio**2)
    gradient = basis.interpolate(displacement).grad  # shape (2, 2, elements, quadrature points)
    strain = 0.5 * (gradient + np.transpose(gradient, (1, 0, 2, 3)))
    exx, eyy, exy = strain[0, 0].mean(axis=1), strain[1, 1].mean(axis=1), strain[0, 1].mean(axis=1)
    sxx = 2 * mu * exx + lam * (exx + eyy)
    syy = 2 * mu * eyy + lam * (exx + eyy)
    sxy = 2 * mu * exy
    vonMises = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    return float(np.percentile(vonMises, stressPercentile))


def defaultLoadCases(plate):
    """Bolt holes fixed, bearings loaded; or a bracket case along the long axis."""
    bolts = [feature for feature in plate.features if featureKind(feature) == "bolt"]
    bearings = [feature for feature in plate.features if featureKind(feature) != "bolt"]
    if bearings and len(bolts) >= 2:
        return [
            LoadCase(
                name=f"bearings {angle:.0f} deg",
                fixedRegions=[bolt.exterior for bolt in bolts],
                loads=[(bearing.exterior, (loadLbf * math.cos(math.radians(angle)), loadLbf * math.sin(math.radians(angle))), "bearing")
                       for bearing in bearings],
            )
            for angle in loadDirectionsDegrees
        ]
    # Bracket case: support one end of the long axis, load the other end.
    coords = np.asarray(plate.outline.exterior.coords)
    centred = coords - coords.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, -1]
    across = np.array([-axis[1], axis[0]])
    holes = plate.features
    if len(holes) >= 2:
        positions = np.array([np.array(featureCenter(hole).coords[0]) @ axis for hole in holes])
        low, high = np.percentile(positions, [33, 67])
        fixed = [hole.exterior for hole, position in zip(holes, positions) if position <= low]
        loaded = [hole.exterior for hole, position in zip(holes, positions) if position >= high]
        style = "bearing"
    else:
        projections = coords @ axis
        span = projections.max() - projections.min()
        ends = [projections.min() + 0.1 * span, projections.max() - 0.1 * span]
        ring = plate.outline.exterior
        fixed = [ring.intersection(halfPlane(axis, ends[0], below=True, plate=plate))]
        loaded = [ring.intersection(halfPlane(axis, ends[1], below=False, plate=plate))]
        style = "edge"
    cases = []
    for name, direction in (("bracket across", across), ("bracket along", axis)):
        force = tuple(loadLbf * direction / max(1, len(loaded)))
        cases.append(LoadCase(name=name, fixedRegions=fixed, loads=[(region, force, style) for region in loaded]))
    return cases


def halfPlane(axis, offset, below, plate):
    size = 10 * math.sqrt(plate.outline.area) + 100
    across = np.array([-axis[1], axis[0]])
    base = axis * offset
    direction = -axis if below else axis
    corners = [base + across * size, base - across * size, base - across * size + direction * size, base + across * size + direction * size]
    return Polygon(corners)


def analyze(plate, pockets, params=None, meshSize=None, loadCases=None, solidCache=None, cacheKey=None):
    """Structural report for a pocketed plate versus the same plate unpocketed.

    solidCache (a dict) reuses the unpocketed solution across layouts of one plate.
    """
    from frcpocket.layout import removedFraction

    meshSize = meshSize or defaultMeshSize(plate, params)
    tolerance = meshSize * 0.25
    cases = loadCases or defaultLoadCases(plate)
    report = StructuralReport(removedFraction=removedFraction(plate, pockets, params))
    pocketedRegion = plate.material.difference(unary_union(pockets)) if pockets else plate.material
    cacheKey = (cacheKey or id(plate), round(meshSize, 6))
    regions = [(pocketedRegion, report.cases)]
    if solidCache is not None and cacheKey in solidCache:
        report.baselineCases = solidCache[cacheKey]
    else:
        regions.insert(0, (plate.material, report.baselineCases))
    for region, target in regions:
        mesh = meshPolygon(region, meshSize)
        basis = Basis(mesh, ElementVector(ElementTriP1()))
        stiffness, _, _ = planeStressForm(plate.thicknessIn)
        stiffnessMatrix = asm(stiffness, basis)
        for case in cases:
            target.append(solveCase(mesh, basis, stiffnessMatrix, case, plate.thicknessIn, tolerance))
    if solidCache is not None:
        solidCache[cacheKey] = report.baselineCases
    return report


def defaultMeshSize(plate, params):
    smallest = min(params.ribIn, params.wallIn) if params else 0.125
    return float(np.clip(smallest / 2.5, 0.02, 0.1))


_workerSolidCache = {}


def analyzeJob(outlineWkb, featureWkbs, thicknessIn, pocketWkbs, paramsDict, meshSize, plateKey):
    """Picklable FEA entry point for worker processes (crash-isolated, cached per plate)."""
    from frcpocket.plate import Plate, PocketParams

    plate = Plate(outline=shapely.from_wkb(outlineWkb), features=[shapely.from_wkb(wkb) for wkb in featureWkbs],
                  thicknessIn=thicknessIn)
    pockets = [shapely.from_wkb(wkb) for wkb in pocketWkbs]
    report = analyze(plate, pockets, PocketParams(**paramsDict), meshSize=meshSize,
                     solidCache=_workerSolidCache, cacheKey=plateKey)
    return {"stiffnessKept": report.stiffnessKept, "stressIncrease": report.stressIncrease,
            "specificStiffness": report.specificStiffness, "maxDisplacementIn": report.maxDisplacementIn,
            "peakStressPsi": report.peakStressPsi, "removedFraction": report.removedFraction}
