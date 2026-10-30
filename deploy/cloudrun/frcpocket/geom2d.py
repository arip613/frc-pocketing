"""2D plate geometry helpers shared by the extractor, scorer, and validator.

All functions here work in inches unless a name says otherwise.
"""

import math

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

minPocketAreaIn2 = 0.15  # below this an inner loop is a feature (hex hole, zip-tie slot)
slotMaxWidthIn = 0.6
maxShaftBoreAreaIn2 = 0.6


def toPolygon(coords):
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
        if polygon.geom_type != "Polygon":
            polygon = max(getattr(polygon, "geoms", [polygon]), key=lambda part: part.area)
    return polygon


def circularity(polygon):
    """4*pi*A / P^2: 1 for a circle, ~0.6 for an equilateral triangle."""
    return 4 * math.pi * polygon.area / polygon.length**2 if polygon.length > 0 else 0.0


def isCircleLoop(loop, polygon):
    radii = loop.get("arcRadii", [])
    if loop.get("nLines", 0) == 0 and radii and max(radii) - min(radii) < 1e-3 * max(radii):
        return True
    return loop.get("nLines", 0) == 0 and circularity(polygon) > 0.97


def isSlotLoop(loop, polygon):
    # Two straight sides and rounded ends (the ends may be split into several arcs).
    if loop.get("nLines", 0) > 2 or len(loop.get("arcRadii", [])) < 2:
        return False
    rotatedRect = polygon.minimum_rotated_rectangle
    xs, ys = rotatedRect.exterior.coords.xy
    sides = sorted({round(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])), 6) for i in range(4)})
    return sides[0] <= slotMaxWidthIn


def straightRuns(polygon, minLength=0.1, maxTurnDegrees=1.0):
    """Lengths of straight runs longer than minLength in a flattened ring."""
    coords = np.asarray(polygon.exterior.coords)
    segments = np.diff(coords, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    keep = lengths > 1e-9
    segments, lengths = segments[keep], lengths[keep]
    if len(segments) == 0:
        return []
    angles = np.degrees(np.arctan2(segments[:, 1], segments[:, 0]))
    runs = []
    runLength = lengths[0]
    for index in range(1, len(segments)):
        turn = abs((angles[index] - angles[index - 1] + 180) % 360 - 180)
        if turn <= maxTurnDegrees:
            runLength += lengths[index]
        else:
            runs.append(runLength)
            runLength = lengths[index]
    runs.append(runLength)
    # The ring starts mid-run as often as not; merge the wrap-around pieces.
    closingTurn = abs((angles[0] - angles[-1] + 180) % 360 - 180)
    if len(runs) > 1 and closingTurn <= maxTurnDegrees:
        runs[0] += runs.pop()
    return [length for length in runs if length > minLength]


def straightRunCount(polygon, minLength=0.1, maxTurnDegrees=1.0):
    return len(straightRuns(polygon, minLength, maxTurnDegrees))


def isShaftBore(polygon):
    """Hex or square shaft bores: 4 or 6 equal straight sides, at most ~0.75 in across."""
    runs = straightRuns(polygon, minLength=0.05)
    if len(runs) not in (4, 6) or polygon.area > maxShaftBoreAreaIn2:
        return False
    return (max(runs) - min(runs)) < 0.08 * max(runs)


def isGearOutline(outline):
    """Gears and sprockets: many teeth around a nearly round hull."""
    hull = outline.convex_hull
    toothy = outline.area / hull.area < 0.97 and outline.length / hull.length > 1.15
    return toothy and circularity(hull) > 0.93


def classifyLoop(loop, polygon):
    """'pocket' for lightening cuts, 'hole' for everything that must be preserved."""
    if circularity(polygon) > 0.95:
        return "hole"
    if isCircleLoop(loop, polygon) or isSlotLoop(loop, polygon):
        return "hole"
    if polygon.area < minPocketAreaIn2:
        return "hole"
    if isShaftBore(polygon):
        return "hole"
    if loop.get("nLines", 0) < 2 and circularity(polygon) > 0.85:
        return "hole"  # big round-ish bore, e.g. a bearing or a pass-through
    if len(loop.get("arcRadii", [])) < 3:
        return "hole"  # sharp-cornered cutout (tube pass-through, motor window), not a routed pocket
    return "pocket"


def principalAngle(polygon):
    coords = np.asarray(polygon.exterior.coords)[:-1]
    centered = coords - coords.mean(axis=0)
    covariance = centered.T @ centered
    eigenValues, eigenVectors = np.linalg.eigh(covariance)
    major = eigenVectors[:, np.argmax(eigenValues)]
    return math.degrees(math.atan2(major[1], major[0]))


def iou(a, b):
    union = a.union(b).area
    return a.intersection(b).area / union if union > 0 else 1.0


def symmetryScore(outline, pockets):
    """Best IoU of the pocket set with its mirror (either principal axis) or 180° turn."""
    if not pockets:
        return 0.0
    pocketUnion = unary_union(pockets)
    center = outline.centroid
    angle = principalAngle(outline)
    aligned = affinity.rotate(pocketUnion, -angle, origin=center)
    candidates = [
        affinity.scale(aligned, xfact=-1, yfact=1, origin=center),
        affinity.scale(aligned, xfact=1, yfact=-1, origin=center),
        affinity.rotate(aligned, 180, origin=center),
    ]
    return max(iou(aligned, candidate) for candidate in candidates)


def pocketStats(outline, holes, pockets):
    """Measurements that describe a pocketing layout (inches)."""
    holeArea = sum(hole.area for hole in holes)
    pocketArea = sum(pocket.area for pocket in pockets)
    solidArea = outline.area - holeArea
    walls = [outline.exterior.distance(pocket) for pocket in pockets]
    ribs = []
    for index, pocket in enumerate(pockets):
        others = [other for otherIndex, other in enumerate(pockets) if otherIndex != index]
        if others:
            ribs.append(min(pocket.distance(other) for other in others))
    bosses = [min(hole.distance(pocket) for pocket in pockets) for hole in holes] if pockets else []
    return {
        "outlineAreaIn2": outline.area,
        "holeCount": len(holes),
        "pocketCount": len(pockets),
        "removedFraction": pocketArea / solidArea if solidArea > 0 else 0.0,
        "wallIn": walls,
        "ribIn": ribs,
        "bossIn": bosses,
        "symmetry": symmetryScore(outline, pockets),
        "pocketCircularity": [circularity(pocket) for pocket in pockets],
    }
