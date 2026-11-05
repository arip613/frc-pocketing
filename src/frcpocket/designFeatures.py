"""Scale-free measurements of how a pocketing layout looks (inputs to the aesthetic scorer).

Each feature maps to something in docs/designNotes.md: uniform pockets and ribs,
no slivers or salt-and-pepper pockets, symmetric when the plate is, pockets that
work around the holes, no big unpocketed regions, triangles over windows, and no
single giant pocket that guts the plate.
Only the pocket pattern is described (relative to the plate), never the outline itself,
so the scorer can't prefer a plate just for its shape.
"""

import math

import numpy as np
from shapely import affinity
from shapely.ops import unary_union

from frcpocket import geom2d

featureNames = [
    "removedFraction",
    "logPocketsPerScale",
    "pocketAreaCV",
    "smallPocketFraction",
    "sliverFraction",
    "meanConvexity",
    "meanCircularity",
    "gapCV",
    "gapOverRib",
    "wallGapCV",
    "symmetryShortfall",
    "holesEngaged",
    "largestSolidRegion",
    "triangleFraction",
    "logMeanPocketArea",
    "maxPocketFraction",
    "maxOpenSpan",
    "pocketSizeVsGold",
    "pocketWidthVsGold",
    "maxElongation",
    "elongatedFraction",
    "meanFatness",
    "sharpCornerFraction",
    "hangingHoleFraction",
]
sharpCornerDegrees = 45  # pockets with a corner tighter than this read as wedges
# A hole whose ring is pocketed nearly all the way round hangs off a rib or two instead of
# sitting on the web or the wall ("cantilevered bolts", the top review criticism of FRC
# pocketing; see docs/pocketingResearch.md).
hangingOpenShare = 0.75

goldElongationLimit = 2.4  # 95% of gold pockets are at most this long relative to their width

# Gold pocket size versus plate size, fitted on the gold training plates by the scorer
# (log-log: pockets grow with the plate, but slower than its area). scale = sqrt(plate area).
# The "...VsGold" features are |log| distances from the fitted size for this plate, so a
# busy truss (too small) and a gutted or blobby plate (too big) both score as far from gold.
styleReference = {"areaIntercept": math.log(0.83), "areaSlope": 1.0, "spanIntercept": math.log(0.371),
                  "spanSlope": 0.5, "referenceScale": 4.44}


def referencePocketArea(scale):
    ref = styleReference
    return math.exp(ref["areaIntercept"] + ref["areaSlope"] * math.log(scale / ref["referenceScale"]))


def referenceSpan(scale):
    ref = styleReference
    return math.exp(ref["spanIntercept"] + ref["spanSlope"] * math.log(scale / ref["referenceScale"]))


def plateSymmetry(plate):
    """How mirror/rotation symmetric the plate itself is (outline + features)."""
    shape = plate.material
    centre = plate.outline.centroid
    angle = geom2d.principalAngle(plate.outline)
    aligned = affinity.rotate(shape, -angle, origin=centre)
    mirrors = [
        affinity.scale(aligned, xfact=-1, yfact=1, origin=centre),
        affinity.scale(aligned, xfact=1, yfact=-1, origin=centre),
        affinity.rotate(aligned, 180, origin=centre),
    ]
    return max(geom2d.iou(aligned, mirror) for mirror in mirrors)


def largestInscribedRadius(region, step):
    """Radius of the biggest disk that fits in region, to within `step`."""
    low, high = 0.0, math.sqrt(region.area / math.pi) + step
    while high - low > step:
        middle = (low + high) / 2
        if region.buffer(-middle, quad_segs=4).is_empty:
            high = middle
        else:
            low = middle
    return low


def sharpestCorner(pocket):
    """Sharpest corner (degrees) between consecutive straight sides of a pocket.

    Fillets and boss arcs between two sides are ignored, so a wedge whose tip is rounded
    off or cut by a boss still counts as the angle its sides make."""
    coords = np.asarray(pocket.exterior.coords)
    if not pocket.exterior.is_ccw:
        coords = coords[::-1]
    segments = np.diff(coords, axis=0)
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    keep = lengths > 1e-9
    segments, lengths = segments[keep], lengths[keep]
    if len(segments) < 3:
        return 180.0
    angles = np.degrees(np.arctan2(segments[:, 1], segments[:, 0]))
    minLength = 0.15 * math.sqrt(pocket.area)
    sides, start = [], 0
    for index in range(1, len(segments) + 1):
        if index == len(segments) or abs((angles[index] - angles[index - 1] + 180) % 360 - 180) > 1.0:
            if lengths[start:index].sum() > minLength:
                sides.append(angles[start])
            start = index
    if len(sides) < 3:
        return 180.0  # a rounded blob, no straight sides to make a corner
    sharpest = 180.0
    for first, second in zip(sides, sides[1:] + sides[:1]):
        turn = (second - first + 180) % 360 - 180
        if turn > 0:
            sharpest = min(sharpest, 180.0 - turn)
    return sharpest


def medianSpan(pockets, scale):
    return float(np.median([largestInscribedRadius(pocket, scale / 200) for pocket in pockets]))


def goldStyleReference(plates):
    """Fit log(median pocket area) and log(median pocket inscribed radius) against
    log(plate scale) over (plate, pockets) pairs of gold designs. Slopes are clamped so a
    handful of plates can't produce something silly."""
    scales, areas, spans = [], [], []
    for plate, pockets in plates:
        if pockets:
            scale = math.sqrt(plate.materialArea)
            scales.append(math.log(scale))
            areas.append(math.log(float(np.median([pocket.area for pocket in pockets]))))
            spans.append(math.log(medianSpan(pockets, scale)))
    scales, areas, spans = np.array(scales), np.array(areas), np.array(spans)
    centre = float(scales.mean())

    def fit(values, low, high):
        slope = float(np.polyfit(scales - centre, values, 1)[0]) if len(values) > 2 else (low + high) / 2
        return float(values.mean()), float(np.clip(slope, low, high))

    areaIntercept, areaSlope = fit(areas, 0.5, 1.5)
    spanIntercept, spanSlope = fit(spans, 0.25, 0.75)
    return {"areaIntercept": areaIntercept, "areaSlope": areaSlope, "spanIntercept": spanIntercept,
            "spanSlope": spanSlope, "referenceScale": math.exp(centre)}


def hangingHoles(plate, params, pocketUnion, samples=24):
    """Share of the holes (not big cutouts) whose ring has pockets nearly all the way round."""
    from frcpocket.plate import isOpening

    holes = [feature for feature in plate.features if not isOpening(feature)]
    if not holes:
        return 0.0
    near = pocketUnion.buffer(params.ribIn)
    count = 0
    for hole in holes:
        ring = hole.buffer(params.boss + 0.5 * params.ribIn).exterior
        points = [ring.interpolate(k / samples, normalized=True) for k in range(samples)]
        if np.mean([near.contains(point) for point in points]) >= hangingOpenShare:
            count += 1
    return count / len(holes)


def describe(plate, params, pockets, plateSym=None):
    scale = math.sqrt(plate.materialArea)
    count = len(pockets)
    if count == 0:
        values = dict.fromkeys(featureNames, 0.0)
        values["largestSolidRegion"] = largestInscribedRadius(plate.material.buffer(-params.wallIn), scale / 50) / scale
        values["symmetryShortfall"] = 0.0
        values["maxPocketFraction"] = 0.0
        values["maxOpenSpan"] = 0.0
        values["pocketSizeVsGold"] = 3.0
        values["pocketWidthVsGold"] = 3.0
        values["maxElongation"] = 0.0
        values["elongatedFraction"] = 0.0
        values["meanFatness"] = 0.0
        values["sharpCornerFraction"] = 0.0
        values["hangingHoleFraction"] = 0.0
        return np.array([values[name] for name in featureNames], dtype=np.float32)

    areas = np.array([pocket.area for pocket in pockets])
    gaps = []
    for index, pocket in enumerate(pockets):
        others = [other for otherIndex, other in enumerate(pockets) if otherIndex != index]
        if others:
            gaps.append(min(pocket.distance(other) for other in others))
    gaps = np.array(gaps) if gaps else np.array([params.ribIn])
    wallGaps = np.array([plate.outline.exterior.distance(pocket) for pocket in pockets])
    nearWall = wallGaps[wallGaps < 3 * params.wallIn]

    slivers = 0
    triangles = 0
    elongations = []
    for pocket in pockets:
        rectangle = pocket.minimum_rotated_rectangle
        xs, ys = rectangle.exterior.coords.xy
        sides = sorted(math.dist((xs[k], ys[k]), (xs[k + 1], ys[k + 1])) for k in range(2))
        elongations.append(sides[1] / max(sides[0], 1e-9))
        if sides[0] > 0 and sides[1] / sides[0] > 4:
            slivers += 1
        if geom2d.straightRunCount(pocket, minLength=params.ribIn) == 3:
            triangles += 1

    engaged = 0
    for feature in plate.features:
        if min(feature.distance(pocket) for pocket in pockets) <= 2.5 * params.boss:
            engaged += 1

    pocketUnion = unary_union(pockets)
    solid = plate.material.buffer(-params.wallIn).difference(pocketUnion.buffer(params.ribIn))
    hanging = hangingHoles(plate, params, pocketUnion)
    largest = largestInscribedRadius(solid, scale / 50) / scale if not solid.is_empty else 0.0

    plateSym = plateSymmetry(plate) if plateSym is None else plateSym
    pocketSym = geom2d.symmetryScore(plate.outline, pockets)

    values = {
        "removedFraction": float(areas.sum() / plate.materialArea),
        "logPocketsPerScale": math.log1p(count),
        "pocketAreaCV": float(areas.std() / areas.mean()),
        "smallPocketFraction": float(np.mean(areas < 4 * params.minPocketArea)),
        "sliverFraction": slivers / count,
        "meanConvexity": float(np.mean([pocket.area / pocket.convex_hull.area for pocket in pockets])),
        "meanCircularity": float(np.mean([geom2d.circularity(pocket) for pocket in pockets])),
        "gapCV": float(gaps.std() / max(gaps.mean(), 1e-9)),
        "gapOverRib": float(np.median(gaps) / params.ribIn),
        "wallGapCV": float(nearWall.std() / nearWall.mean()) if len(nearWall) > 1 else 0.0,
        "symmetryShortfall": float(max(0.0, plateSym - pocketSym)),
        "holesEngaged": engaged / max(1, len(plate.features)),
        "largestSolidRegion": largest,
        "triangleFraction": triangles / count,
        "logMeanPocketArea": math.log(areas.mean() / plate.materialArea),
        # One giant pocket that guts the plate: largest pocket's share, and the widest
        # unsupported span (largest disk inside the largest pocket).
        "maxPocketFraction": float(areas.max() / plate.materialArea),
        "maxOpenSpan": largestInscribedRadius(pockets[int(np.argmax(areas))], scale / 100) / scale,
        "pocketSizeVsGold": abs(math.log(float(np.median(areas)) / referencePocketArea(scale))),
        "pocketWidthVsGold": abs(math.log(max(medianSpan(pockets, scale), 1e-6) / referenceSpan(scale))),
        # Slots (long pockets between parallel ribs) never appear in the gold plates.
        "maxElongation": math.log(max(elongations)),
        "elongatedFraction": float(np.mean(np.array(elongations) > goldElongationLimit)),
        # Fat, rounded cells (the example plates) versus thin wedges: disk area over pocket
        # area, and the share of pockets with a corner sharper than sharpCornerDegrees.
        "meanFatness": float(np.mean([math.pi * largestInscribedRadius(pocket, scale / 200) ** 2 / pocket.area
                                      for pocket in pockets])),
        "sharpCornerFraction": float(np.mean([sharpestCorner(pocket) < sharpCornerDegrees for pocket in pockets])),
        "hangingHoleFraction": hanging,
    }
    return np.array([values[name] for name in featureNames], dtype=np.float32)
