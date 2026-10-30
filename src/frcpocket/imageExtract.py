"""Extract plate geometry (outline, holes, pockets) from a screenshot of a pocketed plate.

For example images the user provides as gold. Two drawing styles are handled:
  filled   the plate is a solid colour on a contrasting background (most CAD views)
  outline  only the edges are drawn (thin lines); material is the region just inside
           the outer edge, and every other enclosed region is a hole or pocket
Scale comes from the bolt holes: the most common small circle is taken to be a #10
clearance hole (0.201 in), and the large bores are printed so the guess can be checked.

    python -I -m frcpocket.imageExtract data/userExamples/images --out data/processed/userExamples.jsonl
"""

import argparse
import json
import math
from pathlib import Path

import contourpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Polygon
from shapely.validation import make_valid

from frcpocket import geom2d

boltHoleIn = 0.201
bearingBoreIn = 1.125  # 1/2 in hex flanged bearing, the usual big bore in FRC plates
roundTolerance = 0.06  # (max - min radius) / mean radius below this counts as a round hole
inkLevel = 235  # gray below this is a drawn line in drawing mode
edgeReach = 4  # pixels across one drawn line, from the outside into the material
sealGap = 3  # drawn lines are thickened this much when telling outside from material


def loadRgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32)


def backgroundColour(rgb):
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return np.median(border, axis=0)


def largestComponent(mask):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def otsuThreshold(values):
    histogram, edges = np.histogram(values, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2
    weights = histogram.astype(float)
    best, bestScore = centres[0], -1.0
    for index in range(1, len(centres)):
        low, high = weights[:index].sum(), weights[index:].sum()
        if low == 0 or high == 0:
            continue
        meanLow = (weights[:index] * centres[:index]).sum() / low
        meanHigh = (weights[index:] * centres[index:]).sum() / high
        score = low * high * (meanLow - meanHigh) ** 2
        if score > bestScore:
            best, bestScore = centres[index], score
    return best


def materialMask(rgb):
    """Boolean mask of plate material, plus which drawing style was detected."""
    distance = np.linalg.norm(rgb - backgroundColour(rgb), axis=2)
    threshold = max(12.0, otsuThreshold(distance.ravel()))
    allDrawn = distance > threshold
    drawn = largestComponent(allDrawn)
    filled = ndimage.binary_fill_holes(drawn)
    if drawn.sum() / max(filled.sum(), 1) > 0.25:
        return ndimage.binary_opening(drawn, iterations=1), "filled"
    # Outline drawing: every line inside the outer edge counts (pocket loops aren't connected
    # to it), slightly thickened to close anti-aliasing gaps. Material is the enclosed region
    # touching the inside of the outer edge; every other enclosed region is a hole or pocket.
    drawn = ndimage.binary_dilation(allDrawn & filled, iterations=1)
    inside = filled & ~drawn
    labels, count = ndimage.label(inside)
    edgeBand = filled & ~ndimage.binary_erosion(filled, iterations=6)
    touching = np.unique(labels[edgeBand & inside])
    touching = touching[touching > 0]
    material = np.isin(labels, touching) | drawn
    # Lines between pockets are drawn on both sides of a rib; close them into material.
    return ndimage.binary_closing(material, iterations=2), "outline"


def isLineDrawing(rgb):
    """Engineering drawing: mostly white page, thin dark lines, almost nothing solid."""
    gray = rgb.mean(axis=2)
    ink = gray < inkLevel
    if (gray > inkLevel).mean() < 0.93 or ink.sum() == 0:
        return False
    solid = ndimage.binary_erosion(ink, np.ones((3, 3)))
    return solid.sum() / ink.sum() < 0.2


def strokeWidth(rgb):
    """Typical drawn line width in pixels: twice the median distance from the line's
    centre (its skeleton) to its edge."""
    ink = rgb.mean(axis=2) < inkLevel
    distance = ndimage.distance_transform_edt(ink)
    centre = thinMask(ink)
    return float(2 * np.median(distance[centre])) if centre.any() else 1.0


def thinMask(mask):
    """Zhang-Suen thinning to 1-px skeletons."""
    image = np.pad(mask, 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2]]
            count = sum(p)
            transitions = sum((p[k] == 0) & (p[(k + 1) % 8] == 1) for k in range(8))
            if step == 0:
                keep = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                keep = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (image[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & keep
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def neighbourCount(mask):
    counts = ndimage.convolve(mask.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant")
    return np.where(mask, counts - 1, 0)


def segmentMask(shape, start, end, radius):
    """Pixels within radius of the segment start-end (points as x, y)."""
    mask = np.zeros(shape, dtype=bool)
    start, end = np.asarray(start, float), np.asarray(end, float)
    x0, x1 = int(max(0, min(start[0], end[0]) - radius - 1)), int(min(shape[1], max(start[0], end[0]) + radius + 2))
    y0, y1 = int(max(0, min(start[1], end[1]) - radius - 1)), int(min(shape[0], max(start[1], end[1]) + radius + 2))
    if x1 <= x0 or y1 <= y0:
        return mask
    ys, xs = np.mgrid[y0:y1, x0:x1]
    delta = end - start
    along = np.clip(((xs - start[0]) * delta[0] + (ys - start[1]) * delta[1]) / max(delta @ delta, 1e-9), 0, 1)
    distance = np.hypot(xs - start[0] - along * delta[0], ys - start[1] - along * delta[1])
    mask[y0:y1, x0:x1] = distance <= radius
    return mask


def inkAt(mask, point):
    x, y = int(round(point[0])), int(round(point[1]))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def arrowheads(gray):
    """Filled arrowheads (the only solid blobs in a line drawing) as tip, unit direction, blob mask."""
    dark = gray < 110
    core = ndimage.binary_opening(dark, np.ones((3, 3)))
    labels, count = ndimage.label(core)
    arrows = []
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        blob = labels[region] == index
        if not 25 <= blob.sum() <= 600:
            continue
        pad = (slice(max(region[0].start - 3, 0), region[0].stop + 3), slice(max(region[1].start - 3, 0), region[1].stop + 3))
        full = np.zeros(gray.shape, dtype=bool)
        full[region] = blob
        full = dark & ndimage.binary_dilation(full, iterations=2)
        ys, xs = np.nonzero(full[pad])
        points = np.column_stack([xs + pad[1].start, ys + pad[0].start]).astype(float)
        centre = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points - centre).T))
        axis = vectors[:, 1]
        along = (points - centre) @ axis
        across = (points - centre) @ vectors[:, 0]
        length, width = along.max() - along.min(), across.max() - across.min()
        if length < 2 * max(width, 1) or length < 8:
            continue
        if along.max() < -along.min():
            axis = -axis
            along = -along
        tip = centre + along.max() * axis
        for _ in range(6):
            if not inkAt(gray < 200, tip + axis):
                break
            tip = tip + axis
        arrows.append({"tip": tip, "direction": axis, "mask": full, "length": length})
    return arrows


def alignToLine(ink, start, direction, spread=4.0, maxGap=3):
    """The shaft direction within spread degrees of direction that runs over the most ink from
    start, and how far that ink goes."""
    best, bestHits, bestReach = direction, -1, 0
    for degrees in np.arange(-spread, spread + 0.01, 0.25):
        angle = math.radians(degrees)
        turned = np.array([direction[0] * math.cos(angle) - direction[1] * math.sin(angle),
                           direction[0] * math.sin(angle) + direction[1] * math.cos(angle)])
        reach, hits = walkLine(ink, start, turned, maxGap=maxGap, countHits=True)
        if hits > bestHits:
            best, bestHits, bestReach = turned, hits, reach
    return best, bestReach


def walkLine(ink, start, direction, maxGap=3, limit=2000, countHits=False):
    """Distance along a ray from start for which ink continues (tolerating short gaps), and
    optionally how many steps found ink."""
    normal = np.array([-direction[1], direction[0]])
    reached, gap, step, hits = 0, 0, 0, 0
    while step < limit:
        step += 1
        point = start + step * direction
        if not (0 <= point[0] < ink.shape[1] and 0 <= point[1] < ink.shape[0]):
            break
        if any(inkAt(ink, point + offset * normal) for offset in (-1, 0, 1)):
            reached, gap, hits = step, 0, hits + 1
        else:
            gap += 1
            if gap > maxGap:
                break
    return (reached, hits) if countHits else reached


def pairDimensions(arrows, ink):
    """Dimension lines: two arrowheads on one axis, pointing apart, joined by ink."""
    candidates = []
    for i, first in enumerate(arrows):
        for j in range(i + 1, len(arrows)):
            second = arrows[j]
            if first["direction"] @ second["direction"] > -0.99:
                continue
            span = second["tip"] - first["tip"]
            length = np.hypot(*span)
            if length < 30:
                continue
            unit = span / length
            if abs(unit @ first["direction"]) < 0.995 or unit @ first["direction"] > 0:
                continue
            samples = [first["tip"] + t * span for t in np.linspace(0, 1, 50)]
            normal = np.array([-unit[1], unit[0]])
            covered = np.mean([any(inkAt(ink, point + offset * normal) for offset in (-1, 0, 1)) for point in samples])
            if covered > 0.8:
                candidates.append((abs(unit[0] * first["direction"][1] - unit[1] * first["direction"][0]), i, j))
    used, dimensions = set(), []
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        used.update((i, j))
        dimensions.append((arrows[i], arrows[j]))
    return dimensions, [arrow for index, arrow in enumerate(arrows) if index not in used]


def removeAnnotation(ink, gray):
    """Erase dimension lines, leaders, and extension lines up to where they cross. Returns the
    cleaned lines, what was erased, the arrowheads, the extension-line crossings (cut too, but
    one may be a corner of the part), and the dimensions (pairs of arrowheads)."""
    arrows = arrowheads(gray)
    dimensions, leaders = pairDimensions(arrows, ink)
    # Arrowheads stay: they are solid, so they enclose nothing, and part edges often run
    # underneath them. Only the shafts go.
    erase = np.zeros(ink.shape, dtype=bool)
    arrowMask = np.zeros(ink.shape, dtype=bool)
    for arrow in arrows:
        arrowMask |= arrow["mask"]
    arrowMask = ink & ndimage.binary_dilation(arrowMask)
    base = lambda arrow: arrow["tip"] - (arrow["length"] + 1) * arrow["direction"]
    for first, second in dimensions:
        erase |= segmentMask(ink.shape, base(first), base(second), 1.6)
    shaftEnds = []
    for arrow in leaders:
        shaft, reach = alignToLine(ink, arrow["tip"], -arrow["direction"])
        # Then follow it across the halo gaps where it passes over part edges.
        shaft, reach = alignToLine(ink, arrow["tip"], shaft, spread=1.0, maxGap=24)
        if reach > arrow["length"] + 1:
            erase |= segmentMask(ink.shape, base(arrow), arrow["tip"] + reach * shaft, 1.6)
        shaftEnds.append((arrow["tip"] + reach * shaft, shaft))
    # A section line (dashed, broken wherever it crosses the part) joins the shaft ends of its
    # two view arrows.
    for i, (first, shaft) in enumerate(shaftEnds):
        for second, other in shaftEnds[i + 1:]:
            span = second - first
            length = np.hypot(*span)
            if length < 30 or abs(shaft @ other) < 0.99 or abs(span @ shaft) / length > 0.05:
                continue
            samples = [first + t * span for t in np.linspace(0, 1, 100)]
            covered = np.mean([any(inkAt(ink, point + offset * shaft) for offset in (-1, 0, 1)) for point in samples])
            if covered > 0.5:
                erase |= segmentMask(ink.shape, first, second, 1.6)
    # Extension lines run through each dimension tip, perpendicular to the dimension line, and
    # end on the part, often tangent to an edge, so they mostly stay. With the dimension line
    # gone they enclose nothing, except where two of them cross: cut each one from its tip to
    # past the last crossing.
    extensions = []
    for first, second in dimensions:
        span = second["tip"] - first["tip"]
        normal = np.array([-span[1], span[0]]) / np.hypot(*span)
        for tip in (first["tip"], second["tip"]):
            along = np.array([normal[1], -normal[0]])
            # The arrow tip stops a pixel or two short of the extension line; slide onto it.
            reaches = {}
            for shift in range(-6, 7):
                reaches[shift] = [walkLine(ink, tip + shift * along, direction) for direction in (normal, -normal)]
            best = max(max(value) for value in reaches.values())
            onLine = [key for key in reaches if max(reaches[key]) >= 0.9 * best]
            shift = onLine[len(onLine) // 2]
            forward, backward = reaches[shift]
            direction = normal if forward >= backward else -normal
            start = tip + shift * along
            extensions.append({"start": start, "direction": direction, "reach": max(forward, backward),
                               "overshoot": min(forward, backward), "cut": 0.0})
    junctions = []
    for i, first in enumerate(extensions):
        for second in extensions[i + 1:]:
            denominator = first["direction"][0] * second["direction"][1] - first["direction"][1] * second["direction"][0]
            if abs(denominator) < 0.2:
                continue
            gap = second["start"] - first["start"]
            alongFirst = (gap[0] * second["direction"][1] - gap[1] * second["direction"][0]) / denominator
            alongSecond = (gap[0] * first["direction"][1] - gap[1] * first["direction"][0]) / denominator
            if 0 < alongFirst <= first["reach"] + 2 and 0 < alongSecond <= second["reach"] + 2:
                first["cut"] = max(first["cut"], alongFirst - 4)
                second["cut"] = max(second["cut"], alongSecond - 4)
                crossing = first["start"] + alongFirst * first["direction"]
                junctions.append(segmentMask(ink.shape, crossing - 4 * first["direction"], crossing + 3 * first["direction"], 1.6)
                                 | segmentMask(ink.shape, crossing - 4 * second["direction"], crossing + 3 * second["direction"], 1.6))
    for line in extensions:
        start, direction = line["start"], line["direction"]
        erase |= segmentMask(ink.shape, start - line["overshoot"] * direction, start + line["cut"] * direction, 1.6)
    for junction in junctions:
        erase |= junction
    cleaned = ink & ~erase
    labels, count = ndimage.label(cleaned, np.ones((3, 3)))
    sizes = ndimage.sum(cleaned, labels, range(1, count + 1))
    cleaned &= np.isin(labels, np.nonzero(sizes >= 12)[0] + 1)
    return {"lines": repairCrossings(cleaned, erase), "erased": erase, "arrows": arrowMask,
            "junctions": junctions, "dimensions": dimensions}


def localDirection(ink, centre, radius=6):
    """Unit direction of the ink line around centre, or None when it isn't line-like."""
    x, y = int(round(centre[0])), int(round(centre[1]))
    window = ink[max(y - radius, 0):y + radius + 1, max(x - radius, 0):x + radius + 1]
    ys, xs = np.nonzero(window)
    if len(xs) < 4:
        return None
    points = np.column_stack([xs, ys]).astype(float)
    values, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
    return vectors[:, 1] if values[1] > 4 * max(values[0], 1e-6) else None


def repairCrossings(cleaned, erase):
    """Re-join part edges cut where erased annotation crossed them: pair the cut ends on either
    side of each erased patch when the edge runs straight across."""
    patches, _ = ndimage.label(erase, np.ones((3, 3)))
    near = ndimage.binary_dilation(erase, iterations=2)
    ports, count = ndimage.label(cleaned & near, np.ones((3, 3)))
    if count == 0:
        return cleaned
    centres = ndimage.center_of_mass(cleaned & near, ports, range(1, count + 1))
    owner = ndimage.grey_dilation(patches, size=(5, 5))
    info = []
    for index, (cy, cx) in enumerate(centres):
        centre = np.array([cx, cy])
        patch = int(owner[int(round(cy)), int(round(cx))])
        direction = localDirection(cleaned & ~ndimage.binary_dilation(erase, iterations=1), centre) if patch else None
        info.append((centre, patch, direction))
    candidates = []
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            (p, patchP, dp), (q, patchQ, dq) = info[i], info[j]
            if not patchP or patchP != patchQ or dp is None or dq is None:
                continue
            gap = q - p
            distance = np.hypot(*gap)
            if not 2 < distance <= 30:
                continue
            unit = gap / distance
            alignment = 0.85 if distance <= 14 else 0.97
            if abs(dp @ unit) < alignment or abs(dq @ unit) < alignment or not inkAt(erase, (p + q) / 2):
                continue
            candidates.append((distance, i, j))
    repaired, used = cleaned.copy(), set()
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        used.update((i, j))
        repaired |= segmentMask(cleaned.shape, info[i][0], info[j][0], 0.75)
    return repaired


def endpointDirections(skeleton, depth=8):
    """Skeleton endpoints with the unit direction the line was heading as it ended, and how far
    back along the skeleton that direction was measured."""
    counts = neighbourCount(skeleton)
    result = []
    for y, x in np.argwhere(counts == 1):
        seen, frontier, reached = {(y, x)}, [(y, x)], 0
        for level in range(1, depth + 1):
            frontier = [(cy + dy, cx + dx) for cy, cx in frontier for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if 0 <= cy + dy < skeleton.shape[0] and 0 <= cx + dx < skeleton.shape[1]
                        and skeleton[cy + dy, cx + dx] and (cy + dy, cx + dx) not in seen]
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            seen.update(frontier)
            reached, far = level, frontier
        if reached == 0:
            continue
        back = np.mean(far, axis=0)
        vector = np.array([x - back[1], y - back[0]], float)
        norm = np.hypot(*vector)
        if norm > 0:
            result.append((np.array([x, y], float), vector / norm, reached))
    return result


def closeGaps(ink, erased, arrowMask, maxGap=40):
    """Bridge breaks in part edges (the halo around annotation erases a few pixels, sometimes a
    corner): join skeleton endpoints that face each other, straight or through the corner where
    their extensions meet. Edges running under an arrowhead are traced as if it weren't there.
    Ends cut by erasing annotation only join straight across, so cut extension lines don't
    close up again."""
    skeleton = thinMask(ink & ~ndimage.binary_dilation(arrowMask))
    ends = endpointDirections(skeleton)
    nearErased = ndimage.binary_dilation(erased, iterations=3)
    nearArrow = ndimage.binary_dilation(arrowMask, iterations=3)
    erasedEnd = [inkAt(nearErased, point) for point, _, _ in ends]
    cut = [erasedEnd[index] or inkAt(nearArrow, point) for index, (point, _, _) in enumerate(ends)]
    candidates = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (p, dp, lp), (q, dq, lq) = ends[i], ends[j]
            gap = q - p
            distance = np.hypot(*gap)
            if distance > maxGap or distance == 0:
                continue
            if distance <= 4:
                candidates.append((distance, i, j, None))
                continue
            if min(lp, lq) < (3 if distance <= 12 else 6):
                continue
            unit = gap / distance
            facing = min(dp @ unit, -(dq @ unit))
            if facing > (0.95 if erasedEnd[i] or erasedEnd[j] else 0.9 if cut[i] or cut[j] or distance > 30 else 0.85):
                candidates.append((distance * (2 - facing), i, j, None))
                continue
            if erasedEnd[i] or erasedEnd[j] or distance > 30:
                continue
            denominator = dp[0] * dq[1] - dp[1] * dq[0]
            if abs(denominator) < 0.2:
                continue
            along = (gap[0] * dq[1] - gap[1] * dq[0]) / denominator
            back = (gap[0] * dp[1] - gap[1] * dp[0]) / denominator
            if 0 < along <= 20 and 0 < back <= 20:
                candidates.append((1.5 * (along + back), i, j, p + along * dp))
    bridged = ink.copy()
    used = set()
    for _, i, j, corner in sorted(candidates, key=lambda item: item[0]):
        if i in used or j in used:
            continue
        used.update((i, j))
        if corner is None:
            bridged |= segmentMask(ink.shape, ends[i][0], ends[j][0], 0.75)
        else:
            bridged |= segmentMask(ink.shape, ends[i][0], corner, 0.75) | segmentMask(ink.shape, corner, ends[j][0], 0.75)
    # An edge whose partner merged into another line (an extension line running on from it)
    # has no end to pair with: run it straight on until it meets ink.
    for index, (point, direction, length) in enumerate(ends):
        if index in used or erasedEnd[index] or length < 6:
            continue
        for step in range(3, 26):
            target = point + step * direction
            if inkAt(skeleton, target) or inkAt(skeleton, target + (1, 0)) or inkAt(skeleton, target + (0, 1)):
                if step > 4:
                    bridged |= segmentMask(ink.shape, point, target, 0.75)
                break
    return bridged


def materialRegions(lines, seal):
    """White regions that are plate material. The lines are first thickened by seal pixels so
    small breaks can't let the outside leak in; the labels are then grown back over the white
    pixels the thickening covered."""
    thick = ndimage.binary_dilation(lines, iterations=seal) if seal else lines
    labels, count = ndimage.label(~thick)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    # Material regions sit one line in from the outside; holes and pockets are further in.
    nearOutside = ndimage.binary_dilation(outside, iterations=edgeReach + 2 * seal)
    touching = np.unique(labels[nearOutside & ~outside & ~thick])
    material = np.isin(labels, touching[touching > 0])
    voids = ~thick & ~outside & ~material
    free = ~lines
    while True:
        claimed = outside | material | voids
        grown = False
        for kind in (material, voids, outside):
            step = ndimage.binary_dilation(kind) & free & ~claimed
            if step.any():
                kind |= step
                claimed |= step
                grown = True
        if not grown:
            return material


def solidFromLines(lines):
    material = materialRegions(lines, sealGap)
    solid = material | (lines & ndimage.binary_dilation(material, iterations=2))
    solid = ndimage.binary_opening(ndimage.binary_closing(solid, iterations=1), iterations=1)
    solid = largestComponent(solid)
    voids = ndimage.binary_fill_holes(solid) & ~solid
    voidLabels, voidCount = ndimage.label(voids)
    if voidCount:
        # Voids that are only ink (an arrowhead lying in the material) or a few pixels are solid.
        openArea = ndimage.sum(voids & ~lines, voidLabels, range(1, voidCount + 1))
        small = openArea < 12
        solid |= np.isin(voidLabels, np.nonzero(small)[0] + 1)
        # A countersink draws a second circle round the hole; the ring between them is material.
        regions, _ = ndimage.label(voids & ~lines & ~solid)
        for index, region in enumerate(ndimage.find_objects(regions), start=1):
            ring = regions[region] == index
            inside = ndimage.binary_fill_holes(ring) & ~ring
            if (inside & (regions[region] > 0)).sum() >= 12:
                solid[region] |= ring | (inside & lines[region] & ndimage.binary_dilation(ring, iterations=2))
        voidCount = ndimage.label(ndimage.binary_fill_holes(solid) & ~solid)[1]
    return solid, voidCount


def drawingMask(rgb):
    """Material mask for a line drawing, plus the dimensions found on it."""
    gray = rgb.mean(axis=2)
    ink = gray < inkLevel
    annotation = removeAnnotation(ink, gray)
    lines = closeGaps(annotation["lines"], annotation["erased"], annotation["arrows"])
    solid, voidCount = solidFromLines(lines)
    # Two extension lines can cross exactly on a corner of the part; then cutting the crossing
    # opens the outline and the plate loses its holes. Keep the crossing when that brings back
    # clearly more of them.
    for junction in annotation["junctions"]:
        restored = lines | (ink & junction)
        trialSolid, trialCount = solidFromLines(restored)
        if trialCount > 1.5 * voidCount + 2:
            lines, solid, voidCount = restored, trialSolid, trialCount
    return solid, annotation["dimensions"]


def maskLoops(mask):
    """Closed boundary loops of a mask as (N, 2) arrays in pixel coordinates (x right, y down)."""
    padded = np.pad(mask.astype(float), 2)
    generator = contourpy.contour_generator(z=padded, line_type="Separate")
    loops = []
    for line in generator.lines(0.5):
        if len(line) >= 8:
            loops.append(line - 2)
    return loops


def toPolygon(points):
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
        polygon = max(getattr(polygon, "geoms", [polygon]), key=lambda part: part.area)
    return polygon


def isRound(polygon):
    """Round hole test by radius spread around the centroid (with a pixel allowance, since
    small traced circles are a few pixels ragged). Large octagonal or rounded-square pockets
    fail it; very small low-resolution octagons can still pass."""
    centre = np.array(polygon.centroid.coords[0])
    radii = np.hypot(*(np.asarray(polygon.exterior.coords) - centre).T)
    return (radii.max() - radii.min()) / max(radii.mean(), 1e-9) < roundTolerance + 1.5 / max(radii.mean(), 1e-9)


def pixelScale(diameters):
    """Pixels per inch from bolt holes (#10) and, when present, a bearing bore (1.125 in)."""
    small = np.percentile(diameters, 30)
    fromBolts = float(np.median(diameters[diameters <= small * 1.25])) / boltHoleIn
    largest = float(diameters.max())
    fromBore = largest / bearingBoreIn
    if largest / small > 3.5:  # there is a real bore to check against
        if abs(fromBolts / fromBore - 1) < 0.25:
            return (fromBolts + fromBore) / 2
        # Disagreement usually means the bolt holes are only a few pixels (low-res image).
        return fromBore if small < 12 else fromBolts
    return fromBolts


def extract(path):
    rgb = loadRgb(path)
    dimensions = []
    if isLineDrawing(rgb):
        mask, dimensions = drawingMask(rgb)
        style = "drawing"
    else:
        mask, style = materialMask(rgb)
    minLoopPixels = max(6.0, 2e-5 * mask.size)
    loops = [toPolygon(loop).simplify(0.5) for loop in maskLoops(mask)]
    loops = [loop for loop in loops if loop.area >= minLoopPixels]
    if not loops:
        raise ValueError(f"{path}: no plate found")
    outline = max(loops, key=lambda loop: loop.area)
    inner = [loop for loop in loops if loop is not outline and outline.contains(loop.representative_point())]
    if style == "drawing":
        # Loops are traced on the inside of each drawn line; move them to the line's
        # centre, where the dimensions are, or small holes read ~10% small and skew the scale.
        half = strokeWidth(rgb) / 2
        outline = Polygon(outline.exterior).buffer(-half, join_style=2)
        outline = max(getattr(outline, "geoms", [outline]), key=lambda part: part.area)
        inner = [loop.buffer(half, join_style=2) for loop in inner]
    holes = [loop for loop in inner if isRound(loop)]
    pockets = [loop for loop in inner if not isRound(loop)]
    if not holes:
        raise ValueError(f"{path}: no round holes found to set the scale")
    diameters = np.array([2 * math.sqrt(hole.area / math.pi) for hole in holes])
    pixelsPerInch = pixelScale(diameters)
    # Image y points down; flip so the plate isn't mirrored, and scale to inches.
    def inches(polygon):
        scaled = affinity.scale(polygon, xfact=1 / pixelsPerInch, yfact=-1 / pixelsPerInch, origin=(0, 0))
        return scaled.buffer(0)
    return {
        "style": style,
        "pixelsPerInch": pixelsPerInch,
        "outline": inches(Polygon(outline.exterior)),
        "holes": [inches(hole) for hole in holes],
        "pockets": [inches(pocket) for pocket in pockets],
        "holeDiametersIn": sorted(float(d / pixelsPerInch) for d in diameters),
        "pixelLoops": (outline, holes, pockets),
    }


def asRecord(name, result, sourcePath):
    outline = result["outline"]
    holes = [hole for hole in result["holes"] if hole.area > 0]
    pockets = [pocket for pocket in result["pockets"] if pocket.area > 0]
    stats = geom2d.pocketStats(outline, holes, pockets)
    return {
        "id": f"userImg_{name}",
        "file": str(sourcePath),
        "format": "image",
        "origin": "userImage",
        "repo": "user-provided example",
        "license": "provided by the user as a gold example",
        "trainingAllowed": True,
        "thicknessIn": None,
        "outline": [list(point) for point in outline.exterior.coords],
        "holes": [[list(point) for point in hole.exterior.coords] for hole in holes],
        "pockets": [[list(point) for point in pocket.exterior.coords] for pocket in pockets],
        "blindPockets": [],
        "pocketMeta": [{"nLines": geom2d.straightRunCount(pocket, minLength=0.15), "arcRadiiIn": []} for pocket in pockets],
        "stats": stats,
        "solidIndex": 0,
    }


def overlay(path, result, outPath):
    rgb = loadRgb(path).astype(np.uint8)
    outline, holes, pockets = result["pixelLoops"]
    figure, axis = plt.subplots(figsize=(6, 6 * rgb.shape[0] / rgb.shape[1]), dpi=90)
    axis.imshow(rgb)
    axis.plot(*outline.exterior.xy, color="yellow", lw=1.2)
    for hole in holes:
        axis.plot(*hole.exterior.xy, color="cyan", lw=1.0)
    for pocket in pockets:
        axis.plot(*pocket.exterior.xy, color="lime", lw=1.0)
    axis.set_title(f"{Path(path).name}: {len(holes)} holes, {len(pockets)} pockets ({result['style']})", fontsize=8)
    axis.axis("off")
    figure.savefig(outPath, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images")
    parser.add_argument("--out", default="data/processed/userExamples.jsonl")
    parser.add_argument("--previews", default="data/userExamples/extracted")
    args = parser.parse_args()
    previewDir = Path(args.previews)
    previewDir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(Path(args.images).glob("*.png")):
        try:
            result = extract(path)
        except ValueError as error:
            print(f"skip {path.name}: {error}")
            continue
        record = asRecord(path.stem, result, path)
        records.append(record)
        overlay(path, result, previewDir / f"{path.stem}_overlay.png")
        bounds = result["outline"].bounds
        print(f"{path.name}: {result['style']}, {result['pixelsPerInch']:.1f} px/in, "
              f"{bounds[2] - bounds[0]:.1f} x {bounds[3] - bounds[1]:.1f} in, {len(result['holes'])} holes "
              f"(largest {max(result['holeDiametersIn']):.2f} in), {len(result['pockets'])} pockets, "
              f"{record['stats']['removedFraction']:.0%} removed")
    with open(args.out, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
