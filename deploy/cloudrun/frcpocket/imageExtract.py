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
    mask, style = materialMask(rgb)
    minLoopPixels = max(6.0, 2e-5 * mask.size)
    loops = [toPolygon(loop).simplify(0.5) for loop in maskLoops(mask)]
    loops = [loop for loop in loops if loop.area >= minLoopPixels]
    if not loops:
        raise ValueError(f"{path}: no plate found")
    outline = max(loops, key=lambda loop: loop.area)
    inner = [loop for loop in loops if loop is not outline and outline.contains(loop.representative_point())]
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
