"""Crop the face-on plate view out of each page of a drawing package, for imageExtract.

Team 2056's packages are one part per letter page: a border, a title block along the
bottom, a shaded isometric view, a thin side view, and the dimensioned face view. The
face view is the line drawing that encloses the most area, so each page is rendered,
the border, title block and shaded views are blanked, and the largest enclosed region
(with its dimensions around it) is cropped. Pages that are assemblies or not plates fail
later in imageExtract or the plate filters.

    python -I -m frcpocket.drawingPages data/raw/drawings/2056/*.pdf --out data/raw/drawings/2056/pages
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

renderDpi = 150
inkLevel = 220  # lines are thin and anti-aliased: keep light grey edges so outlines stay connected
marginFraction = 0.035  # the border frame sits inside this margin
titleBlockFraction = 0.215  # the title block and logo strip along the bottom
cropPadPixels = 25


def blankShading(gray):
    """White out shaded regions (isometric renders): wide areas of mid-grey fill."""
    shaded = (gray > 90) & (gray < 235)
    shaded = ndimage.binary_opening(shaded, np.ones((5, 5)))
    labels, count = ndimage.label(ndimage.binary_dilation(shaded, np.ones((15, 15))))
    out = gray.copy()
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[box] == index
        if region.sum() > 0.004 * gray.size:
            out[box][region] = 255
    return out


def faceViewBox(gray):
    """Bounding box of the face view: the connected line drawing with the largest
    bounding box that isn't a thin strip (the side view) or a lone straight line."""
    ink = ndimage.binary_closing(gray < inkLevel, np.ones((3, 3)))
    labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
    best, bestArea = None, 0
    minSide = 0.06 * min(gray.shape)
    for box in ndimage.find_objects(labels):
        height, width = box[0].stop - box[0].start, box[1].stop - box[1].start
        if min(height, width) < minSide:
            continue
        if height * width > bestArea:
            best, bestArea = box, height * width
    if best is None or bestArea < 0.01 * gray.size:
        return None
    return best[0].start, best[0].stop, best[1].start, best[1].stop


def cropPage(image):
    gray = np.asarray(image.convert("L")).astype(np.uint8)
    height, width = gray.shape
    top, left = int(marginFraction * height), int(marginFraction * width)
    bottom, right = int((1 - titleBlockFraction) * height), int((1 - marginFraction) * width)
    work = np.full_like(gray, 255)
    work[top:bottom, left:right] = gray[top:bottom, left:right]
    work = blankShading(work)
    box = faceViewBox(work)
    if box is None:
        return None
    r0, r1, c0, c1 = box
    r0, c0 = max(0, r0 - cropPadPixels), max(0, c0 - cropPadPixels)
    r1, c1 = min(height, r1 + cropPadPixels), min(width, c1 + cropPadPixels)
    # Keep dimension lines near the view (imageExtract removes them) but nothing else.
    crop = np.full((r1 - r0, c1 - c0), 255, dtype=np.uint8)
    crop[:] = work[r0:r1, c0:c1]
    return Image.fromarray(crop).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    outDir = Path(args.out)
    outDir.mkdir(parents=True, exist_ok=True)
    for pdf in map(Path, args.pdfs):
        with tempfile.TemporaryDirectory() as work:
            subprocess.run(["pdftoppm", "-r", str(renderDpi), "-png", str(pdf), f"{work}/p"], check=True)
            pages = sorted(Path(work).glob("p-*.png"))
            kept = 0
            for page in pages:
                crop = cropPage(Image.open(page))
                if crop is None:
                    continue
                crop.save(outDir / f"{pdf.stem}_{page.stem.split('-')[-1]}.png")
                kept += 1
        print(f"{pdf.name}: {kept} of {len(pages)} pages cropped", flush=True)


if __name__ == "__main__":
    main()
