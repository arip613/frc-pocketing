"""PNG previews of plates: material in grey, preserved holes outlined blue, pockets red."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon

materialColor = "#9aa4ad"
holeEdgeColor = "#1f5fbf"
pocketEdgeColor = "#c0392b"
blindColor = "#d9c27a"


def drawPlate(axis, outline, holes=(), pockets=(), blindPockets=(), title=None):
    axis.add_patch(PatchPolygon(outline, closed=True, facecolor=materialColor, edgecolor="black", linewidth=0.8))
    for blind in blindPockets:
        axis.add_patch(PatchPolygon(blind, closed=True, facecolor=blindColor, edgecolor="#8a6d1f", linewidth=0.6))
    for hole in holes:
        axis.add_patch(PatchPolygon(hole, closed=True, facecolor="white", edgecolor=holeEdgeColor, linewidth=0.7))
    for pocket in pockets:
        axis.add_patch(PatchPolygon(pocket, closed=True, facecolor="white", edgecolor=pocketEdgeColor, linewidth=0.7))
    xs = [point[0] for point in outline]
    ys = [point[1] for point in outline]
    pad = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys))
    axis.set_xlim(min(xs) - pad, max(xs) + pad)
    axis.set_ylim(min(ys) - pad, max(ys) + pad)
    axis.set_aspect("equal")
    axis.axis("off")
    if title:
        axis.set_title(title, fontsize=7)


def renderRecord(record, path, title=None):
    figure, axis = plt.subplots(figsize=(4, 4), dpi=120)
    drawPlate(axis, record["outline"], record["holes"], record["pockets"], record.get("blindPockets", ()), title)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def contactSheet(records, path, titles=None, columns=6, cellSize=2.6):
    rows = max(1, math.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * cellSize, rows * cellSize), dpi=110)
    axes = [axes] if rows * columns == 1 else list(getattr(axes, "flat", axes))
    for index, axis in enumerate(axes):
        if index < len(records):
            record = records[index]
            drawPlate(axis, record["outline"], record["holes"], record["pockets"], record.get("blindPockets", ()),
                      titles[index] if titles else None)
        else:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
