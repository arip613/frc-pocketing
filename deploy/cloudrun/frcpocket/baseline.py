"""Procedural truss baseline: triangulate between features and the perimeter.

Keeps every planar (Delaunay) candidate edge as a rib, so the pockets are the
triangles of that mesh, filleted by the tool. This is the comparison point for the ML
generator and a source of training candidates.
"""

import math

import numpy as np

from frcpocket.layout import candidateGraph, makeLayout


def baselineAnchorSpacing(plate):
    """Sparser wall anchors than the ML graph, so the plain triangulation isn't all slivers."""
    return float(np.clip(math.sqrt(plate.outline.area) / 4, 0.75, 2.5))


def trussBaseline(plate, params, anchorSpacing=None):
    graph = candidateGraph(plate, params, anchorSpacing=anchorSpacing or baselineAnchorSpacing(plate), withInterior=False)
    ribs = [edge for edge, planar in zip(graph.edges, graph.isDelaunay) if planar]
    return makeLayout(plate, params, graph, ribs, name="trussBaseline")
