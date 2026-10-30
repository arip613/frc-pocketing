"""Rasterize a plate design into image channels for the CNN scorer.

Channels: kept material (after pockets), features, and plate outline, drawn
anti-aliased (4x supersampled) in a frame aligned to the plate's principal axis and
scaled to fit. Every design of one plate shares that plate's frame.
"""

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity

from frcpocket import geom2d
from frcpocket.layout import polygons

rasterSize = 96
supersample = 4


def plateFrame(plate):
    """Affine map (as a function) taking plate inches to pixel coordinates."""
    centre = plate.outline.centroid
    angle = geom2d.principalAngle(plate.outline)
    aligned = affinity.rotate(plate.outline, -angle, origin=centre)
    minx, miny, maxx, maxy = aligned.bounds
    span = max(maxx - minx, maxy - miny) * 1.06
    pixels = rasterSize * supersample

    def toPixels(geometry):
        geometry = affinity.rotate(geometry, -angle, origin=centre)
        geometry = affinity.translate(geometry, -(minx + maxx) / 2, -(miny + maxy) / 2)
        return affinity.affine_transform(geometry, [pixels / span, 0, 0, -pixels / span, pixels / 2, pixels / 2])

    return toPixels


def drawRegion(draw, geometry, fill):
    for polygon in polygons(geometry):
        draw.polygon(list(polygon.exterior.coords), fill=fill)
        for ring in polygon.interiors:
            draw.polygon(list(ring.coords), fill=0)


def rasterize(plate, pockets, toPixels=None):
    toPixels = toPixels or plateFrame(plate)
    pixels = rasterSize * supersample
    channels = []
    material = plate.material.difference(geom2d.unary_union(pockets)) if pockets else plate.material
    featureUnion = geom2d.unary_union(plate.features) if plate.features else None
    for geometry in (material, featureUnion, plate.outline):
        image = Image.new("L", (pixels, pixels), 0)
        if geometry is not None and not geometry.is_empty:
            drawRegion(ImageDraw.Draw(image), toPixels(geometry), 255)
        image = image.resize((rasterSize, rasterSize), Image.Resampling.BOX)
        channels.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(channels)
