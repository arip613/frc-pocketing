"""Project a real pocketing design onto the rib-graph representation.

Candidate ribs are picked greedily to explain the real design's rib material (what's
solid beyond the wall and bosses) without running over its pockets. Ribs the real design clearly has become
positive labels for the generator, and rebuilding them gives a "real-style" layout
drawn by our own builder. That way the aesthetic scorer can't tell real from generated
designs by drawing style alone. Rib width classes let wide real ribs be represented.
"""

from dataclasses import dataclass

from shapely.ops import unary_union

from frcpocket.layout import arcSegments, candidateGraph, defaultGraph, makeLayout, polygons

ribWidthClasses = (1.0, 1.6, 2.4)  # multiples of the minimum rib width
coverageThreshold = 0.85
minExplained = 0.5  # a picked rib's free band must be at least this much real rib material
maxSpill = 0.2  # and may run over at most this much of the real pockets
alignTolerance = 0.5  # real ribs this far (in rib widths) off a candidate's centre line still count as on it (46 gold plates: median match 0.687 -> 0.697, 19 better, 4 worse)


@dataclass
class Projection:
    layout: object
    coverage: list  # per candidate edge: fraction of its free band that is real material
    widthClass: list  # per candidate edge: index into ribWidthClasses (widest that fits)
    pocketIoU: float  # IoU of kept material inside the wall: do the ribs match the real ribs


def alwaysKept(plate, params):
    inner = plate.outline.buffer(-params.wallIn, quad_segs=arcSegments)
    bosses = [feature.buffer(params.ringFor(feature), quad_segs=arcSegments) for feature in plate.features]
    return unary_union([plate.outline.difference(inner), *bosses])


def bandCoverage(band, kept, realMaterial):
    free = band.difference(kept)
    if free.area < 1e-6:
        return None  # band lies entirely in wall/boss material: says nothing either way
    return free.intersection(realMaterial).area / free.area


def projectRealDesign(plate, params, realPockets, graph=None, maxRibs=400):
    """Greedy cover: repeatedly add the candidate rib that best explains the real design's
    rib material (material not already explained by the wall and bosses) without running
    over the real pockets. Then rebuild it and measure how well the ribs match.
    """
    graph = graph or defaultGraph(plate, params)
    kept = alwaysKept(plate, params)
    realUnion = unary_union(realPockets) if realPockets else None
    realMaterial = plate.material.difference(realUnion) if realUnion is not None else plate.material
    ribMaterial = realMaterial.difference(kept)
    uncovered = ribMaterial
    bands, reach, core = {}, {}, {}
    tolerance = alignTolerance * params.ribIn
    for edge in graph.edges:
        segment = graph.segment(edge)
        for classIndex, factor in enumerate(ribWidthClasses):
            half = params.ribIn * factor / 2
            bands[edge, classIndex] = segment.buffer(half, cap_style="flat")
            # A real rib drawn a little off the candidate's line is still that rib: real
            # material within the tolerance explains the band, and only the band's core
            # counts as running over a real pocket.
            reach[edge, classIndex] = segment.buffer(half + tolerance, cap_style="flat") if tolerance else bands[edge, classIndex]
            core[edge, classIndex] = segment.buffer(max(half - tolerance, 0.5 * half), cap_style="flat") if tolerance else bands[edge, classIndex]
    chosen, widthClassOf = [], {}
    for _ in range(maxRibs):
        best, bestGain = None, 0.0
        for (edge, classIndex), band in bands.items():
            if edge in widthClassOf:
                continue
            free = band.difference(kept)
            if free.area < 1e-6:
                continue
            explains = min(reach[edge, classIndex].difference(kept).intersection(uncovered).area, free.area)
            spills = core[edge, classIndex].difference(kept).intersection(realUnion).area if realUnion is not None else 0.0
            if explains < minExplained * free.area or spills > maxSpill * free.area:
                continue
            gain = explains - 3 * spills
            if gain > bestGain:
                best, bestGain = (edge, classIndex), gain
        if best is None:
            break
        edge, classIndex = best
        segment = graph.segment(edge)
        if any(segment.crosses(graph.segment(other)) and not (set(edge) & set(other)) for other in chosen):
            bands = {key: band for key, band in bands.items() if key[0] != edge}
            continue
        chosen.append(edge)
        widthClassOf[edge] = classIndex
        uncovered = uncovered.difference(reach[best].buffer(params.ribIn * 0.25))
    widths = {edge: params.ribIn * ribWidthClasses[widthClassOf[edge]] for edge in chosen}
    layout = makeLayout(plate, params, graph, chosen, name="projectedReal", ribWidths=widths)
    builtMaterial = plate.material.difference(unary_union(layout.pockets)) if layout.pockets else plate.material
    region = plate.outline.buffer(-params.wallIn)
    a, b = builtMaterial.intersection(region), realMaterial.intersection(region)
    iou = a.intersection(b).area / max(a.union(b).area, 1e-9)
    chosenSet = set(chosen)
    coverage = [1.0 if edge in chosenSet else 0.0 for edge in graph.edges]
    widthClass = [widthClassOf.get(edge, 0) for edge in graph.edges]
    return Projection(layout=layout, coverage=coverage, widthClass=widthClass, pocketIoU=iou)


def dropCrossings(graph, ribs, score):
    """Where two chosen ribs cross (only possible with non-Delaunay extras), keep the better one."""
    chosen = sorted(ribs, key=lambda edge: -(score.get(edge) or 0))
    kept = []
    for edge in chosen:
        segment = graph.segment(edge)
        crosses = any(
            segment.crosses(graph.segment(other)) and not (set(edge) & set(other))
            for other in kept
        )
        if not crosses:
            kept.append(edge)
    return kept
