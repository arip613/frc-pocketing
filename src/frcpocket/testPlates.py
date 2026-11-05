"""Synthetic test plates (built in inches, written as millimetre STEP like real CAD).

    python -I -m frcpocket.testPlates --out data/testPlates
"""

import argparse
import math
from pathlib import Path

import cadquery as cq

boltHole = 0.201  # #10 clearance
rivetHole = 0.196  # 3/16 rivet
bearingBore = 1.125  # 1/2 in hex flanged bearing
hexBore = 0.5 / math.cos(math.radians(30))  # 1/2 in hex, across corners


def holes(workplane, points, diameter):
    return workplane.faces(">Z").workplane(origin=(0, 0, 0)).pushPoints(points).hole(diameter)


def rectangleBoltPattern():
    """8 x 4 x 1/4 in plate: 1 in tube-mount pattern along both long edges, motor square."""
    plate = cq.Workplane("XY").rect(8.0, 4.0).extrude(0.25).edges("|Z").fillet(0.25)
    edgeHoles = [(x, sy * 1.5) for x in (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5) for sy in (-1, 1)]
    plate = holes(plate, edgeHoles, boltHole)
    plate = holes(plate, [(sx * 0.6, sy * 0.6) for sx in (-1, 1) for sy in (-1, 1)], boltHole)
    return plate


def gusset():
    """Right-angle frame gusset, 6 in legs, 1/8 in, rivet rows along both legs."""
    points = [(0, 0), (6, 0), (6, 1.25), (1.25, 6), (0, 6)]
    plate = cq.Workplane("XY").polyline(points).close().extrude(0.125).edges("|Z").fillet(0.125)
    rivets = [(x, 0.5) for x in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)] + [(0.5, y) for y in (1.5, 2.5, 3.5, 4.5, 5.5)]
    return holes(plate, rivets, rivetHole)


def bearingPlate():
    """Gearbox plate: 3 bearing bores in a line, corner standoffs, one counterbored bolt."""
    plate = cq.Workplane("XY").rect(7.0, 4.0).extrude(0.25).edges("|Z").fillet(0.5)
    bearings = [(-2.25, 0.0), (0.0, 0.4), (2.25, 0.0)]
    plate = holes(plate, bearings, bearingBore)
    standoffs = [(sx * 3.0, sy * 1.5) for sx in (-1, 1) for sy in (-1, 1)]
    plate = holes(plate, standoffs, boltHole)
    # Bolt circles around each bearing (bearing retainer screws).
    for cx, cy in bearings:
        circle = [(cx + 0.9 * math.cos(math.radians(a)), cy + 0.9 * math.sin(math.radians(a))) for a in (90, 210, 330)]
        plate = holes(plate, circle, 0.15)
    plate = plate.faces(">Z").workplane(origin=(0, 0, 0)).pushPoints([(0.0, -1.5)]).cboreHole(boltHole, 0.375, 0.1)
    return plate


def irregularArm():
    """Arm side plate: tapered outline with a concave notch and a round pivot end."""
    sketch = (
        cq.Sketch()
        .polygon([(0, -1.5), (9, -0.9), (10.5, 0.0), (9, 0.9), (5.0, 1.1), (4.0, 0.5), (3.0, 1.5), (0, 1.5)])
        .vertices().fillet(0.2)
    )
    plate = cq.Workplane("XY").placeSketch(sketch).extrude(0.1875)
    pivot = cq.Workplane("XY").circle(1.75).extrude(0.1875)
    plate = plate.union(pivot)
    plate = holes(plate, [(0, 0)], bearingBore)
    plate = plate.faces(">Z").workplane(origin=(0, 0, 0)).center(9.6, 0).polygon(6, hexBore).cutThruAll()
    bolts = [(0.0, 1.35), (1.17, -0.68), (-1.17, -0.68), (6.0, 0.55), (6.0, -0.55), (8.5, 0.0)]
    return holes(plate, bolts, boltHole)


testPlateBuilders = {
    "rectangleBoltPattern": rectangleBoltPattern,
    "gusset": gusset,
    "bearingPlate": bearingPlate,
    "irregularArm": irregularArm,
}


def writeTestPlates(outDir, overwrite=False):
    """Write each test plate's STEP unless it already exists (re-exporting the same plate
    changes the file's timestamp and entity numbering, which shows up as a git change)."""
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, builder in testPlateBuilders.items():
        path = outDir / f"{name}.step"
        if overwrite or not path.exists():
            cq.exporters.export(builder().val().scale(25.4), str(path))
        paths[name] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/testPlates")
    args = parser.parse_args()
    for name, path in writeTestPlates(args.out, overwrite=True).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
