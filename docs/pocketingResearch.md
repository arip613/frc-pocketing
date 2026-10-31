# What experienced FRC teams say about pocketing

Collected 2026-10-30 from Chief Delphi threads, GM0, FRCDesign.org, and general CNC
design-for-manufacturing guides. This is reading material only. Nothing from these
sources is training data.

The community mostly pockets "by feel", and few hard numbers exist. A number marked
**[default]** is an engineering default, not something a source stated.

## Rules that separate good pocketing from mediocre

1. **Load paths.** Every rib runs straight between two load points (bearing, bolt,
   standoff). Delete ribs that carry no load. Indirect ribs bend instead of carrying
   tension or compression, and bending is much weaker. (CD 500469, 495200, 483120)
2. **Perimeter ring.** Keep a continuous ring of material at the edge, so no bolt or
   bearing hangs off a single rib. In review, "cantilevered bolts" is the criticism that
   comes up most. (CD 145006, 386027)
3. **Bosses.** Put a ring around every hole (hole radius plus an offset). Use at most
   3–4 distinct offsets, and join the circles with tangent lines. Bolt edge distance is
   at least 1.5D from hole center to edge; 2D is safer in aluminum. (GM0; Aluminum Design
   Manual via engineeringexpress)
4. **Fillets.** Make inside fillets at least 1.3× the tool radius, and as large as the
   geometry allows. When the radius equals the tool radius, a corner takes nearly 100%
   tool engagement. At 3× the tool radius it drops to about 52%. (CNC DFM guides; CD 133186)
5. **Ribs.** Use one consistent rib width per part, scaled with thickness. That means
   about 0.25" on 1/4" plate, 0.3" on 3/16" plate, and 1.5–2× t on 1/8" plate.
   (CD 495200)
6. **Pocket size.** Fewer, larger pockets. Drop micropockets the tool can't clear or that
   save little weight. One review called out "2–4 too many pockets". (CD 495200, 145006)
7. **Angles.** No acute slivers. Keep rib-to-rib angles at about 30° or more
   **[default]**, use triangle-ish cells, and have 3–5 ribs meet at each node.
   (CD 459854, 130624)
8. **Outline first.** Trim the outer outline (material outside the load paths) before
   pocketing. That is the cheapest weight to remove. (CD 500469)
9. **Where not to pocket.** Leave impact plates, bottom plates, swerve bottom plates,
   bearing seats and tube side walls solid or only lightly pocketed. One fully pocketed
   1/4" swerve bottom plate deflected 3× more than solid. (CD 392353, 386027, 441210)
10. **Symmetry and timing.** Mirror the pattern where the hole layout is symmetric, and
    pocket only once the design is frozen. (GM0; CD 145006)

## Other numbers

- **How much to remove.** Aim for 30–50% on 1/4" gearbox plates **[default]**, and less on
  impact or bottom plates. Pocketing stock 0.09" or thinner is "a waste of time"
  (CD 386027). Team 254 says it would miss weight 9 years out of 10 without pocketing,
  at 3–4 minutes of CNC time per plate (CD 146081).
- **Tools.** Use a 1/4" end mill where possible. A single-flute upcut leaves a clean bottom
  edge. 254 runs a 1/4" 3-flute at full depth with 40% stepover (CD 133186).
- **Blind pockets.** These are much stiffer but slower to cut. Team 100 leaves floors of
  about 0.060" (CD 130624).
- **Disagreement.** Sources disagree on triangles versus slots for tubes: triangles add
  corners, so large-radius slots may be better (CD 500469). They also disagree on thick
  pocketed stock versus thin solid stock (CD 523754).

## What the tool does with this

- **Already enforced or measured.**
  - Hard rules: wall, boss ring, rib, and tool radius.
  - What the scorer measures:
    - slivers
    - rib-width consistency (gap variation)
    - symmetry
    - sharp corners
    - holes engaged
    - solid blobs
    - pocket size versus gold
  - The polish pass links facing bosses (hole-to-hole ribs) and drops ribs that hug a
    wall.
- **Added from this research.**
  - Inside fillets are at least 1.3× the tool radius (`layout.minFilletFactor`).
  - The loosest generation style now keeps ribs at least 30° apart, up from 20° (`pocketer.ribStyles`).
  - The scorer has a new feature, `hangingHoleFraction`: the share of holes whose ring
    is pocketed nearly all the way round (rule 2).
- **Not encoded.**
  - Rule 9 (where not to pocket) depends on how the plate is used, which the tool can't
    see. Set it with the rules instead, for example a thicker wall or no pocketing on
    bottom plates.
  - Rule 6's "drop small pockets" conflicts with the user's preference to pocket strips
    along walls whenever a cutter fits. The user's preference wins.
