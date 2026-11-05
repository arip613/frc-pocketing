# What good FRC pocketing looks like

Design notes for the pocketing tool. They combine the Chief Delphi discussions
(reference only, see "Sources") with measurements from the plates we pulled out of
public team CAD on GitHub. Each rule is tagged with where it is enforced:
**[validator]** = hard rule, **[scorer]** = learned or scored preference,
**[generator]** = built into how layouts are constructed.

## 1. The mental model good teams use

Good pocketing is not cutting holes until the plate is light enough. You decide where
material has to stay, and everything else becomes a pocket:

1. Start from the plate outline and all of its features (bolt holes, bearing bores,
   hex bores, slots).
2. Keep a **wall** that follows the whole perimeter.
3. Keep a **boss** (a ring of material) around every feature.
4. Draw straight **ribs** between the points that carry load: bearing to bearing,
   bearing to bolt pattern, hole to hole, and hole to perimeter.
5. Whatever is left becomes a pocket, with every corner rounded to the cutter radius.

Chief Delphi describes this as "turn one plate into a bunch of little beams" and
"connect all points of interest directly", usually done with the Onshape *Lighten* /
*Part Lighten* FeatureScripts. Those tools let you set wall width and rib width
separately, which is the same parameterization this tool uses. Auto-pattern tools such
as CheeseIt (isogrid fill) needed a companion "PatchIt" script to fix **features left
floating inside pockets**. That is exactly the failure our hard rule 2 rejects.
**[generator, validator]**

## 2. Rules of thumb

| Topic | What good pocketing does | Typical numbers |
|---|---|---|
| Perimeter wall | Continuous band along every outer edge, including concave notches. Material near the edges is the most useful material in bending. | 0.125–0.25 in; see §4 for measured values |
| Ribs | Straight, constant width, connecting load points. Usually about 1–1.5× plate thickness. | 1/8–3/16 in on 1/8 in plate; "for 1/4 in aluminum we used 1/4 in wide lines" |
| Bosses | A full ring around every hole, at least as wide as the wall. Bearing bores get a wider ring than screw holes because they carry the shaft load and need flange engagement. | ring ≥ wall; bearings often 0.15–0.25 in |
| Corner radius | Every inside pocket corner gets at least the cutter radius. Use large radii: small radii are stress risers and force small, slow end mills. | 1/8 in end mill → ≥ 0.0625 in; 1/4 in end mill → ≥ 0.125 in |
| Pocket shape | Triangles beat rectangles. A rectangular cell can rack (shear) while a triangle cannot. Truss members typically sit 30–60° from the edge they run along. | mostly triangles; quads where two ribs run parallel |
| Load paths | Stress "wants to flow in smooth, uninterrupted, straight lines". Ribs should end on a boss or the wall, never in the middle of a pocket. Where a rib terminates changes peak stress a lot. | — |
| Pocket size | Skip pockets too small to be worth machining. Tiny pockets add machine time and stress risers for almost no weight. | inscribed circle well above the tool diameter |
| Consistency | One rib width and one wall width across the part. Mixed widths look amateur. | — |
| Symmetry | If the outline and hole pattern are symmetric, the pocketing should be too. | — |
| Where not to pocket | High-impact parts, corner brackets and gussets that carry the frame, and the area around bearings on gearbox plates, where stiffness keeps gears aligned. Leave these solid or lightly pocketed. | — |
| Large open areas | Fill big hole-free regions with an isogrid (triangle grid) or a blind half-pocket instead of one huge pocket. | — |

## 3. What bad pocketing looks like

These are the negative examples the aesthetic scorer should learn to reject. Most of
them are also synthetic "bad layout" generators for training. **[scorer]**

- A hole or bore sitting inside a pocket, or a boss connected to nothing (floating).
- Sliver ribs thinner than the rest, or widths that vary rib to rib.
- Curved, wandering, or kinked ribs that don't run between two load points.
- Ribs that dead-end in the middle of the plate.
- Sharp pocket corners, which can't be cut with a round tool.
- Many tiny pockets (salt-and-pepper look) or one giant pocket that guts the plate.
- Asymmetric pockets on a symmetric plate.
- Pockets crowding the perimeter, with a wall thinner than the ribs.
- Big rectangular windows with no diagonal bracing.

## 4. What the real CAD measures out to

From 932 GitHub CAD files we detected **460 unique flat plates**. **37 are pocketed**,
and **15 are "exemplars"** (≥ 3 pockets and ≥ 15% of material removed) from 8 repos.
By eye about 11–13 of those 15 are clean plate pocketing. Most are COTS gearbox and
VersaPlanetary plates (Spectrum's COTS library) plus team plates from 4400 and 7900.
The rest are spoked wheels or rings, or noisy extractions. Previews:
`data/processed/previews/realPocketed_*.png`. Numbers are for the 15 exemplars
(interquartile range in brackets); full detail is in `data/processed/datasetReport.json`.

| Measurement | Median [IQR] | Reading |
|---|---|---|
| Plate thickness | 0.25 in [0.19–0.25] | Pocketed plates are mostly 3/16–1/4 in. Thin plates rarely get pocketed. |
| Thinnest perimeter wall per plate | 0.187 in [0.186–0.25] | 3/16 in is the common wall; 1/4 in on heavier plates. |
| Rib width (pocket-to-pocket) | 0.25 in [0.25–0.375] | 1/4 in ribs dominate. |
| Rib width / plate thickness | 1.06 [1.0–2.0] | **Rib ≈ plate thickness** is the norm. |
| Hole edge → nearest pocket (boss) | 0.39 in [0.17–0.39] | Bosses are usually wider than the wall. |
| Pocket corner radius | 0.13 in [0.125–0.22] | **1/8 in radius (1/4 in end mill)** is the de-facto standard; 0.0625 in is the low end. |
| Material removed | 40% [29–60%] | Ambitious layouts reach 50–60%. Over 70% are spoked wheels and rings. |
| Pocketing symmetry (IoU with mirror) | 1.00 [0.89–1.00] | Most layouts are exactly symmetric. |
| Pocket shapes | 48 triangles, 37 quads, 23 curved, 6 five- or six-sided | Triangles slightly lead. "Curved" pockets fill the space between round bosses. |

So the tool's sensible **defaults** are: wall 0.1875 in, rib = plate thickness
(0.125–0.25 in), corner radius 0.125 in, aiming for 30–50% removed.

**Data caveat:** 15 exemplars is a small dataset. They also over-represent one style
(VEX/WCP-era COTS gearbox plates with "curved" pockets around big bosses). This shapes
the ML plan: the aesthetic scorer has to be regularized hard, augmented (rotations,
mirrors, scale), and lean on hard negatives built by breaking real plates, so it
doesn't just memorize these 15.

## 5. When pocketing isn't worth it (context for the user)

Experienced mentors on Chief Delphi repeatedly say the biggest weight win is usually
**thinner material**, not pocketing. A 1/8 → 1/16 in change halves the weight with
zero machining time, and "the plate is almost always plenty strong". Pocketing a
thicker plate *does* win in bending stiffness per pound (I scales with h³). That's why
bearing and gearbox plates are usually 3/16–1/4 in and pocketed: thickness keeps
bearings aligned and engaged. The tool doesn't make this call. It assumes you have
already chosen the plate, and it reports mass removed and stiffness so you can judge.

## 6. How this maps onto the tool

- **Validator (hard rules):** wall ≥ wall thickness everywhere along the perimeter.
  Ring ≥ wall around every feature, connected to the rib network. Every rib ≥ rib
  thickness. Every pocket corner ≥ tool radius, and pockets the tool can't cut are
  dropped. Pockets are subtracted from the original solid.
- **Generator:** builds layouts as *keep-material graphs* (wall + bosses + straight
  ribs between features and perimeter anchors), the same way teams design them.
  As in the user's example plates, ribs only join meaningful points: hole to hole, hole
  to the wall straight out from it, bore spokes every 30°, and evenly spaced wall
  points. Earlier versions also had free junctions on a hex grid. Those produced
  random-looking cells, so they were dropped.
  The learned part chooses **which ribs** to keep (and later their widths). Clean
  CAD therefore comes out by construction: straight ribs, exact widths, exact fillets.
- **Aesthetic scorer:** learns §2 and §3 from real plates versus perturbed and random
  ones, then from your thumbs up/down in the UI.
- **Structural score:** 2D plane-stress FEA with loads at bearing bores and supports
  at bolt holes. It reports stiffness and peak stress relative to the unpocketed plate,
  plus % mass removed.

## 7. Round 0: the truss baseline against real plates

From `data/eval/round0_baseline/summary.md`. On the real plates, the baseline uses each
real plate's own measured wall, rib, and effective tool radius.

| Group | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept |
|---|---|---|---|---|---|
| Baseline, 4 test plates | 100% | 46% | 27% | ×3.4 | 0.50 |
| Baseline, 15 real plates | 100% | 35% | 49% | ×2.4 | 0.75 |
| Real teams, same 15 plates | 80% | 45% | 44% | ×4.1 | 0.83 |

Readings:
- Real designers are **more mass-efficient** than a plain triangulation (0.83 vs 0.75).
- The baseline also **looks worse**: lots of small wedge-shaped pockets near the edges,
  where real plates use fewer, larger, rounder pockets with ribs running hole to hole.
  That gap, choosing *which* ribs to keep, is the ML generator's job.
- 3 real plates fail rule 4 even at their effective tool radius. Their pockets have
  truly sharp tips (likely waterjet or laser cut, or finished with a smaller tool).
  One is the WCP gearbox plate rated as the target style. The tool will produce
  rounded versions of tips like those.

## 8. The ML generator against the baseline and real teams

From `data/eval/round2_ml/summary.md`, with full FEA. "Looks" is the aesthetic scorer
(0–1). The scorer ranks good above bad layouts 95% of the time on held-out gold plates,
but it is still the model's own judge, so the preview images matter more.

| Group | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Looks |
|---|---|---|---|---|---|
| ML, 15 real plates | 45% | 38% | ×3.1 | 0.74 | 0.81 |
| Truss baseline, same plates | 29% | 57% | ×1.9 | 0.82 | 0.70 |
| Real teams, same plates | 45% | 44% | ×4.1 | 0.83 | 0.82 |
| ML, 4 held-out gold plates | 52% | 23% | ×4.0 | 0.46 | 0.91 |
| Real teams, same 4 plates | 40% | 35% | ×2.5 | 0.60 | 0.83 |

Readings:
- On looks and mass removed the ML matches real teams and beats the truss. Every
  design is valid, and the test plates export single-solid STEPs.
- Structurally it is a little weaker per unit of mass than real teams and the truss.
  On the held-out gold plates it pockets more aggressively than the real designs did.
- Looks were prioritized over stiffness at the user's request (utility 0.70 aesthetic).
  Gutted or near-disconnected designs are still blocked: decoding caps pocket size and
  elongation, and the UI ranks anything below the FEA floor last with a warning.
- On the gusset, every ML design falls below the floor (under 8% of the solid
  gusset's stiffness), so the tool picks the truss there. Across the 19 evaluation
  plates the tool picks the ML design on 12 and the truss on 7.
- What made the difference: soft per-pocket fillets (the WCP look), pocket size and
  elongation taken from the gold plates, a minimum 20° angle between ribs, and gutted
  and ladder layouts as scorer negatives.

## 9. Grading by eye (Claude as the judge)

At the user's request, looks are now judged by eye. Claude grades blind rating sheets
(`ratingSheets.py`) 1–5 against the example plates, and both models retrain on the
grades after each round. Three rounds graded 156, 156 and 97+ designs on about 25
plates each, every round drawn from the generator retrained on the grades before it.

What the eye picked up, and what the measurements confirm:
- **Polygon cells, not starbursts.** On the gold plates the tightest angle between ribs
  at a node is 104° at the median, and holes carry about 2 ribs. On Outside Plate R the
  earlier designs had 3.3 ribs per hole at about 50°, which reads as a triangulated
  starburst. A 40° minimum angle between ribs, as a decoding style, gives the example
  plates' pentagon and hexagon cells. Average grades: round 1, 20° 3.26, 40° with at
  most 4 ribs per hole 3.25, 50° with 3 ribs 2.87; round 2, 40° 3.39, 40° / 4 ribs
  3.31, 20° 3.16. The truss baseline averaged about 2. Each style wins on some plates,
  so the UI proposes in all three and the scorer picks.
- **No leftover strips.** A long, narrow pocket squeezed between the wall and a cutout
  is left solid. Long pockets between ribs, as in a fan, stay.
- **Never fill a big slot.** Filling long slots left the gusset almost unpocketed,
  which looks worse than the slots. Only small slivers are filled now. Slots are
  merged with a neighbour where that stays within the size limits.
- **Grades improve across rounds.** On the 14 plates graded in all three rounds, the
  ML designs averaged 3.17, then 3.34, then 3.62, and the best per plate 3.82, 3.79,
  3.93. Outside Plate R averaged 2.88, 3.44, then 3.62 (best 3.5, then 4.0), and the
  example 4 plate reached a 5. The truss baseline stayed near 1.9.
- **The scorer learned the grader's eye.** After all three rounds, on graded plates it
  never trained on, it agrees with the grades on 82% of pairs (136 pairs). The scorer
  before grading agreed on 66%. Held-out gold ranking accuracy also rose, from 0.85 to 0.89.
- **Full FEA evaluation** (`data/eval/round4_graded/summary.md`). On 23 real plates the
  ML removes 48% (real designs 46%, truss 32%) and keeps 33% of the solid plate's
  stiffness (real 39%, truss 51%), with stiffness per mass kept 0.61 (real 0.73, truss
  0.74). On the 6 held-out gold plates it is weaker per mass than the real designs
  (0.44 vs 0.61): looks were prioritized. Every ML design is valid. The UI puts ML first
  on 23 of 27 plates.
- **Big gaps get a consistent pattern.** On plates with few holes spread far apart (the
  Shooter Right Plate: 26 bolts over 26 in), ribs between holes alone left pockets of
  7–11 in², three times the size limit, with nothing to split them on. Free junctions
  now go on a triangular lattice wherever no hole or wall point is within 0.8 lattice
  steps. The decoder keeps the whole lattice, so the gap fills with equal triangles of
  the target pocket size, which is what a designer does with a big empty region. The
  lattice is centred on the plate, so symmetry is kept. The example and gold plates
  get no junctions at all, so their look is unchanged.

## 10. What the expert's picks say

An experienced FRC designer rated 282 designs 1–9 (rounds m3 and m3b) and made 258
A-vs-B picks on the Plate Pairs page (round n1 against m3 and m3b). Held-out plates
were never shown to him.
- **Claude's grades hurt.** The scorer was retrained with and without Claude's own
  sheet grades, cross-validated over three fifths of the graded plates. On held-out
  plates it matched his picks 94 of 144 times without Claude's grades and 81 with
  them. Leaving them out won in all three folds. The live scorer, trained before he
  rated anything, matched 93, so his data has not yet produced a scorer that beats it.
  The runs are in `models/marathon/cv/` (`agreement.log`); `models/marathon/e1` is
  the fold-0 run without Claude's grades.
- **Everyone prefers lighter plates with fewer ribs.** Within a plate, the design
  that removes more material wins 76–79% of graded pairs for the user, the expert and
  Claude alike, and the one with fewer ribs wins 71–76%. Material removed alone predicts
  his 1–9 ratings (78%) and his picks (64%) as well as the live scorer does (76%,
  62%).
- **Uncapping the mass term barely changes the output.** The ranking stops
  rewarding removal at 40%, and 74% of designs are past that. Rewarding it linearly
  (0.70 looks + 0.75 × removed) matched him a little better on every check (picks
  62% → 68% with the live scorer; held out, his ratings 72% → 73% and the user's grades
  74% → 77%). But it chose the same design on 6 of 8 look-test plates, and only a
  slightly lighter one on the other two (`data/eval/marathon/massCap/`). One run's
  candidates remove about the same amount, so only the generator can make plates
  lighter. Not adopted.
- **The training rounds make plates heavier.** With the cap, each generate-and-score
  round of the e1 generator removed less: 47.1%, 45.8%, 44.8%. On 18 open plates, all
  ranked by the live scorer, the u1 and e1 generators remove a little less than the live
  one (46.0% and 47.0% against 47.7%) and use more ribs on 12 of the 18. That is the
  opposite of what the raters reward, so neither replaced the live generator.
  `trainGenerator --linear-mass 0.75` scores the rounds without the cap instead. The
  m1 run (live scorer, no Claude grades) kept removal level in its one round (48.1%
  to 47.9% on validation) but stopped at the plateau, and on the 18 plates it removes
  46.9–47.0%, like e1. The live generator is still the lightest. Every generator
  learns mostly from imitating the gold plates, so the rounds move it very little.
  The numbers are in `data/eval/marathon/generators/traits.jsonl`.
- **Density.** Both the user and the expert grade the 1.0 pocket-size factor above
  0.6. In his picks the more open design won 66 of 89 times. The user grades 1.5 below
  1.0. The live default, 1.25, sits in that range. Rating sheets now offer 1.0 and
  1.25.

## 11. v2: built from the user's and the expert's own choices

- **The user's picks.** On a blind page (data/curation/sheets/p1) the user picked their
  favourite of the live, e1, u1 and m1 designs on 18 of 22 open plates: live 6, u1 5,
  e1 4, m1 4. No single set wins. The live scorer chose the user's pick on 5 of 18, about
  chance (4.7), and the lightest design on 10. On 8 fresh plates picked blind afterwards
  (p2), the lightest was the pick on 6 of 7 and the live ranking's on none (random 1.8).
  The cards showed each design's share removed, which may have helped. `CombinedPocketer`
  runs several sets and offers the lightest sound design first.
- **The taste model** (`taste.py`, `models/marathon/v2/taste.json`) is a linear model on
  the scorer's design features and its looks score, fitted only to people: 2,012
  within-plate preferences from the user's sheet grades and picks and the expert's ratings
  and pair picks, the user and the expert weighted equally, no Claude grades. Held out by
  plate it agrees with the expert's pair picks 70% (live ranking 62%), his ratings 79%
  (76%) and the user's model-set picks 77% (59%). On the user's sheet grades it gets 88%,
  where the live scorer's 93% is in-sample. It rewards looks, material removed, bigger and
  more even pockets, triangles and symmetry.
- **v2** proposes with all four generators, ranks with the taste model through the usual
  structural floor, and polishes, including the hole rule below. A v2 generator trained
  on the user's and the expert's top designs, with rounds scored by the taste model, is
  in `models/marathon/v2`.
- **No hole hangs off one rib** (the user's rule). A hole off the outer wall held by a
  single rib gets a second in the last polish step. Holes whose bosses overlap count as
  one, and a hole closer to the wall than a pocket could fit counts as on the wall. On the
  114 designs of the pick pages no hole was left hanging by one arm.

## 12. Robots the user exported, their style, and what they pick

- **Six Onshape robots, exported by hand** (STEP per robot, `data/processed_manual`). The
  extractor first found 53 pocketed plates. Three parallel diagnoses found why, and the
  fixes brought it to 195. Every solid of a file is now scanned, not just the first 400.
  Bent sheet metal is read from its flat base. Plates with blind or floored pockets pass.
  Rounded-triangle and kidney pockets are no longer taken for slots. Belts and rings are
  dropped. The user kept 95 of the 191 shown (`data/curation/plateKeeper.json`). They are
  gold weighted as the user's taste. The Sim robot's 13 are held out as the style test.
- **Their style, measured** (kept against not kept, same robots). Pockets are dense:
  0.67 per in² of material against 0.35. Pocket area is about 0.165 · S^1.05 in², with
  S = √(material area). Ribs are thin and one width, about 0.156 in, with the wall and boss
  rings the same. Open junctions are mostly 3-way, the smallest angle about 92°. Every hole
  is engaged, with about 3 ribs per interior hole. Edge bolts get ribs more often than a
  plain band. About 50% of the material is removed.
- **Style test** (`styleTest.py`): rebuild the held-out kept plates from outline and holes.
  The live tool matches pocket count and weight but not the layout (material IoU 0.55,
  0.46 for an unpocketed plate).
- **Truss engine** (`truss.py`, generator "truss", off by default). Its rib junctions are
  free: relaxed even triangles, a version with sharp junction angles removed, and aligned
  rails with diagonal or square rungs. It can draw layouts the fixed-point generator
  cannot. In a blind round against the live tool's design (p5, 18 plates) the user picked
  live 13 times and the truss engine 5, all of those on bigger plates. They never picked
  the aligned rows.
- **Choosers.** A looks scorer retrained with the kept plates as gold (s4) liked truss
  designs and matched the user's p5 picks 6 of 18 times. The lightest design matched 13.
  Across all three blind rounds (43 plates), the rule now live matched the user 31 times.
  It shows the lightest sound design among the taste model's top three. The taste model's
  own first choice matched 24 and the lightest overall 29.

## 13. The expert's taste only (v5 chooser)

- **The user's call (2026-11-01):** the expert's taste is what matters; the user's does
  not count. The chooser is judged by his mean rating of design A, the design shown first.
- **His ratings of the live tool's own output** (round L1f, page part 6, 63 of 378 so far):
  mean 4.1 of 9. Design A 4.4 (28 rated), B 4.0, C 3.2. Across these designs the v2 taste
  model's score does not track his rating (Spearman -0.09). His 7-9s have every hole tied
  into ribs with bosses, ribs running hole to hole, fewer and rounder pockets. His 1-2s
  have "cracked glass" layouts that ignore the holes, holes left in solid areas, long
  slivers, over 60% removed, or pockets added to plates that already have dense hole rows.
- **The rules, checked on his own preferences within a plate:** the lighter design wins
  78% of 348 pairs (75% of 259 when both remove 35%+), and on symmetric plates the more
  symmetric one wins 83% of 77.
- **Design A, simulated** on the 56 plates where he rated 2+ designs, taste scores held
  out by plate (`taste.py` reports the live-round part of this; all versions choose among
  the designs he rated): the average design gets 3.10 and the best possible 5.12. The live
  chooser (v2 taste, lightest of the top 3, symmetry rule) gets 3.82, in-sample, since v2
  saw these ratings. The expert-only taste model (v5) gets 3.77; with the symmetry rule
  3.88; with lightest-of-top-3 as well 3.89. So the symmetry rule helps (+0.11, +0.24 on
  symmetric plates) and lightest-of-top-3 adds nothing once the model is his (+0.01).
  The model already puts its largest weight on material removed. Comparing his ratings
  across plates, of the live round or all rounds, lowered the score and the held-out
  agreement, so they are compared within a plate only.
- **v5** (`models/marathon/v5/taste.json`): trained on his ratings and pair picks only,
  within a plate. With the symmetry rule and no lightest-of-top-3, it ties the live
  chooser on his design-A rating (+0.05, 95% interval -0.30 to +0.39 over 56 plates). The
  gap to the best possible (1.2 points) is what the features miss. Round L2 runs it on the
  190 plates of L1 so his ratings can settle it.
- **Round L2** (v5 + symmetry rule, no lightest-of-top-3, same plates, generators and
  seed as L1): 189 plates have a design A. It is the live chooser's design A on 91 and
  differs on 98; there it is live's B or C on 53 and a design live didn't show on 45. Where
  they differ, v5's A removes more material on only 33 of 98. His ratings of both A designs
  on those 98 plates settle it: 22 were already rated; the other 174 open his part 7+6 page
  (`plate-rater-parts76.html`, which carries over his part 6 progress), ahead of the rest
  of part 6. Round L2f holds v5's new A designs (45) and L1x the live A designs the part 6
  filter left off (3); both are shown even below the filter, since each is what its
  chooser would show first.

## 14. Round L2 result: v5 vs live on his design-A ratings (2026-11-05)

He rated all 426 designs on the part 7+6 page in one sitting, from a different browser
(rater expert:befrf1yi, so part 6 was rated again from scratch), and edited 27 designs by
hand.

- **Design A, live vs v5**, on the 98 plates where the two choosers show different A
  designs (his latest session, every one rated): live 5.48, v5 5.82, **+0.34 (95% interval
  -0.14 to +0.83)**; v5 better on 43, worse on 34, tied on 21 (sign test p = 0.36). Over all
  184 rated plates: live 5.22, v5 5.40. Where v5 picked live's own B or C design: +0.51 (53
  plates); where it picked a design live never showed: +0.13 (45).
- **With the team's original open:** 205 of the 426 ratings were given with that panel
  open (he left it open from rating 138 on). On the 80 deciding plates where neither A was
  rated with it open, v5 and live tie (+0.11, 30 better, 31 worse). So v5 is at least as
  good as live, not clearly better.
- **His repeatability:** the 63 designs he rated twice (part 6, then again here) agree
  only loosely: Pearson 0.38, mean gap 1.9 points, the same rating on 15. A real
  difference of a third of a point needs about 100 plates to show.
- **Weight:** between the two A designs, the lighter one was rated higher on 36 of 96
  plates, lower on 39 (mean -0.26); where v5 chose the heavier design it did better (+0.44
  on 63 plates). Over all his within-plate pairs the lighter design still wins 74% (441),
  so weight matters between a good and a poor design, not between two good ones. Dropping
  lightest-of-top-3 holds.
- **v6** (`models/marathon/v6/taste.json`): retrained on all his data (714 preferences;
  the live-design pairs grew from 63 ratings to 489). Cross-validation picked a much
  stronger penalty (10 vs 1), as his ratings are noisy, so the weights are smaller; the
  strongest terms are removedFraction +0.76, meanConvexity +0.46, logMeanPocketArea +0.40,
  symmetryShortfall -0.39, elongatedFraction +0.37, pocketSizeVsGold -0.37, holesEngaged
  +0.36. Simulated design A (held out by plate, choosing among his rated designs): v6
  5.12, v5 5.09, v2 4.99, the live tool's own order 5.21, the best possible 6.31. With
  the symmetry rule, v6 is +0.15 over live (-0.17 to +0.46) on 168 plates. Not tested in a
  live round.
- **His edits** (27 designs, mostly ones he rated 5-8): ribs added on 24 (44 ribs),
  material erased on 9, pocket count up on 22, removed share 46% -> 45%. The pockets he
  split with a rib are long (perimeter²/4πA 1.79 vs 1.40 for the ones he left alone)
  and large for their plate (1.42x the plate's median pocket area vs 1.00x; 24% of them
  over 2x vs 3% of the rest), not wide (inscribed width 0.71 vs 0.80 in). So designs
  he likes often lose points to one or two long, oversized pockets.
- **Where the room is:** every chooser lands within noise of the others (4.5-5.4) and the
  best design on each plate averages 6.2-6.3, part of which is noise in his ratings. The
  rest has to come from the designs: splitting long oversized pockets the way he does.

## 15. Live switched to his taste (2026-11-05)

The user: use his ratings any way we can (they took him two hours). So:

- **Chooser:** live now uses **v6** (`models/marathon/v6/taste.json`), trained only on his
  ratings, pair picks and hand edits; **lightest-of-top-3 is off** (`lightestOfTop = 1`);
  the symmetric-plate rule stays. The round L2 test above backs the expert-only chooser
  (v5 +0.34 over v2 on design A); v6 adds his latest ratings and edits and ties v5 held out.
- **His edits as data:** round E1 (`expertRater edits`) holds his 27 edited designs rebuilt
  from his strokes; the taste model counts each as beating the design he edited. Held out,
  v6 sides with his edit 78% of the time (removed share alone: 15%, since he mostly adds
  ribs).
- **Splitting long pockets:** polish now splits long pockets from elongation 2.2 (was 3.0;
  `polish.splitTriggerElongation`), fitted to the pockets he split by hand. On 12 of his
  edited plates (v5 chooser, against round L2) design A changed on 6, with fewer long
  pockets (-0.58 per plate), 0.6% less material removed and every design still sound.
  Thin truss wedges smaller than a typical pocket are left alone, as before.
- **Revert:** `pocketer.defaultScorerPath` back to `models/marathon/v2/taste.json`,
  `pocketer.lightestOfTop = 3`, `polish.splitTriggerElongation = 3.0`.

## Sources

See `data/sources.jsonl` for the per-item log (URL, timestamp, license, sha256,
trainingAllowed).

- **GitHub** (public team CAD repos via the REST API, rate-limited): 932 STEP/DXF files
  from 45 team and COTS-library repos so far (the crawl is still extending). 789 of
  them come from repos that declare no license, which means all rights reserved. They
  are used locally to train this tool, kept out of git, and not redistributed.
- **Chief Delphi** (Discourse JSON API, 3 s between requests): 25 pocketing threads
  and 120 individual matching posts (464 posts total), plus 40 reference images. Its robots.txt sets
  `Content-Signal: ai-train=no, use=reference`. Everything from Chief Delphi is
  therefore logged `trainingAllowed=false` and used only for these notes and as visual
  reference, **never as model training data**.
  GitHub's Acceptable Use Policy says API collection is not "scraping". It lists
  research and archival as permitted uses of public information, and copyright still
  applies. A local, unpublished learning project is closest to "research".
- **Onshape (hand exports only):** Onshape's API terms state the API "may not be used
  by any robot, spider, scraper or other automated means for data mining, data
  gathering or extraction method from the Onshape Public Documents" (see
  [API Limits](https://onshape-public.github.io/docs/auth/limits)). Its robots.txt
  also disallows crawling. So nothing is pulled from Onshape automatically. You export
  plates yourself in the browser (`docs/onshapeExportList.md`), and
  `python -I -m frcpocket.ingestManual` copies them in as the curated tier
  (`origin=manual`). Chief Delphi CAD-release threads were used only to list links to
  those public documents.
- **User ratings:** `data/curation/ratings.jsonl` (three WCP / Team 221 gearbox
  plates rated as the target style so far).
- **Not used:** GrabCAD (robots.txt is blocked by the CDN for automated clients, and
  downloads require a login).
