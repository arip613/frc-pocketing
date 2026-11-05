# Overnight summary

Everything is committed and pushed to `main-kguole`. Nothing was deployed, and the live models were not switched.

## What changed

1. **Polish pass (now on by default).** After generation it:
   - fills unpocketed blobs
   - adds hole-to-hole ribs
   - symmetrizes, which gives the X's you asked for
   - splits oversized pockets

   Every step is checked against the hard rules and FEA, and a step is kept only if the STEP export rebuilds it exactly. This targets your feedback on items 2, 3, 9 and 11. The code is in `src/frcpocket/polish.py`, and all 43 tests pass.
2. **Team 2056 drawings.** 32 plates added to the training data (9 gold, 23 keep), following your curation.
3. **SDS Parasolid.** All 15 SDS `.x_t` files now parse. The plates in them are unpocketed, so they aren't useful as gold.
4. **A second model set trained with the 2056 gold** (`models/archive_2056/`). Compared with the live models:

   | Measure | Live models | 2056 models |
   |---|---|---|
   | Scorer gold-holdout | 0.831 | 0.787 |
   | Generator utility | 0.786 | 0.803 |

   The two numbers point in opposite directions, so it's your eye that decides.

## Renders: `data/eval/showcase/polish/`

Each image shows four panels: real | current | current + polish | 2056 models + polish.

**What I see:**
- **Polish clearly helps.**
  - On the 3-CIM, Sidewinder, Example 6 and Example 4, the blobs are gone and ribs now run between the holes.
  - The gusset went from completely unpocketed to a symmetric truss.
- **The 2056 models are denser and more triangulated, with more X's.**
  - They are better on the shooter right plate: the current models leave it completely unpocketed, but the 2056 models pocket it.
  - They are busier or more irregular on Example 3 and on part of the Sidewinder, where there is one large mid-right pocket.

## Decisions for you

1. **Live models:** keep the current ones, or switch to `archive_2056`? I lean toward the 2056 models, mainly because of the shooter plate, but your eye beat mine last time. Switching means copying `models/archive_2056/*` into `models/`.
2. **Polish:** it's on by default. I'd keep it.
3. **Weekly source search:** to set it up, I need your timezone.

## Not done yet
- The candidate-rib ceiling (oracle) check.
- The fixed look-test panel.
- Further grading rounds.
