# frcpocket

ML-driven pocketing for flat FRC plates: STEP in, pocketed STEP out.

Status: reference data, validator, truss baseline, exact STEP export, FEA, aesthetic
scorer, ML rib generator, and the local UI all work. See "Results" for how the ML
compares with the baseline.

## Setup

Requires macOS/Linux with [uv](https://docs.astral.sh/uv/) (or any Python 3.11–3.12).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Step 1: reference data

Scripts are run with `python -I` (isolated mode), so nothing in the downloaded data
folders can shadow a Python module.

```bash
# Team CAD from public GitHub repos (slow without a token: 60 API calls/hour)
.venv/bin/python -I -m frcpocket.scrape.github --out data/raw/github --max-repos 70 --max-files 500
# Larger mechanism/robot assemblies, picked from the cached repo listings (no API quota)
.venv/bin/python -I -m frcpocket.scrape.github --out data/raw/github --big-assemblies 30
# Chief Delphi pocketing threads (reference only; robots.txt says ai-train=no)
.venv/bin/python -I -m frcpocket.scrape.chiefdelphi --out data/raw/chiefdelphi
# Plates you exported by hand (Onshape/vendor STEP or DXF saved to ~/Downloads/frcPlates)
.venv/bin/python -I -m frcpocket.ingestManual --inbox ~/Downloads/frcPlates
# Find flat plates in the CAD and separate holes from pockets
.venv/bin/python -I -m frcpocket.extract --raw data/raw/github data/raw/manual --out data/processed --workers 2 --timeout 1200
# Statistics and preview sheets
.venv/bin/python -I -m frcpocket.datasetReport
```

Set `GITHUB_TOKEN` to raise the GitHub API limit to 5000 calls/hour.

Onshape is never scraped: its API terms forbid automated extraction from public
documents. Export plates by hand instead, following `docs/onshapeExportList.md`.

Outputs:
- `data/sources.jsonl`: one line per fetched item (URL, license, sha256, trainingAllowed)
- `data/processed/plates.jsonl`: extracted plates (outline, holes, pockets, stats, in inches)
- `data/processed/previews/realPocketed_*.png`: contact sheets of real pocketed plates
- `docs/designNotes.md`: what good FRC pocketing looks like

## Step 2 core: pocket a plate with the baseline

```bash
# Synthetic test plates (rectangle bolt pattern, gusset, bearing plate, irregular arm)
.venv/bin/python -I -m frcpocket.testPlates --out data/testPlates
# Round-0 evaluation: baseline on test + real plates, validator, FEA, previews, STEP export
.venv/bin/python -I -m frcpocket.evaluate --generator baseline --round round0_baseline
# Tests
.venv/bin/python -m pytest -q
```

Results land in `data/eval/<round>/`: `summary.md`, `report.json`, `previews/` (real
design next to the generated one), and `steps/` (pocketed STEP files).

How a design is built: pocketing is a set of straight **ribs** between nodes (feature
centres and anchors on the wall centre line). Material kept is the perimeter wall, a
boss around every feature, and the ribs. The rest is pocket, *opened* by the tool radius
so every corner gets at least that radius. The validator then checks the five hard
rules independently, and the export rebuilds the pockets with exact OpenCascade
geometry and cuts them from the untouched original solid.

## Step 2: train the ML

```bash
# Both models in one go (scorer, then generator against it), with timings per stage.
# Meant to finish in under 20 minutes on a laptop.
.venv/bin/python -I -m frcpocket.retrain
# Or one at a time. Aesthetic scorer: real gold plates vs bad layouts of the same plates
# (+ your UI ratings and graded rating sheets). Training sets are built one plate per
# process (--parallel, default cores - 2, max 6), with per-plate seeds so results don't
# depend on scheduling.
.venv/bin/python -I -m frcpocket.scorer --out models/aestheticScorer.pt
# Rib generator: imitation, then generate -> validate -> score -> retrain until plateau
# (default: one round after imitation; --rounds N for more, stopping at the first without a gain)
.venv/bin/python -I -m frcpocket.trainGenerator
# Evaluate ML vs baseline vs real designs (previews + STEP)
.venv/bin/python -I -m frcpocket.evaluate --generator ml --round round2_ml
```

### Grading by eye (rating sheets)

Looks are judged by eye, not just by the scorer. Each round draws a sheet per plate:
up to 9 valid candidates (every rib style and density, plus the truss baseline),
shuffled, numbered, and drawn in the colours of the example screenshots. Gold plates
also show the real design as panel R. A grader (Claude, at your request) scores each
design 1–5 against the example plates. The grades then train both models:
- the scorer learns that a higher-graded design beats a lower-graded one on the same
  plate (one plate in five is held out to check it learned the grader's eye);
- the generator imitates every design graded 4 or higher.

```bash
.venv/bin/python -I -m frcpocket.ratingSheets make --round r2
# grade data/curation/sheets/r2/*.png into data/curation/sheets/r2/grades.json
.venv/bin/python -I -m frcpocket.ratingSheets ingest --round r2 --rater claude
.venv/bin/python -I -m frcpocket.scorer --out models/aestheticScorer.pt
.venv/bin/python -I -m frcpocket.trainGenerator --rounds 6 --imitation-epochs 100
```

Grades accumulate in `data/curation/sheetRatings.jsonl`.

How it works:
- **Representation.** A design is a set of straight ribs chosen from a candidate graph
  built the way your example plates are drawn (`layout.holeGraph`). Nodes are hole
  centres and wall points only: the wall point straight out from each hole, spoke points
  every 30° around bores of 0.5 in or more, outline corners, and evenly spaced points
  between those. Ribs run hole to hole, hole to wall, or bore to rim. Where holes are
  far apart (big gaps), free junctions on a regular triangular lattice fill the gap,
  and the decoder keeps the whole lattice. That gives one consistent pattern of equal
  triangles at the target pocket size, and the model's ribs tie it to the holes and the
  wall. Hole-dense plates like your examples get no junctions. Candidate ribs are
  Delaunay edges, hole-to-hole edges, and each hole's own spokes. On symmetric plates
  the graph is closed under every symmetry the plate has, so every rib has its mirror
  image. Clean CAD comes out by construction.
- **Soft fillets.** Every pocket's corner radius is 33% of that pocket's own inscribed
  radius (between the tool radius and 0.25 in), measured from the gold pockets, for the rounded look of the
  WCP / Team 221 plates. The STEP export uses the same per-pocket radius with exact arcs.
- **Gold data.** 10 hand-picked real plates (`src/frcpocket/splits.py`, plus your
  thumbs-up ratings). 4 of them, from 2 source files, are held out from all training.
  Real designs are projected onto the rib graph for imitation labels.
- **Aesthetic scorer** (`scorer.py`). A small MLP on scale-free design features
  (`designFeatures.py`), including pocket size, elongation, fatness and sharp corners
  relative to the gold plates, plus a CNN on rasters, blended by held-out skill. Both
  are trained on same-plate ranking pairs: gold vs. random, dense, salt-and-pepper,
  grid, asymmetric, gutted, and ladder layouts, and the graded rating sheets.
- **Generator** (`generator.py`). A graph neural network (~85k parameters) predicts a
  keep probability and a width class per candidate rib. On symmetric plates, decoding
  keeps or drops each rib together with all of its mirror images (both axes and the
  180° turn, whichever the plate has). It drops crossings and ribs meeting at too sharp
  an angle, and connects floating holes along their most likely crossing-free path.
  Two rib styles are decoded and the scorer picks between them. One allows ribs to
  meet at 20°. The other needs 40° and allows at most 4 ribs per small hole, which gives
  the polygon cells of the example plates. Gold plates: the tightest angle between ribs
  at a node is 104° at the median, and holes carry about 2 ribs. It then splits any pocket
  bigger than 3x the gold typical size (3.7% of the plate) or longer than 3x its width,
  using the likely rib that cuts it most evenly into compact pieces. Every split must
  keep the design valid.
- **Training loop** (`trainGenerator.py`). The generator first imitates the gold plates
  and every rating-sheet design graded 4 or higher. Each round then proposes designs
  in both rib styles and keeps the best as new targets. Utility = 0.70 aesthetic + 0.15 structure
  (stiffness kept, stiffness per mass, peak stress) + 0.15 mass removed: looks first,
  by your call. Training uses a fast mode without FEA (looks + mass + a guard against
  one pocket taking over); add `--fea` to include it. Invalid designs are rejected.
  Evaluation and the UI always run FEA, in crash-isolated worker processes.

## Step 4: the UI

```bash
.venv/bin/python -I -m frcpocket.app
```

Open http://127.0.0.1:7860. Upload a flat plate STEP and set wall, rib, corner radius,
boss ring around holes, depth (through or blind), and units (in or mm). The boss ring
defaults to the wall thickness. Real plates often use about half the wall (e.g. 0.12 in
rings with 0.25 in walls on the WCP and Team 221 plates), which looks much closer to them. Click **Generate**. Click a candidate
to select it, rate it 👍/👎, and **Export** it to STEP. Exports run the exact-geometry
checks first and are refused if any rule fails. Ratings go to
`data/curation/ratings.jsonl`, and retraining the scorer picks them up.

**Pocket density** sets the pocket size the ML decodes toward, relative to the size
fitted on the gold plates: Dense (0.6×, the default, more ribs, like your example
plates), Medium (1.0×), or Open (1.5×). The ML still chooses every rib.

Candidates are ML proposals plus the truss baseline (labelled), ranked looks-first.
Designs below the structural floor (FEA: under 8% of the solid plate's stiffness, or
peak stress over 10x) go last and are marked ⚠ structurally weak.

## Host it as a website (Google Cloud Run)

`deploy/build.py` assembles a self-contained copy of the UI in `deploy/cloudrun/`: the
`frcpocket` package, the trained models, `app.py`, pinned `requirements.txt` (CPU torch),
and a `Dockerfile`. Google's Cloud Build turns it into the container, so Docker isn't
needed locally.

```bash
# once: a Google Cloud project with billing, the gcloud CLI, and a sign-in
gcloud init
# every update: rebuild the folder and deploy it
.venv/bin/python -I deploy/build.py --target cloudrun
gcloud run deploy frc-pocketing --source deploy/cloudrun --region us-central1 \
    --allow-unauthenticated --memory 4Gi --cpu 2 --max-instances 1 --concurrency 40 \
    --timeout 900 --session-affinity
```

One instance at most: the UI keeps each visitor's candidates in memory between Generate
and Export, and it caps the cost. Concurrency 40, because one browser tab holds several
connections open (keep-alive, results stream, uploads); Gradio's queue still runs the
heavy work one job at a time. Note that an open tab keeps the instance billed, so close it
when done. To take the site down: `gcloud run services delete frc-pocketing --region us-central1`. Light use stays inside Cloud Run's free tier; storing
the container image (a few GB) costs a few cents a month. The site sleeps when idle, so
the first visit after a quiet spell takes a minute to start. Ratings saved there are
lost when it restarts. `--target space` builds the same app as a Hugging Face Space
instead (Gradio Spaces need Hugging Face PRO).

## Results

Previews, drawn in the colours of your example screenshots:
- `data/eval/showcase/realVsBaselineVsMl_graded.png`: your plate and your example
  plates, real design vs truss baseline vs ML pick;
- `data/eval/showcase/outsidePlateR_graded.png`: Outside Plate R, the scorer's top picks;
- `data/curation/sheets/r1..r3/`: every graded rating sheet.

**Looks (graded by eye).** Over three rounds of grading and retraining, ML designs on
the 14 plates graded every round averaged 3.17, then 3.34, then 3.62 out of 5, and the
truss baseline about 1.9. Outside Plate R went from 2.9 to 3.6, with a best of 4. On
graded plates it never trained on, the scorer now agrees with the grades on 82% of
pairs (136 pairs); the scorer before grading managed 66%. The closest matches are the
plates with many holes (your examples 1 and 4, the WCP and Team 221 gearbox plates).
Plates with few holes come out as clean polygon-cell trusses, still plainer than the
examples.

**Full evaluation with FEA** (`data/eval/round4_graded/summary.md`; "looks" here is the
final scorer, 0–1):

| Group | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Looks |
|---|---|---|---|---|---|---|
| ML, 23 real plates | 100% | 48% | 33% | ×3.6 | 0.61 | 0.72 |
| Truss baseline, same plates | 100% | 32% | 51% | ×2.1 | 0.74 | 0.52 |
| Real designs, same plates | 52% | 46% | 39% | ×3.6 | 0.73 | 0.69 |
| ML, 6 held-out gold plates | 100% | 49% | 24% | ×4.3 | 0.44 | 0.65 |
| Real designs, same 6 plates | 67% | 38% | 38% | ×2.3 | 0.61 | 0.54 |

The ML removes as much material as the real designs and looks at least as good to the
scorer. It is weaker per unit of mass, especially on the held-out plates, because looks
come first (your call). Every ML design passes the validator, and the test plates export as single
valid solids. Real designs "fail" mostly on rule 4: they have pocket corners sharper
than their measured tool radius, and the screenshot traces have pixel-sharp corners.
The UI puts the ML design first on 23 of 27 plates and the truss baseline on 4.

**Plates with big gaps** (`data/eval/showcase/shooterRightPlate_before.png` / `_after.png`):
on the Shooter Right Plate, ribs between holes alone left pockets of up to 11 in². The gap
lattice now fills them with an even web. FEA on its top design: 71% removed, 9% of the
solid plate's stiffness kept, peak stress ×6.2 (truss baseline: 11%, ×5.4). The full FEA
evaluation above was not rerun for this change.

**Speed.** A full retrain (`frcpocket.retrain`) takes about 11 minutes: scorer 7 (sets built
in parallel, about 4 min; fit about 3 min), generator 4. The UI generates candidates for a
large plate in about 35–65 s, roughly half what it took before.

Not verified here: opening the STEP files in Onshape, SolidWorks, or Fusion. They
re-import as one valid solid in OpenCascade (mm units).

## Layout

```
src/frcpocket/
  scrape/polite.py       robots.txt checks, per-host rate limiting, source log
  scrape/github.py       GitHub repo search + STEP/DXF download
  scrape/chiefdelphi.py  Discourse API scraper
  cadio.py               STEP import, flat-plate detection, face loops
  dxfio.py               DXF profiles to polygons
  geom2d.py              hole/pocket classification, rib/wall/boss measurements
  extract.py             raw CAD -> plates.jsonl
  datasetReport.py       dataset statistics + previews
  ingestManual.py        hand-exported plates (Downloads/frcPlates) -> data/raw/manual
  render.py              PNG previews
  plate.py               input plate (outline + keep-out features) and pocketing parameters
  layout.py              rib-graph layouts: candidate graph, pockets from ribs
  baseline.py            procedural truss baseline (Delaunay between features and wall)
  validator.py           the five hard rules (2D) and the solid checks (rule 5)
  cadExport.py           exact pockets cut from the original solid, STEP out
  fea.py                 2D plane-stress FEA structural score (scikit-fem + Triangle)
  testPlates.py          synthetic test plates
  evaluate.py            evaluation rounds: metrics, previews, STEP export
  splits.py              gold plates and the held-out split
  projection.py          real designs mapped onto the rib graph (imitation labels)
  designFeatures.py      scale-free aesthetic features
  raster.py              design rasters for the CNN
  negatives.py           bad layouts for scorer training
  scorer.py              aesthetic scorer (feature MLP + CNN), training + loading
  generator.py           GNN rib generator and decoding
  trainGenerator.py      imitation + generate/validate/score/retrain loop
  pocketer.py            plate -> ranked valid candidates (used by UI and evaluation)
  app.py                 Gradio UI
models/                  aestheticScorer.pt, generator.pt (+ per-round checkpoints)
```
