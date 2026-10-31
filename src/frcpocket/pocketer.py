"""High-level API: plate + parameters -> ranked, validated pocketing candidates.

Used by the UI and the evaluation. Candidates come from the trained ML generators (v2:
the live generator and the marathon e1, u1 and m1 generators, each proposing) and the
truss baseline, which is always included and labelled so the two can be compared. Each
generator proposes in each rib style, and the taste model (taste.py, fitted to the user's
and the expert's preferences) picks among them. Every candidate shown has passed the
validator. The v1 setup was models/generator.pt with models/aestheticScorer.pt.

ML designs then go through the polish pass (polish.py: symmetry, hole-to-hole links,
blob filling, pocket-size cap), on by default, and are validated and scored again.
"""

from dataclasses import replace
from pathlib import Path

from shapely.ops import unary_union

from frcpocket import fea, polish as polishPass
from frcpocket.baseline import trussBaseline
from frcpocket.trainGenerator import Judge, coarseMeshFactor, isSound

# v2: in blind picks on 8 new plates the user chose it over v1 on 5 of 5 plates where they differed.
defaultGeneratorPath = (Path("models/generator.pt"), Path("models/marathon/e1/generator.pt"),
                        Path("models/marathon/u1/generator.pt"), Path("models/marathon/m1/generator.pt"))
defaultScorerPath = Path("models/marathon/v2/taste.json")
duplicateIoU = 0.97
# Rib styles the generator proposes in (min rib-rib angle, max ribs per small hole; 0 = no
# limit). Average grades on the rating sheets: round 1, 20 deg 3.26 and 40 deg / 4 ribs 3.25
# (50 deg / 3 ribs 2.87); round 2, 40 deg 3.39, 40 deg / 4 ribs 3.31, 20 deg 3.16. Each wins
# on some plates, so all three are proposed and the scorer picks.
# Rib styles (min rib-rib angle, max ribs per small hole). 30 deg is the floor experienced
# teams give so filleted corners do not end in wedges (docs/pocketingResearch.md).
ribStyles = ((30.0, 0), (40.0, 0), (40.0, 4))


class Pocketer:
    def __init__(self, generatorPath=defaultGeneratorPath, scorerPath=defaultScorerPath, workers=2):
        """generatorPath may also be a list of generators: each proposes for every plate,
        and the scorer and the structural floor pick among all their designs."""
        from frcpocket.generator import RibGenerator

        paths = [generatorPath] if isinstance(generatorPath, (str, Path)) else list(generatorPath)
        self.generators = [RibGenerator(path) for path in paths if Path(path).exists()]
        self.generator = self.generators[0] if self.generators else None
        self.judge = Judge(scorerPath, workers=workers)

    def candidates(self, plate, params, count=6, proposals=16, seed=0, polish=None, relax=True):
        """Ranked, validated designs. polish=None follows polish.enabled (on by default).

        If no ML design passes the structural floor (small or thin plates under big pockets
        and 3/16 in ribs), it tries again with smaller pockets, then with a 1/8 in rib, and
        returns the first attempt that has a sound ML design; each result notes the settings
        used in result["relaxed"]."""
        results = self._candidates(plate, params, count, proposals, seed, polish)
        if not relax or any(r["sound"] and r["source"] == "ML generator" for r in results):
            return results
        attempts = [("smaller pockets", replace(params, pocketSizeFactor=params.pocketSizeFactor * 0.6))]
        if params.ribIn > 0.125:
            attempts.append(("smaller pockets, 1/8 in rib", replace(params, pocketSizeFactor=params.pocketSizeFactor * 0.6, ribIn=0.125)))
        for note, relaxed in attempts:
            retry = self._candidates(plate, relaxed, count, proposals, seed, polish)
            if any(r["sound"] and r["source"] == "ML generator" for r in retry):
                for r in retry:
                    r["relaxed"] = note
                    r["params"] = relaxed
                return retry
        return results

    def _candidates(self, plate, params, count, proposals, seed, polish):
        polish = polishPass.enabled if polish is None else polish
        layouts = []
        for generator in self.generators:
            for angle, cap in ribStyles:
                styleParams = replace(params, minRibAngleDegrees=angle, maxHoleRibs=cap)
                layouts += generator.propose(plate, styleParams, count=max(4, proposals // len(ribStyles)), seed=seed)
        baseline = trussBaseline(plate, params)
        ranked = self.judge.rank(plate, params, layouts, feaCount=max(count * 2, 6)) if layouts else []
        baselineRanked = self.judge.rank(plate, params, [baseline], feaCount=1)
        chosen = []
        for result in ranked:
            if len(chosen) >= count - (1 if baselineRanked else 0):
                break
            pockets = unary_union(result["layout"].pockets)
            duplicate = any(
                pockets.intersection(other).area / max(pockets.union(other).area, 1e-9) > duplicateIoU
                for other in (unary_union(item["layout"].pockets) for item in chosen)
            )
            if not duplicate:
                result["source"] = "ML generator"
                chosen.append(result)
        if polish and chosen:
            chosen = self.polished(plate, params, chosen)
        for result in baselineRanked:
            result["source"] = "truss baseline"
            chosen.append(result)
        # One ranking across sources: sound designs first (looks-weighted utility), then any
        # below the soundness floor, flagged so the UI can warn.
        for result in chosen:
            known = result["stiffnessKept"] is not None
            result["sound"] = (not known) or isSound(result["stiffnessKept"], result["stressIncrease"])
        return sorted(chosen, key=lambda result: (not result["sound"], -result["utility"]))

    @staticmethod
    def sound(result):
        return result["stiffnessKept"] is None or isSound(result["stiffnessKept"], result["stressIncrease"])

    def isSoundDesign(self, plate, params, pockets):
        """The structural floor (coarse FEA, as in ranking); unknown (FEA failed) counts as sound."""
        meshSize = fea.defaultMeshSize(plate, params) * coarseMeshFactor
        result = self.judge.structural(plate, params, [pockets], meshSize)[0] if self.judge.useFea else None
        return result is None or isSound(result["stiffnessKept"], result["stressIncrease"])

    def polished(self, plate, params, results):
        """Polish each chosen ML design, then validate and score it again.

        The steps are first applied with the validator only; if the finished design falls
        below the structural floor while the original was above it, polishing is redone with
        the floor checked after every step, so the offending steps are reverted."""
        out = []
        for result in results:
            original = result["layout"]
            outcome = polishPass.polishLayout(plate, params, original)
            if not outcome.changed:
                result["polish"] = outcome.log
                out.append(result)
                continue
            rescored = self.judge.rank(plate, params, [outcome.layout], feaCount=1)
            if rescored and not self.sound(rescored[0]):
                # Too weak: redo it with the floor checked after every step, and without the
                # blob and hugger steps (they remove material); keep the best sound result.
                retries = [(outcome, rescored)]
                for kwargs in ({"structuralCheck": lambda pockets: self.isSoundDesign(plate, params, pockets)},
                               {"skip": ("blobs", "huggers")}):
                    retry = polishPass.polishLayout(plate, params, original, **kwargs)
                    if retry.changed:
                        retries.append((retry, self.judge.rank(plate, params, [retry.layout], feaCount=1)))
                sound = [(o, r) for o, r in retries if r and self.sound(r[0])]
                if sound:
                    outcome, rescored = max(sound, key=lambda item: item[1][0]["utility"])
                elif self.sound(result):
                    rescored = []  # every polished version is weak and the original isn't: keep it
            if not rescored:
                result["polish"] = outcome.log + ["kept unpolished"]
                out.append(result)
                continue
            rescored[0]["source"] = result["source"]
            rescored[0]["polish"] = outcome.log
            out.append(rescored[0])
        unique = []
        for result in out:
            pockets = unary_union(result["layout"].pockets)
            if not any(pockets.intersection(other).area / max(pockets.union(other).area, 1e-9) > duplicateIoU
                       for other in (unary_union(item["layout"].pockets) for item in unique)):
                unique.append(result)
        return unique


class CombinedPocketer:
    """Several model sets, each a (generator, scorer) pair: each designs the plate with its
    own ranking, and their best sound designs are offered lightest first.

    On 18 plates where the user picked their favourite of the four marathon sets' designs
    (live, e1, u1, m1), the lightest design was the pick on 10, the live ranking's choice
    on 5, and a random one would be on about 4.7. On 7 fresh plates picked blind after
    that rule was set, the lightest was the pick on 6, the live ranking's on 0 (random 1.8)."""

    def __init__(self, modelSets, workers=2):
        self.pocketers = [Pocketer(generatorPath, scorerPath, workers=workers) for generatorPath, scorerPath in modelSets]

    def candidates(self, plate, params, count=6, **kwargs):
        finalists, baselines = [], []
        for index, pocketer in enumerate(self.pocketers):
            for result in pocketer.candidates(plate, params, count=2, **kwargs):
                result["modelSet"] = index
                (finalists if result["source"] == "ML generator" else baselines).append(result)
        unique = []
        for result in sorted(finalists, key=lambda r: (not r["sound"], -r["removed"])):
            pockets = unary_union(result["layout"].pockets)
            if not any(pockets.intersection(other).area / max(pockets.union(other).area, 1e-9) > duplicateIoU
                       for other in (unary_union(item["layout"].pockets) for item in unique)):
                unique.append(result)
        return unique[: max(1, count - 1)] + baselines[:1]
