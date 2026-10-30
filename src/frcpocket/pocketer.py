"""High-level API: plate + parameters -> ranked, validated pocketing candidates.

Used by the UI and the evaluation. Candidates come from the trained ML generator (if
models/generator.pt exists) and the truss baseline, which is always included and
labelled so the two can be compared. The generator proposes in each rib style, and the
aesthetic scorer (trained on graded rating sheets) picks among them. Every candidate
shown has passed the validator.

ML designs then go through the polish pass (polish.py: symmetry, hole-to-hole links,
blob filling, pocket-size cap), on by default, and are validated and scored again.
"""

from dataclasses import replace
from pathlib import Path

from shapely.ops import unary_union

from frcpocket import fea, polish as polishPass
from frcpocket.baseline import trussBaseline
from frcpocket.trainGenerator import Judge, coarseMeshFactor, isSound

defaultGeneratorPath = Path("models/generator.pt")
defaultScorerPath = Path("models/aestheticScorer.pt")
duplicateIoU = 0.97
# Rib styles the generator proposes in (min rib-rib angle, max ribs per small hole; 0 = no
# limit). Average grades on the rating sheets: round 1, 20 deg 3.26 and 40 deg / 4 ribs 3.25
# (50 deg / 3 ribs 2.87); round 2, 40 deg 3.39, 40 deg / 4 ribs 3.31, 20 deg 3.16. Each wins
# on some plates, so all three are proposed and the scorer picks.
ribStyles = ((20.0, 0), (40.0, 0), (40.0, 4))


class Pocketer:
    def __init__(self, generatorPath=defaultGeneratorPath, scorerPath=defaultScorerPath, workers=2):
        from frcpocket.generator import RibGenerator

        self.generator = RibGenerator(generatorPath) if Path(generatorPath).exists() else None
        self.judge = Judge(scorerPath, workers=workers)

    def candidates(self, plate, params, count=6, proposals=16, seed=0, polish=None):
        """Ranked, validated designs. polish=None follows polish.enabled (on by default)."""
        polish = polishPass.enabled if polish is None else polish
        layouts = []
        if self.generator is not None:
            for angle, cap in ribStyles:
                styleParams = replace(params, minRibAngleDegrees=angle, maxHoleRibs=cap)
                layouts += self.generator.propose(plate, styleParams, count=max(4, proposals // len(ribStyles)), seed=seed)
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
                # blob step (the one that removes material); keep the best sound result.
                retries = [(outcome, rescored)]
                for kwargs in ({"structuralCheck": lambda pockets: self.isSoundDesign(plate, params, pockets)},
                               {"skip": ("blobs",)}):
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
