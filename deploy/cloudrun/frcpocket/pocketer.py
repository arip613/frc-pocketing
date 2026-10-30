"""High-level API: plate + parameters -> ranked, validated pocketing candidates.

Used by the UI and the evaluation. Candidates come from the trained ML generator (if
models/generator.pt exists) and the truss baseline, which is always included and
labelled so the two can be compared. The generator proposes in each rib style, and the
aesthetic scorer (trained on graded rating sheets) picks among them. Every candidate
shown has passed the validator.
"""

from dataclasses import replace
from pathlib import Path

from shapely.ops import unary_union

from frcpocket.baseline import trussBaseline
from frcpocket.trainGenerator import Judge, isSound

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

    def candidates(self, plate, params, count=6, proposals=16, seed=0):
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
        for result in baselineRanked:
            result["source"] = "truss baseline"
            chosen.append(result)
        # One ranking across sources: sound designs first (looks-weighted utility), then any
        # below the soundness floor, flagged so the UI can warn.
        for result in chosen:
            known = result["stiffnessKept"] is not None
            result["sound"] = (not known) or isSound(result["stiffnessKept"], result["stressIncrease"])
        return sorted(chosen, key=lambda result: (not result["sound"], -result["utility"]))
