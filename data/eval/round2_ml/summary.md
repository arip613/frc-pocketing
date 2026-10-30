# Evaluation: round2_ml

| Group | Plates | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Aesthetic |
|---|---|---|---|---|---|---|---|
| test plates / ml alone | 4 | 100% | 39% | 38% | x3.24 | 0.52 | 0.90 |
| test plates / tool pick (ML or baseline) | 4 | 100% | 46% | 28% | x3.29 | 0.52 | 0.84 |
| test plates / baseline | 4 | 100% | 44% | 30% | x3.15 | 0.53 | 0.82 |
| real plates (all 15) / ml alone | 15 | 100% | 45% | 38% | x3.06 | 0.74 | 0.81 |
| real plates (all 15) / tool pick (ML or baseline) | 15 | 100% | 44% | 37% | x2.84 | 0.67 | 0.82 |
| real plates (all 15) / baseline | 15 | 100% | 29% | 57% | x1.90 | 0.82 | 0.70 |
| real plates (all 15) / real teams | 15 | 80% | 45% | 44% | x4.07 | 0.83 | 0.82 |
| held-out gold plates / ml alone | 4 | 100% | 52% | 23% | x4.04 | 0.46 | 0.91 |
| held-out gold plates / tool pick (ML or baseline) | 4 | 100% | 52% | 23% | x4.04 | 0.46 | 0.91 |
| held-out gold plates / baseline | 4 | 100% | 26% | 49% | x2.08 | 0.65 | 0.69 |
| held-out gold plates / real teams | 4 | 100% | 40% | 35% | x2.49 | 0.60 | 0.83 |

Tool pick: ML design on 12 of 19 plates, truss baseline on 7.

STEP export (exact geometry, rule 5) on test plates: 100% pass.

Real designs failing the validator under their own measured parameters:
- 2d57a227b9_0 (JSchallert/Archangel-FRC-2016): 4 corner radius
- f917254f76_0 (RedHotChiliBots/FRC4453_COTS): 4 corner radius
- f1da8bbfec_90 (Spectrum3847/Spectrum-CAD-Library): 4 corner radius

Previews: `previews/` (real design, then baseline, then the generator).
