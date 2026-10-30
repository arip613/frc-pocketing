# Evaluation: round4_graded

| Group | Plates | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Aesthetic |
|---|---|---|---|---|---|---|---|
| test plates / ml alone | 4 | 100% | 42% | 38% | x4.13 | 0.53 | 0.82 |
| test plates / tool pick (ML or baseline) | 4 | 100% | 54% | 16% | x5.17 | 0.35 | 0.73 |
| test plates / baseline | 4 | 100% | 45% | 29% | x3.24 | 0.52 | 0.57 |
| real plates (all 15) / ml alone | 23 | 100% | 48% | 33% | x3.57 | 0.61 | 0.72 |
| real plates (all 15) / tool pick (ML or baseline) | 23 | 100% | 48% | 33% | x3.44 | 0.62 | 0.72 |
| real plates (all 15) / baseline | 23 | 100% | 32% | 51% | x2.06 | 0.74 | 0.52 |
| real plates (all 15) / real teams | 23 | 52% | 46% | 39% | x3.60 | 0.73 | 0.69 |
| held-out gold plates / ml alone | 6 | 100% | 49% | 24% | x4.28 | 0.44 | 0.65 |
| held-out gold plates / tool pick (ML or baseline) | 6 | 100% | 49% | 24% | x4.28 | 0.44 | 0.65 |
| held-out gold plates / baseline | 6 | 100% | 24% | 51% | x1.93 | 0.66 | 0.35 |
| held-out gold plates / real teams | 6 | 67% | 38% | 38% | x2.33 | 0.61 | 0.54 |

Tool pick: ML design on 23 of 27 plates, truss baseline on 4.

STEP export (exact geometry, rule 5) on test plates: 100% pass.

Real designs failing the validator under their own measured parameters:
- 2d57a227b9_0 (JSchallert/Archangel-FRC-2016): 4 corner radius
- f917254f76_0 (RedHotChiliBots/FRC4453_COTS): 4 corner radius
- f1da8bbfec_90 (Spectrum3847/Spectrum-CAD-Library): 4 corner radius
- userImg_example1 (user-provided example): 4 corner radius
- userImg_example2 (user-provided example): 4 corner radius
- userImg_example3 (user-provided example): 4 corner radius
- userImg_example4 (user-provided example): 4 corner radius
- userImg_example5 (user-provided example): 4 corner radius
- userImg_example6 (user-provided example): 4 corner radius, 4 pocket size
- userImg_example7 (user-provided example): 4 corner radius
- userImg_example8 (user-provided example): 4 corner radius

Previews: `previews/` (real design, then baseline, then the generator).
