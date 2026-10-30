# Evaluation: round0_baseline

| Group | Plates | Valid | Mass removed | Stiffness kept | Peak stress | Stiffness per mass kept | Aesthetic |
|---|---|---|---|---|---|---|---|
| test plates / baseline | 4 | 100% | 46% | 27% | x3.39 | 0.50 | n/a |
| real plates / baseline | 15 | 100% | 35% | 49% | x2.43 | 0.75 | n/a |
| real plates / real teams | 15 | 80% | 45% | 44% | x4.06 | 0.83 | n/a |

STEP export (exact geometry, rule 5) on test plates: 100% pass.

Real designs failing the validator under their own measured parameters:
- 2d57a227b9_0 (JSchallert/Archangel-FRC-2016): 4 corner radius
- f917254f76_0 (RedHotChiliBots/FRC4453_COTS): 4 corner radius
- f1da8bbfec_90 (Spectrum3847/Spectrum-CAD-Library): 4 corner radius

Previews: `previews/` (real design left, generated right).
