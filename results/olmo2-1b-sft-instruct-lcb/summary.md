# olmo2-1b-sft-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.997 |  | 0.895 |  | 1.000 | 0.00 / 0.09 |
| L12H8 zero | 0.928 | -0.076 [-0.095, -0.059] | 0.644 | -0.247 [-0.273, -0.221] | 1.000 | 0.03 / 0.31 |
| L12H8 mean | 0.960 | -0.038 [-0.052, -0.025] | 0.798 | -0.091 [-0.109, -0.074] | 1.000 | 0.01 / 0.14 |
| L12H6 zero (control) | 0.992 | -0.004 [-0.009, +0.000] | 0.893 | -0.003 [-0.010, +0.003] | 1.000 | 0.00 / 0.09 |
| L12H13 zero (control) | 0.997 | +0.000 [+0.000, +0.000] | 0.894 | +0.000 [-0.008, +0.008] | 1.000 | 0.00 / 0.08 |
| L12H14 zero (control) | 0.995 | -0.003 [-0.006, +0.000] | 0.885 | -0.008 [-0.018, +0.002] | 1.000 | 0.00 / 0.09 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.90 | 0.90 | 0.90 | 0.88 |
| L12H8 zero | 0.93 | 0.90 | 0.96 | 0.79 | 1.00 | 0.66 | 0.61 | 0.71 | 0.58 |
| L12H8 mean | 0.96 | 0.96 | 0.98 | 0.88 | 1.00 | 0.86 | 0.77 | 0.82 | 0.74 |
| L12H6 zero (control) | 0.99 | 1.00 | 1.00 | 0.98 | 1.00 | 0.91 | 0.89 | 0.91 | 0.87 |
| L12H13 zero (control) | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.89 | 0.90 | 0.91 | 0.87 |
| L12H14 zero (control) | 0.99 | 0.99 | 1.00 | 0.98 | 1.00 | 0.90 | 0.88 | 0.89 | 0.87 |
