# gemma3-4b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.990 |  | 0.133 |  | 0.985 | 0.00 / 0.44 |
| L24H0 zero | 0.781 | -0.259 [-0.292, -0.227] | 0.027 | -0.107 [-0.126, -0.090] | 1.000 | 0.04 / 0.55 |
| L24H0 mean | 0.902 | -0.120 [-0.145, -0.095] | 0.080 | -0.053 [-0.066, -0.040] | 0.990 | 0.00 / 0.48 |
| L24H4 zero (control) | 0.989 | +0.000 [-0.007, +0.007] | 0.133 | +0.000 [-0.008, +0.008] | 0.995 | 0.00 / 0.43 |
| L24H6 zero (control) | 0.989 | -0.001 [-0.010, +0.006] | 0.140 | +0.007 [-0.003, +0.017] | 0.980 | 0.00 / 0.43 |
| L24H7 zero (control) | 0.991 | +0.000 [-0.006, +0.006] | 0.145 | +0.012 [+0.003, +0.020] | 0.995 | 0.00 / 0.43 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.13 | 0.11 | 0.15 |
| L24H0 zero | 0.83 | 0.83 | 0.84 | 0.00 | 1.00 | 0.05 | 0.03 | 0.02 | 0.00 |
| L24H0 mean | 0.91 | 0.80 | 0.98 | 0.49 | 0.99 | 0.11 | 0.06 | 0.07 | 0.07 |
| L24H4 zero (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.14 | 0.17 | 0.09 | 0.14 |
| L24H6 zero (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.98 | 0.17 | 0.13 | 0.09 | 0.17 |
| L24H7 zero (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.16 | 0.17 | 0.11 | 0.15 |
