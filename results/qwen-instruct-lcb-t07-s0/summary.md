# qwen-instruct: Language Confusion Benchmark

Sampling at temperature 0.7, top-p 0.8, top-k 20 (seed 0), 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.985 |  | 0.690 |  | 0.995 | 0.00 / 0.29 |
| L22H6 zero | 0.745 | -0.273 [-0.303, -0.241] | 0.444 | -0.245 [-0.272, -0.220] | 1.000 | 0.02 / 0.38 |
| L22H6 mean | 0.784 | -0.238 [-0.267, -0.209] | 0.458 | -0.232 [-0.258, -0.208] | 0.995 | 0.02 / 0.36 |
| L22H0 zero (control) | 0.992 | +0.006 [-0.003, +0.015] | 0.678 | -0.013 [-0.024, -0.001] | 0.995 | 0.00 / 0.30 |
| L22H7 zero (control) | 0.990 | +0.005 [-0.001, +0.013] | 0.692 | +0.002 [-0.007, +0.011] | 1.000 | 0.00 / 0.29 |
| L22H11 zero (control) | 0.984 | +0.000 [-0.008, +0.008] | 0.695 | +0.005 [+0.000, +0.011] | 1.000 | 0.00 / 0.28 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.99 | 0.98 | 0.99 | 0.73 | 0.59 | 0.77 | 0.67 |
| L22H6 zero | 0.73 | 0.93 | 0.86 | 0.00 | 1.00 | 0.60 | 0.55 | 0.63 | 0.00 |
| L22H6 mean | 0.71 | 0.93 | 0.98 | 0.00 | 0.99 | 0.59 | 0.54 | 0.71 | 0.00 |
| L22H0 zero (control) | 1.00 | 1.00 | 0.99 | 0.98 | 0.99 | 0.71 | 0.57 | 0.77 | 0.66 |
| L22H7 zero (control) | 0.99 | 1.00 | 0.99 | 0.99 | 1.00 | 0.73 | 0.59 | 0.77 | 0.67 |
| L22H11 zero (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.73 | 0.60 | 0.78 | 0.68 |
