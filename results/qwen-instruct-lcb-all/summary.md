# qwen-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.973 |  | 0.666 |  | 0.995 | 0.01 / 0.30 |
| L22H6 zero | 0.777 | -0.213 [-0.231, -0.195] | 0.386 | -0.277 [-0.292, -0.263] | 0.995 | 0.04 / 0.45 |
| L22H6 mean | 0.738 | -0.257 [-0.276, -0.237] | 0.376 | -0.286 [-0.302, -0.273] | 0.995 | 0.04 / 0.43 |
| L22H0 zero (control) | 0.981 | +0.007 [+0.000, +0.014] | 0.666 | +0.001 [-0.005, +0.008] | 0.995 | 0.00 / 0.31 |
| L22H7 zero (control) | 0.975 | +0.002 [-0.003, +0.006] | 0.668 | +0.002 [-0.003, +0.008] | 0.990 | 0.01 / 0.30 |
| L22H11 zero (control) | 0.977 | +0.004 [+0.000, +0.008] | 0.665 | -0.001 [-0.004, +0.003] | 0.995 | 0.01 / 0.31 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 0.99 | 0.99 | 0.90 | 1.00 | 0.96 | 0.95 | 0.95 | 1.00 | 0.95 | 1.00 | 0.98 | 0.99 | 0.73 | 0.61 | 0.75 | 0.76 | 0.74 | 0.65 | 0.69 | 0.54 | 0.50 | 0.65 | 0.72 | 0.56 | 0.61 | 0.81 |
| L22H6 zero | 0.96 | 0.96 | 0.83 | 0.73 | 0.39 | 0.79 | 0.00 | 0.95 | 0.87 | 0.54 | 1.00 | 0.39 | 0.82 | 0.98 | 0.99 | 0.64 | 0.55 | 0.57 | 0.58 | 0.13 | 0.19 | 0.00 | 0.51 | 0.34 | 0.08 | 0.70 | 0.22 | 0.10 | 0.78 |
| L22H6 mean | 0.96 | 0.96 | 0.97 | 0.68 | 0.39 | 0.67 | 0.01 | 0.93 | 0.89 | 0.03 | 1.00 | 0.46 | 0.68 | 0.97 | 0.99 | 0.63 | 0.56 | 0.67 | 0.59 | 0.07 | 0.14 | 0.00 | 0.49 | 0.28 | 0.00 | 0.69 | 0.27 | 0.06 | 0.78 |
| L22H0 zero (control) | 0.98 | 1.00 | 0.99 | 0.99 | 1.00 | 0.93 | 1.00 | 0.97 | 0.98 | 0.96 | 1.00 | 0.94 | 0.99 | 0.99 | 0.99 | 0.74 | 0.60 | 0.76 | 0.75 | 0.72 | 0.64 | 0.68 | 0.58 | 0.50 | 0.67 | 0.72 | 0.55 | 0.61 | 0.81 |
| L22H7 zero (control) | 0.99 | 0.99 | 0.97 | 0.98 | 1.00 | 0.94 | 1.00 | 0.95 | 0.95 | 0.96 | 1.00 | 0.96 | 0.99 | 0.99 | 0.99 | 0.72 | 0.60 | 0.75 | 0.75 | 0.76 | 0.65 | 0.69 | 0.57 | 0.51 | 0.66 | 0.72 | 0.56 | 0.60 | 0.81 |
| L22H11 zero (control) | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.92 | 1.00 | 0.96 | 0.95 | 0.96 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 0.73 | 0.60 | 0.74 | 0.76 | 0.74 | 0.65 | 0.70 | 0.55 | 0.50 | 0.66 | 0.72 | 0.56 | 0.60 | 0.82 |
