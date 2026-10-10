# olmo2-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.976 |  | 0.873 |  | 0.990 | 0.01 / 0.09 |
| L12H8 zero | 0.662 | -0.346 [-0.367, -0.327] | 0.491 | -0.379 [-0.394, -0.364] | 0.990 | 0.20 / 0.35 |
| L12H8 mean | 0.537 | -0.445 [-0.467, -0.424] | 0.430 | -0.442 [-0.458, -0.427] | 1.000 | 0.13 / 0.21 |
| L12H6 zero (control) | 0.975 | -0.002 [-0.006, +0.002] | 0.872 | -0.002 [-0.007, +0.003] | 1.000 | 0.01 / 0.09 |
| L12H13 zero (control) | 0.977 | +0.002 [-0.003, +0.007] | 0.883 | +0.010 [+0.004, +0.015] | 0.995 | 0.01 / 0.08 |
| L12H14 zero (control) | 0.973 | -0.001 [-0.008, +0.005] | 0.881 | +0.007 [-0.000, +0.015] | 0.990 | 0.01 / 0.09 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.98 | 0.99 | 0.99 | 0.92 | 0.86 | 0.95 | 0.98 | 1.00 | 0.97 | 1.00 | 1.00 | 0.96 | 0.93 | 0.99 | 0.84 | 0.92 | 0.94 | 0.95 | 0.87 | 0.78 | 0.92 | 0.84 | 0.76 | 0.87 | 0.92 | 0.86 | 0.89 | 0.87 |
| L12H8 zero | 0.55 | 0.47 | 0.77 | 0.81 | 0.91 | 0.31 | 0.49 | 0.40 | 0.56 | 0.28 | 0.75 | 0.93 | 0.85 | 0.57 | 0.99 | 0.26 | 0.57 | 0.64 | 0.64 | 0.40 | 0.44 | 0.52 | 0.30 | 0.32 | 0.18 | 0.62 | 0.61 | 0.65 | 0.73 |
| L12H8 mean | 0.10 | 0.60 | 0.92 | 0.94 | 0.78 | 0.23 | 0.65 | 0.26 | 0.17 | 0.09 | 0.04 | 0.98 | 0.81 | 0.43 | 1.00 | 0.00 | 0.80 | 0.79 | 0.87 | 0.13 | 0.42 | 0.68 | 0.20 | 0.07 | 0.06 | 0.05 | 0.72 | 0.62 | 0.57 |
| L12H6 zero (control) | 0.99 | 0.98 | 0.99 | 0.99 | 0.93 | 0.84 | 0.96 | 0.99 | 1.00 | 0.96 | 1.00 | 1.00 | 0.96 | 0.92 | 1.00 | 0.85 | 0.92 | 0.95 | 0.94 | 0.86 | 0.79 | 0.91 | 0.85 | 0.75 | 0.86 | 0.91 | 0.86 | 0.89 | 0.86 |
| L12H13 zero (control) | 0.99 | 0.97 | 0.99 | 0.98 | 0.93 | 0.85 | 0.97 | 0.96 | 1.00 | 0.98 | 1.00 | 1.00 | 0.95 | 0.95 | 0.99 | 0.85 | 0.93 | 0.95 | 0.94 | 0.87 | 0.78 | 0.91 | 0.85 | 0.78 | 0.90 | 0.93 | 0.87 | 0.91 | 0.89 |
| L12H14 zero (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.92 | 0.85 | 0.98 | 0.97 | 0.98 | 0.97 | 1.00 | 1.00 | 0.96 | 0.93 | 0.99 | 0.84 | 0.91 | 0.94 | 0.94 | 0.88 | 0.79 | 0.91 | 0.84 | 0.78 | 0.91 | 0.93 | 0.86 | 0.90 | 0.88 |
