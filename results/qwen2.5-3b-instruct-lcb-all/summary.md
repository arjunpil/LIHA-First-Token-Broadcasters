# qwen2.5-3b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.987 |  | 0.869 |  | 1.000 | 0.00 / 0.09 |
| L27H13 zero | 0.478 | -0.520 [-0.543, -0.498] | 0.406 | -0.457 [-0.473, -0.441] | 1.000 | 0.12 / 0.43 |
| L27H13 mean | 0.472 | -0.555 [-0.577, -0.533] | 0.401 | -0.463 [-0.478, -0.446] | 0.995 | 0.06 / 0.31 |
| L27H6 zero (control) | 0.990 | +0.002 [-0.003, +0.007] | 0.876 | +0.007 [+0.002, +0.012] | 0.995 | 0.00 / 0.09 |
| L27H12 zero (control) | 0.986 | -0.001 [-0.006, +0.004] | 0.880 | +0.009 [+0.004, +0.015] | 1.000 | 0.00 / 0.08 |
| L27H14 zero (control) | 0.988 | +0.001 [-0.004, +0.006] | 0.830 | -0.037 [-0.045, -0.030] | 0.995 | 0.00 / 0.14 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 | 0.93 | 0.87 | 0.92 | 0.91 | 0.88 | 0.81 | 0.86 | 0.86 | 0.88 | 0.88 | 0.90 | 0.82 | 0.81 | 0.85 |
| L27H13 zero | 0.33 | 0.48 | 0.83 | 0.20 | 0.62 | 0.26 | 0.00 | 0.30 | 0.00 | 0.42 | 0.63 | 0.35 | 0.03 | 0.98 | 1.00 | 0.43 | 0.56 | 0.63 | 0.52 | 0.64 | 0.24 | 0.03 | 0.51 | 0.10 | 0.41 | 0.54 | 0.16 | 0.02 | 0.69 |
| L27H13 mean | 0.05 | 0.31 | 0.90 | 0.52 | 0.36 | 0.08 | 0.00 | 0.56 | 0.00 | 0.39 | 0.05 | 0.40 | 0.00 | 0.98 | 0.99 | 0.24 | 0.65 | 0.77 | 0.73 | 0.50 | 0.13 | 0.06 | 0.61 | 0.02 | 0.45 | 0.17 | 0.30 | 0.00 | 0.70 |
| L27H6 zero (control) | 1.00 | 0.99 | 0.98 | 0.98 | 1.00 | 0.97 | 0.98 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.92 | 0.87 | 0.92 | 0.92 | 0.90 | 0.83 | 0.86 | 0.88 | 0.87 | 0.87 | 0.90 | 0.84 | 0.82 | 0.87 |
| L27H12 zero (control) | 1.00 | 0.99 | 0.97 | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.91 | 0.87 | 0.93 | 0.91 | 0.89 | 0.82 | 0.86 | 0.89 | 0.88 | 0.89 | 0.93 | 0.85 | 0.83 | 0.86 |
| L27H14 zero (control) | 1.00 | 1.00 | 0.98 | 0.99 | 0.99 | 0.96 | 0.97 | 0.99 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 | 0.88 | 0.82 | 0.90 | 0.89 | 0.83 | 0.75 | 0.83 | 0.82 | 0.83 | 0.83 | 0.84 | 0.78 | 0.76 | 0.85 |
