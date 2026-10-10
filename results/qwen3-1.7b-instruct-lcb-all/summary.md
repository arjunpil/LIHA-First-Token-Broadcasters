# qwen3-1.7b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.971 |  | 0.789 |  | 0.985 | 0.00 / 0.16 |
| L18H12 zero | 0.658 | -0.293 [-0.313, -0.273] | 0.198 | -0.592 [-0.608, -0.577] | 0.990 | 0.12 / 0.76 |
| L18H12 mean | 0.661 | -0.304 [-0.324, -0.284] | 0.294 | -0.497 [-0.514, -0.481] | 0.995 | 0.04 / 0.57 |
| L18H6 zero (control) | 0.975 | +0.003 [-0.004, +0.009] | 0.796 | +0.008 [+0.001, +0.014] | 0.985 | 0.00 / 0.16 |
| L18H13 zero (control) | 0.966 | -0.004 [-0.011, +0.002] | 0.791 | +0.002 [-0.005, +0.009] | 0.980 | 0.00 / 0.16 |
| L18H14 zero (control) | 0.967 | -0.004 [-0.010, +0.003] | 0.797 | +0.009 [+0.001, +0.016] | 0.975 | 0.00 / 0.16 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.00 | 1.00 | 0.98 | 0.99 | 1.00 | 0.92 | 1.00 | 0.97 | 1.00 | 0.95 | 0.99 | 0.98 | 0.99 | 0.89 | 0.98 | 0.82 | 0.78 | 0.85 | 0.85 | 0.84 | 0.71 | 0.81 | 0.74 | 0.71 | 0.84 | 0.84 | 0.71 | 0.76 | 0.77 |
| L18H12 zero | 0.12 | 0.90 | 0.86 | 0.87 | 0.78 | 0.78 | 0.62 | 0.40 | 0.39 | 0.78 | 0.69 | 0.53 | 0.97 | 0.88 | 0.99 | 0.07 | 0.28 | 0.29 | 0.40 | 0.28 | 0.18 | 0.29 | 0.05 | 0.00 | 0.27 | 0.27 | 0.13 | 0.21 | 0.04 |
| L18H12 mean | 0.08 | 0.99 | 0.92 | 0.96 | 0.84 | 0.43 | 0.74 | 0.69 | 0.18 | 0.55 | 0.54 | 0.79 | 0.78 | 0.84 | 0.99 | 0.07 | 0.56 | 0.60 | 0.65 | 0.27 | 0.28 | 0.56 | 0.05 | 0.01 | 0.30 | 0.23 | 0.29 | 0.21 | 0.02 |
| L18H6 zero (control) | 1.00 | 1.00 | 0.98 | 0.99 | 1.00 | 0.94 | 0.99 | 0.97 | 1.00 | 0.95 | 0.99 | 1.00 | 0.98 | 0.91 | 0.98 | 0.84 | 0.78 | 0.88 | 0.86 | 0.84 | 0.70 | 0.84 | 0.75 | 0.73 | 0.84 | 0.84 | 0.72 | 0.76 | 0.78 |
| L18H13 zero (control) | 1.00 | 1.00 | 0.98 | 0.98 | 0.98 | 0.94 | 1.00 | 0.95 | 0.99 | 0.95 | 0.99 | 0.99 | 0.98 | 0.88 | 0.98 | 0.82 | 0.77 | 0.87 | 0.84 | 0.84 | 0.69 | 0.81 | 0.74 | 0.76 | 0.85 | 0.83 | 0.72 | 0.75 | 0.77 |
| L18H14 zero (control) | 0.99 | 1.00 | 0.97 | 0.99 | 1.00 | 0.93 | 1.00 | 0.98 | 0.99 | 0.94 | 0.99 | 0.98 | 0.99 | 0.88 | 0.97 | 0.84 | 0.80 | 0.88 | 0.85 | 0.84 | 0.69 | 0.85 | 0.75 | 0.69 | 0.86 | 0.85 | 0.72 | 0.79 | 0.77 |
