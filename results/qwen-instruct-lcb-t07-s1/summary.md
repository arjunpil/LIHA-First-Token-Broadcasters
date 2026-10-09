# qwen-instruct: Language Confusion Benchmark

Sampling at temperature 0.7, top-p 0.8, top-k 20 (seed 1), 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.982 |  | 0.698 |  | 0.990 | 0.00 / 0.28 |
| L22H6 zero | 0.747 | -0.274 [-0.306, -0.241] | 0.423 | -0.276 [-0.302, -0.249] | 1.000 | 0.03 / 0.38 |
| L22H6 mean | 0.802 | -0.223 [-0.253, -0.191] | 0.456 | -0.242 [-0.267, -0.216] | 1.000 | 0.02 / 0.35 |
| L22H0 zero (control) | 0.980 | +0.000 [-0.008, +0.008] | 0.702 | +0.004 [-0.008, +0.017] | 0.990 | 0.00 / 0.28 |
| L22H7 zero (control) | 0.982 | +0.001 [-0.005, +0.009] | 0.697 | -0.003 [-0.012, +0.007] | 0.990 | 0.00 / 0.28 |
| L22H11 zero (control) | 0.978 | -0.003 [-0.010, +0.005] | 0.701 | +0.003 [-0.005, +0.009] | 0.990 | 0.00 / 0.28 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 1.00 | 0.99 | 0.98 | 0.99 | 0.74 | 0.61 | 0.77 | 0.68 |
| L22H6 zero | 0.74 | 0.93 | 0.84 | 0.00 | 1.00 | 0.58 | 0.54 | 0.58 | 0.00 |
| L22H6 mean | 0.74 | 0.93 | 0.98 | 0.00 | 1.00 | 0.60 | 0.54 | 0.69 | 0.00 |
| L22H0 zero (control) | 0.98 | 1.00 | 0.98 | 0.99 | 0.99 | 0.74 | 0.62 | 0.76 | 0.68 |
| L22H7 zero (control) | 0.98 | 1.00 | 0.98 | 0.99 | 0.99 | 0.73 | 0.62 | 0.77 | 0.67 |
| L22H11 zero (control) | 0.98 | 1.00 | 0.98 | 0.98 | 0.99 | 0.74 | 0.61 | 0.77 | 0.69 |
