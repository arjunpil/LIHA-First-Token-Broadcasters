# qwen-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.982 |  | 0.704 |  | 0.995 | 0.00 / 0.28 |
| L16H9 zero | 0.971 | -0.009 [-0.021, +0.003] | 0.247 | -0.459 [-0.487, -0.430] | 0.995 | 0.01 / 0.74 |
| L16H9 mean | 0.992 | +0.008 [-0.003, +0.018] | 0.428 | -0.277 [-0.303, -0.251] | 0.995 | 0.00 / 0.56 |
| L16H0 zero (control) | 0.984 | +0.001 [-0.010, +0.013] | 0.674 | -0.029 [-0.045, -0.013] | 0.995 | 0.00 / 0.31 |
| L16H6 zero (control) | 0.982 | +0.001 [-0.010, +0.011] | 0.761 | +0.057 [+0.042, +0.074] | 0.995 | 0.00 / 0.23 |
| L16H11 zero (control) | 0.984 | +0.003 [-0.006, +0.010] | 0.709 | +0.003 [-0.011, +0.016] | 0.995 | 0.00 / 0.27 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.62 | 0.75 | 0.69 |
| L16H9 zero | 0.97 | 0.99 | 0.97 | 0.99 | 0.99 | 0.24 | 0.26 | 0.26 | 0.22 |
| L16H9 mean | 0.99 | 0.98 | 1.00 | 1.00 | 0.99 | 0.43 | 0.40 | 0.48 | 0.40 |
| L16H0 zero (control) | 0.98 | 1.00 | 0.99 | 0.98 | 0.99 | 0.72 | 0.57 | 0.75 | 0.66 |
| L16H6 zero (control) | 0.99 | 1.00 | 0.98 | 0.99 | 0.99 | 0.81 | 0.68 | 0.79 | 0.77 |
| L16H11 zero (control) | 0.98 | 1.00 | 0.98 | 0.99 | 0.99 | 0.77 | 0.62 | 0.75 | 0.68 |
