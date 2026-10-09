# olmo2-1b-dpo-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.989 |  | 0.933 |  | 0.995 | 0.00 / 0.05 |
| L12H8 zero | 0.821 | -0.199 [-0.230, -0.172] | 0.687 | -0.244 [-0.268, -0.219] | 1.000 | 0.10 / 0.20 |
| L12H8 mean | 0.899 | -0.120 [-0.146, -0.098] | 0.809 | -0.123 [-0.145, -0.103] | 1.000 | 0.02 / 0.07 |
| L12H6 zero (control) | 0.996 | +0.005 [+0.001, +0.010] | 0.932 | -0.002 [-0.009, +0.005] | 0.995 | 0.00 / 0.05 |
| L12H13 zero (control) | 0.988 | -0.001 [-0.008, +0.005] | 0.937 | +0.005 [-0.003, +0.012] | 1.000 | 0.00 / 0.04 |
| L12H14 zero (control) | 0.992 | +0.003 [-0.005, +0.009] | 0.930 | -0.005 [-0.016, +0.006] | 0.990 | 0.00 / 0.05 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 1.00 | 0.97 | 0.99 | 0.93 | 0.93 | 0.94 | 0.93 |
| L12H8 zero | 0.87 | 0.53 | 0.89 | 0.52 | 1.00 | 0.76 | 0.64 | 0.75 | 0.61 |
| L12H8 mean | 0.96 | 0.58 | 0.95 | 0.64 | 1.00 | 0.91 | 0.81 | 0.81 | 0.70 |
| L12H6 zero (control) | 1.00 | 1.00 | 1.00 | 0.97 | 0.99 | 0.93 | 0.93 | 0.94 | 0.93 |
| L12H13 zero (control) | 0.99 | 1.00 | 1.00 | 0.96 | 1.00 | 0.93 | 0.94 | 0.94 | 0.94 |
| L12H14 zero (control) | 1.00 | 1.00 | 0.99 | 0.98 | 0.99 | 0.93 | 0.93 | 0.92 | 0.94 |
