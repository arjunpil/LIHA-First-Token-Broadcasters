# qwen3-1.7b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.985 |  | 0.823 |  | 0.985 | 0.00 / 0.14 |
| L18H12 zero | 0.842 | -0.149 [-0.174, -0.123] | 0.315 | -0.508 [-0.535, -0.478] | 0.990 | 0.05 / 0.64 |
| L18H12 mean | 0.924 | -0.067 [-0.086, -0.048] | 0.593 | -0.230 [-0.257, -0.205] | 0.995 | 0.00 / 0.36 |
| L18H6 zero (control) | 0.983 | -0.003 [-0.013, +0.006] | 0.841 | +0.018 [+0.007, +0.028] | 0.985 | 0.00 / 0.14 |
| L18H13 zero (control) | 0.981 | -0.003 [-0.010, +0.005] | 0.824 | +0.000 [-0.011, +0.011] | 0.980 | 0.00 / 0.14 |
| L18H14 zero (control) | 0.979 | -0.004 [-0.011, +0.005] | 0.844 | +0.021 [+0.008, +0.035] | 0.975 | 0.00 / 0.12 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.78 | 0.85 | 0.81 |
| L18H12 zero | 0.87 | 0.90 | 0.86 | 0.62 | 0.99 | 0.40 | 0.28 | 0.29 | 0.29 |
| L18H12 mean | 0.96 | 0.99 | 0.92 | 0.74 | 0.99 | 0.65 | 0.56 | 0.60 | 0.56 |
| L18H6 zero (control) | 0.99 | 1.00 | 0.98 | 0.99 | 0.98 | 0.86 | 0.78 | 0.88 | 0.84 |
| L18H13 zero (control) | 0.98 | 1.00 | 0.98 | 1.00 | 0.98 | 0.84 | 0.77 | 0.87 | 0.81 |
| L18H14 zero (control) | 0.99 | 1.00 | 0.97 | 1.00 | 0.97 | 0.85 | 0.80 | 0.88 | 0.85 |
