# qwen2.5-7b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.950 |  | 1.000 | 0.00 / 0.04 |
| L19H1 zero | 0.972 | -0.009 [-0.020, +0.003] | 0.875 | -0.074 [-0.089, -0.059] | 0.995 | 0.00 / 0.10 |
| L19H1 mean | 0.986 | +0.003 [-0.005, +0.010] | 0.915 | -0.035 [-0.047, -0.024] | 1.000 | 0.00 / 0.06 |
| L19H13 zero (control) | 0.987 | +0.003 [-0.003, +0.009] | 0.950 | +0.000 [-0.007, +0.007] | 1.000 | 0.00 / 0.04 |
| L19H14 zero (control) | 0.986 | +0.003 [-0.004, +0.010] | 0.942 | -0.008 [-0.016, -0.001] | 0.995 | 0.00 / 0.04 |
| L19H25 zero (control) | 0.985 | +0.001 [-0.004, +0.008] | 0.950 | +0.001 [-0.006, +0.008] | 1.000 | 0.00 / 0.04 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
| L19H1 zero | 0.96 | 0.98 | 0.98 | 0.99 | 0.99 | 0.89 | 0.87 | 0.89 | 0.86 |
| L19H1 mean | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.92 | 0.91 | 0.92 | 0.90 |
| L19H13 zero (control) | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
| L19H14 zero (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.99 | 0.93 | 0.94 | 0.96 | 0.93 |
| L19H25 zero (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.96 | 0.96 | 0.93 |
