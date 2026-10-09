# gemma3-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.118 |  | 0.985 | 0.00 / 0.55 |
| L11H3 zero | 0.383 | -0.628 [-0.662, -0.593] | 0.021 | -0.091 [-0.110, -0.073] | 1.000 | 0.34 / 0.89 |
| L11H3 mean | 0.775 | -0.244 [-0.275, -0.213] | 0.026 | -0.092 [-0.110, -0.075] | 0.980 | 0.01 / 0.76 |
| L5H0 zero | 0.984 | +0.000 [-0.011, +0.011] | 0.132 | +0.012 [-0.003, +0.026] | 0.985 | 0.00 / 0.54 |
| L5H0 mean | 0.980 | -0.005 [-0.015, +0.004] | 0.118 | -0.001 [-0.013, +0.010] | 0.990 | 0.00 / 0.57 |
| L5H1 zero (control) | 0.984 | +0.003 [-0.006, +0.011] | 0.121 | +0.003 [-0.008, +0.015] | 0.995 | 0.00 / 0.56 |
| L5H2 zero (control) | 0.984 | +0.001 [-0.006, +0.009] | 0.140 | +0.021 [+0.010, +0.033] | 0.980 | 0.00 / 0.54 |
| L5H3 zero (control) | 0.986 | +0.001 [-0.009, +0.010] | 0.104 | -0.014 [-0.025, -0.003] | 0.990 | 0.00 / 0.55 |
| L11H0 zero (control) | 0.982 | -0.003 [-0.010, +0.005] | 0.158 | +0.038 [+0.026, +0.052] | 0.995 | 0.00 / 0.52 |
| L11H1 zero (control) | 0.987 | +0.003 [-0.006, +0.011] | 0.154 | +0.038 [+0.025, +0.051] | 1.000 | 0.00 / 0.52 |
| L11H2 zero (control) | 0.982 | -0.005 [-0.015, +0.005] | 0.095 | -0.023 [-0.038, -0.009] | 0.995 | 0.00 / 0.60 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.97 | 0.99 | 0.99 | 1.00 | 0.98 | 0.16 | 0.08 | 0.11 | 0.14 |
| L11H3 zero | 0.35 | 0.13 | 0.43 | 0.36 | 1.00 | 0.05 | 0.00 | 0.00 | 0.03 |
| L11H3 mean | 0.59 | 0.23 | 0.99 | 0.97 | 0.98 | 0.05 | 0.01 | 0.04 | 0.01 |
| L5H0 zero | 0.97 | 1.00 | 0.99 | 0.99 | 0.98 | 0.17 | 0.09 | 0.13 | 0.14 |
| L5H0 mean | 0.96 | 1.00 | 0.99 | 1.00 | 0.99 | 0.16 | 0.07 | 0.12 | 0.13 |
| L5H1 zero (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.99 | 0.16 | 0.09 | 0.10 | 0.13 |
| L5H2 zero (control) | 0.97 | 1.00 | 0.99 | 1.00 | 0.98 | 0.17 | 0.12 | 0.13 | 0.14 |
| L5H3 zero (control) | 0.98 | 0.99 | 0.98 | 1.00 | 0.99 | 0.13 | 0.06 | 0.11 | 0.12 |
| L11H0 zero (control) | 0.96 | 1.00 | 0.99 | 1.00 | 0.99 | 0.18 | 0.13 | 0.15 | 0.18 |
| L11H1 zero (control) | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 | 0.18 | 0.11 | 0.15 | 0.17 |
| L11H2 zero (control) | 0.96 | 1.00 | 0.99 | 0.99 | 0.99 | 0.10 | 0.10 | 0.09 | 0.10 |
