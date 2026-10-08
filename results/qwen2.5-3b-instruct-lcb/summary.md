# qwen2.5-3b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.982 |  | 0.888 |  | 1.000 | 0.00 / 0.09 |
| L27H13 zero | 0.467 | -0.537 [-0.575, -0.503] | 0.435 | -0.449 [-0.478, -0.421] | 1.000 | 0.07 / 0.35 |
| L27H13 mean | 0.612 | -0.409 [-0.444, -0.373] | 0.556 | -0.331 [-0.358, -0.303] | 0.995 | 0.01 / 0.22 |
| L27H6 zero (control) | 0.983 | +0.001 [-0.008, +0.011] | 0.891 | +0.003 [-0.006, +0.013] | 0.995 | 0.00 / 0.08 |
| L27H12 zero (control) | 0.981 | +0.000 [-0.009, +0.009] | 0.894 | +0.004 [-0.004, +0.013] | 1.000 | 0.00 / 0.08 |
| L27H14 zero (control) | 0.983 | +0.003 [-0.006, +0.011] | 0.862 | -0.025 [-0.038, -0.013] | 0.995 | 0.00 / 0.11 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 0.91 | 0.87 | 0.92 | 0.86 |
| L27H13 zero | 0.20 | 0.48 | 0.83 | 0.00 | 1.00 | 0.52 | 0.56 | 0.63 | 0.03 |
| L27H13 mean | 0.52 | 0.31 | 0.90 | 0.00 | 0.99 | 0.73 | 0.65 | 0.77 | 0.06 |
| L27H6 zero (control) | 0.98 | 0.99 | 0.98 | 0.98 | 0.99 | 0.92 | 0.87 | 0.92 | 0.86 |
| L27H12 zero (control) | 0.99 | 0.99 | 0.97 | 0.98 | 1.00 | 0.91 | 0.87 | 0.93 | 0.86 |
| L27H14 zero (control) | 0.99 | 1.00 | 0.98 | 0.97 | 0.99 | 0.89 | 0.82 | 0.90 | 0.83 |
