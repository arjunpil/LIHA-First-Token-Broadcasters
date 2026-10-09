# olmo3-7b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.972 |  | 0.874 |  | 1.000 | 0.01 / 0.09 |
| L14H25 zero | 0.936 | -0.041 [-0.060, -0.024] | 0.858 | -0.018 [-0.033, -0.003] | 1.000 | 0.04 / 0.11 |
| L14H25 mean | 0.973 | +0.001 [-0.011, +0.015] | 0.890 | +0.015 [+0.003, +0.028] | 0.995 | 0.01 / 0.08 |
| L14H12 zero (control) | 0.961 | -0.011 [-0.022, +0.000] | 0.880 | +0.007 [-0.003, +0.016] | 1.000 | 0.01 / 0.09 |
| L14H24 zero (control) | 0.977 | +0.004 [-0.005, +0.014] | 0.876 | +0.002 [-0.006, +0.009] | 1.000 | 0.01 / 0.10 |
| L14H28 zero (control) | 0.967 | -0.004 [-0.011, +0.004] | 0.875 | +0.001 [-0.009, +0.011] | 1.000 | 0.01 / 0.10 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.97 | 0.97 | 1.00 | 0.88 | 0.87 | 0.89 | 0.85 |
| L14H25 zero | 0.94 | 0.89 | 0.94 | 0.94 | 1.00 | 0.88 | 0.79 | 0.91 | 0.85 |
| L14H25 mean | 0.96 | 1.00 | 0.99 | 0.98 | 0.99 | 0.91 | 0.89 | 0.90 | 0.87 |
| L14H12 zero (control) | 0.96 | 0.97 | 0.96 | 0.97 | 1.00 | 0.88 | 0.89 | 0.88 | 0.87 |
| L14H24 zero (control) | 0.97 | 0.98 | 0.98 | 0.98 | 1.00 | 0.88 | 0.87 | 0.90 | 0.86 |
| L14H28 zero (control) | 0.97 | 0.98 | 0.97 | 0.97 | 1.00 | 0.88 | 0.87 | 0.89 | 0.86 |
