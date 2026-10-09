# olmo2-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.986 |  | 0.931 |  | 0.990 | 0.00 / 0.05 |
| L12H8 zero | 0.740 | -0.272 [-0.302, -0.239] | 0.592 | -0.335 [-0.362, -0.308] | 0.990 | 0.16 / 0.28 |
| L12H8 mean | 0.875 | -0.128 [-0.152, -0.104] | 0.786 | -0.143 [-0.165, -0.123] | 1.000 | 0.03 / 0.08 |
| L0H10 zero | 0.965 | -0.021 [-0.036, -0.006] | 0.883 | -0.054 [-0.072, -0.037] | 1.000 | 0.03 / 0.08 |
| L0H10 mean | 0.962 | -0.025 [-0.039, -0.013] | 0.867 | -0.066 [-0.084, -0.047] | 0.995 | 0.03 / 0.09 |
| L0H6 zero (control) | 0.987 | +0.001 [-0.006, +0.009] | 0.932 | +0.001 [-0.011, +0.013] | 0.995 | 0.00 / 0.05 |
| L0H13 zero (control) | 0.983 | +0.003 [-0.008, +0.013] | 0.928 | -0.003 [-0.015, +0.009] | 1.000 | 0.00 / 0.05 |
| L0H14 zero (control) | 0.986 | +0.003 [-0.004, +0.009] | 0.928 | -0.003 [-0.014, +0.007] | 0.995 | 0.00 / 0.05 |
| L12H0 zero (control) | 0.989 | +0.004 [+0.000, +0.009] | 0.936 | +0.004 [-0.005, +0.014] | 0.995 | 0.00 / 0.05 |
| L12H6 zero (control) | 0.987 | +0.003 [+0.000, +0.006] | 0.929 | -0.003 [-0.010, +0.003] | 1.000 | 0.00 / 0.05 |
| L12H15 zero (control) | 0.982 | +0.000 [-0.006, +0.006] | 0.933 | +0.002 [-0.007, +0.010] | 0.995 | 0.01 / 0.05 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.95 | 0.92 | 0.94 | 0.92 |
| L12H8 zero | 0.81 | 0.47 | 0.77 | 0.49 | 0.99 | 0.64 | 0.57 | 0.64 | 0.52 |
| L12H8 mean | 0.94 | 0.60 | 0.92 | 0.65 | 1.00 | 0.87 | 0.80 | 0.79 | 0.68 |
| L0H10 zero | 0.96 | 0.93 | 0.98 | 0.94 | 1.00 | 0.90 | 0.88 | 0.90 | 0.85 |
| L0H10 mean | 0.97 | 0.91 | 0.98 | 0.94 | 0.99 | 0.90 | 0.89 | 0.89 | 0.78 |
| L0H6 zero (control) | 0.99 | 1.00 | 0.99 | 0.93 | 0.99 | 0.93 | 0.94 | 0.95 | 0.90 |
| L0H13 zero (control) | 0.98 | 1.00 | 0.99 | 0.97 | 1.00 | 0.91 | 0.94 | 0.94 | 0.93 |
| L0H14 zero (control) | 0.99 | 0.99 | 0.99 | 0.96 | 0.99 | 0.95 | 0.92 | 0.94 | 0.91 |
| L12H0 zero (control) | 0.99 | 0.98 | 0.99 | 0.97 | 0.99 | 0.94 | 0.93 | 0.96 | 0.92 |
| L12H6 zero (control) | 0.99 | 0.98 | 0.99 | 0.96 | 1.00 | 0.94 | 0.92 | 0.95 | 0.91 |
| L12H15 zero (control) | 0.98 | 0.98 | 0.99 | 0.96 | 0.99 | 0.95 | 0.92 | 0.95 | 0.91 |
