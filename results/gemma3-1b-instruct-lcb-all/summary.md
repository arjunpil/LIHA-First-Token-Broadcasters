# gemma3-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.978 |  | 0.129 |  | 0.985 | 0.00 / 0.56 |
| L11H3 zero | 0.266 | -0.730 [-0.748, -0.710] | 0.013 | -0.113 [-0.123, -0.102] | 1.000 | 0.26 / 0.91 |
| L11H3 mean | 0.515 | -0.488 [-0.510, -0.466] | 0.011 | -0.117 [-0.127, -0.107] | 0.980 | 0.04 / 0.85 |
| L11H0 zero (control) | 0.977 | +0.000 [-0.006, +0.006] | 0.159 | +0.029 [+0.021, +0.036] | 0.995 | 0.00 / 0.54 |
| L11H1 zero (control) | 0.978 | +0.001 [-0.005, +0.007] | 0.177 | +0.050 [+0.041, +0.058] | 1.000 | 0.00 / 0.52 |
| L11H2 zero (control) | 0.981 | +0.001 [-0.005, +0.008] | 0.114 | -0.014 [-0.022, -0.006] | 0.995 | 0.00 / 0.61 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.00 | 0.99 | 0.99 | 0.97 | 1.00 | 0.89 | 1.00 | 0.97 | 1.00 | 0.96 | 1.00 | 1.00 | 0.99 | 0.93 | 0.98 | 0.07 | 0.08 | 0.11 | 0.16 | 0.12 | 0.22 | 0.14 | 0.14 | 0.16 | 0.13 | 0.09 | 0.27 | 0.07 | 0.07 |
| L11H3 zero | 0.13 | 0.13 | 0.43 | 0.35 | 0.08 | 0.00 | 0.36 | 0.19 | 0.26 | 0.45 | 0.45 | 0.17 | 0.01 | 0.08 | 1.00 | 0.02 | 0.00 | 0.00 | 0.05 | 0.00 | 0.01 | 0.03 | 0.02 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.01 |
| L11H3 mean | 0.08 | 0.23 | 0.99 | 0.59 | 0.21 | 0.00 | 0.97 | 0.67 | 0.67 | 0.95 | 0.20 | 0.16 | 0.00 | 0.41 | 0.98 | 0.00 | 0.01 | 0.04 | 0.05 | 0.01 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| L11H0 zero (control) | 1.00 | 1.00 | 0.99 | 0.96 | 0.99 | 0.92 | 1.00 | 0.98 | 1.00 | 0.97 | 1.00 | 0.99 | 0.99 | 0.92 | 0.99 | 0.11 | 0.13 | 0.15 | 0.18 | 0.13 | 0.27 | 0.18 | 0.17 | 0.15 | 0.14 | 0.12 | 0.33 | 0.08 | 0.09 |
| L11H1 zero (control) | 1.00 | 1.00 | 0.99 | 0.98 | 1.00 | 0.87 | 1.00 | 0.99 | 1.00 | 0.96 | 1.00 | 0.99 | 1.00 | 0.93 | 1.00 | 0.10 | 0.11 | 0.15 | 0.18 | 0.21 | 0.25 | 0.17 | 0.21 | 0.23 | 0.17 | 0.13 | 0.36 | 0.08 | 0.13 |
| L11H2 zero (control) | 1.00 | 1.00 | 0.99 | 0.96 | 0.99 | 0.93 | 0.99 | 0.98 | 0.99 | 0.97 | 1.00 | 0.99 | 0.99 | 0.94 | 0.99 | 0.05 | 0.10 | 0.09 | 0.10 | 0.10 | 0.21 | 0.10 | 0.14 | 0.13 | 0.11 | 0.08 | 0.31 | 0.05 | 0.05 |
