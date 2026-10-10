# llama3.2-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.985 |  | 0.778 |  | 1.000 | 0.00 / 0.16 |
| L8H25 zero | 0.902 | -0.069 [-0.081, -0.058] | 0.084 | -0.695 [-0.709, -0.681] | 1.000 | 0.02 / 0.89 |
| L8H25 mean | 0.872 | -0.101 [-0.115, -0.087] | 0.292 | -0.485 [-0.500, -0.470] | 0.990 | 0.02 / 0.64 |
| L8H12 zero (control) | 0.984 | -0.001 [-0.007, +0.005] | 0.764 | -0.015 [-0.024, -0.005] | 0.995 | 0.00 / 0.17 |
| L8H24 zero (control) | 0.986 | +0.000 [-0.005, +0.005] | 0.781 | +0.003 [-0.004, +0.011] | 1.000 | 0.00 / 0.16 |
| L8H28 zero (control) | 0.984 | -0.003 [-0.008, +0.003] | 0.774 | -0.006 [-0.014, +0.003] | 1.000 | 0.00 / 0.16 |

LPR per task and language

| condition | monolingual/ar | monolingual/de | monolingual/es | monolingual/fr | monolingual/hi | monolingual/id | monolingual/it | monolingual/ja | monolingual/ko | monolingual/pt | monolingual/ru | monolingual/tr | monolingual/vi | monolingual/zh | monolingual/en | crosslingual/ar | crosslingual/de | crosslingual/es | crosslingual/fr | crosslingual/hi | crosslingual/id | crosslingual/it | crosslingual/ja | crosslingual/ko | crosslingual/pt | crosslingual/ru | crosslingual/tr | crosslingual/vi | crosslingual/zh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 0.91 | 1.00 | 1.00 | 0.99 | 0.97 | 0.98 | 0.95 | 1.00 | 0.97 | 1.00 | 0.68 | 0.88 | 0.88 | 0.87 | 0.86 | 0.80 | 0.87 | 0.67 | 0.58 | 0.82 | 0.74 | 0.73 | 0.86 | 0.64 |
| L8H25 zero | 0.89 | 0.97 | 1.00 | 0.99 | 1.00 | 0.93 | 0.91 | 0.82 | 0.60 | 0.95 | 0.96 | 0.64 | 0.99 | 0.88 | 1.00 | 0.03 | 0.15 | 0.22 | 0.15 | 0.15 | 0.13 | 0.13 | 0.00 | 0.00 | 0.09 | 0.01 | 0.00 | 0.10 | 0.00 |
| L8H25 mean | 0.80 | 0.97 | 0.99 | 0.99 | 1.00 | 0.85 | 0.93 | 0.85 | 0.38 | 0.94 | 0.91 | 0.63 | 0.97 | 0.84 | 0.99 | 0.09 | 0.52 | 0.55 | 0.49 | 0.35 | 0.48 | 0.49 | 0.01 | 0.01 | 0.44 | 0.14 | 0.06 | 0.44 | 0.03 |
| L8H12 zero (control) | 0.99 | 1.00 | 0.99 | 0.98 | 1.00 | 0.94 | 0.99 | 1.00 | 1.00 | 0.96 | 0.99 | 0.96 | 1.00 | 0.97 | 0.99 | 0.65 | 0.86 | 0.88 | 0.88 | 0.84 | 0.80 | 0.88 | 0.62 | 0.53 | 0.82 | 0.71 | 0.77 | 0.87 | 0.57 |
| L8H24 zero (control) | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.94 | 1.00 | 0.99 | 1.00 | 0.96 | 0.98 | 0.98 | 1.00 | 0.96 | 1.00 | 0.71 | 0.88 | 0.87 | 0.88 | 0.86 | 0.82 | 0.86 | 0.69 | 0.58 | 0.79 | 0.75 | 0.75 | 0.87 | 0.63 |
| L8H28 zero (control) | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 0.99 | 0.97 | 1.00 | 0.98 | 1.00 | 0.94 | 1.00 | 0.68 | 0.86 | 0.86 | 0.87 | 0.85 | 0.79 | 0.87 | 0.68 | 0.57 | 0.83 | 0.74 | 0.75 | 0.85 | 0.62 |
