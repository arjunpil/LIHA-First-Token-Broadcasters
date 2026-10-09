# llama3.2-1b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.997 |  | 0.874 |  | 1.000 | 0.00 / 0.09 |
| L8H25 zero | 0.983 | -0.016 [-0.026, -0.008] | 0.164 | -0.711 [-0.737, -0.684] | 1.000 | 0.00 / 0.81 |
| L8H25 mean | 0.983 | -0.015 [-0.025, -0.005] | 0.515 | -0.357 [-0.384, -0.329] | 0.990 | 0.00 / 0.44 |
| L0H1 zero | 0.992 | -0.004 [-0.009, +0.000] | 0.845 | -0.027 [-0.040, -0.015] | 0.995 | 0.00 / 0.11 |
| L0H1 mean | 0.990 | -0.005 [-0.011, -0.001] | 0.846 | -0.026 [-0.039, -0.013] | 0.995 | 0.00 / 0.11 |
| L0H28 zero (control) | 0.991 | -0.004 [-0.010, +0.001] | 0.880 | +0.004 [-0.007, +0.015] | 0.995 | 0.00 / 0.08 |
| L8H12 zero (control) | 0.987 | -0.009 [-0.019, +0.000] | 0.875 | +0.001 [-0.013, +0.016] | 0.995 | 0.00 / 0.08 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.87 | 0.88 | 0.88 | 0.87 |
| L8H25 zero | 0.99 | 0.97 | 1.00 | 0.91 | 1.00 | 0.15 | 0.15 | 0.22 | 0.13 |
| L8H25 mean | 0.99 | 0.97 | 0.99 | 0.93 | 0.99 | 0.49 | 0.52 | 0.55 | 0.49 |
| L0H1 zero | 1.00 | 1.00 | 0.98 | 1.00 | 0.99 | 0.86 | 0.87 | 0.83 | 0.83 |
| L0H1 mean | 1.00 | 1.00 | 0.98 | 1.00 | 0.99 | 0.85 | 0.86 | 0.83 | 0.83 |
| L0H28 zero (control) | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 0.88 | 0.89 | 0.89 | 0.87 |
| L8H12 zero (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.99 | 0.88 | 0.86 | 0.88 | 0.88 |
