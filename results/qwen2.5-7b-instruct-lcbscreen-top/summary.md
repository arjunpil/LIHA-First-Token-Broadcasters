# qwen2.5-7b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.950 |  | 1.000 | 0.00 / 0.04 |
| L0H1 zero | 0.849 | -0.116 [-0.142, -0.088] | 0.674 | -0.271 [-0.300, -0.245] | 0.989 | 0.10 / 0.29 |
| L0H1 mean | 0.987 | +0.004 [-0.005, +0.013] | 0.944 | -0.005 [-0.013, +0.003] | 1.000 | 0.00 / 0.04 |
| L0H13 zero (control) | 0.982 | -0.001 [-0.008, +0.004] | 0.951 | +0.001 [-0.006, +0.008] | 1.000 | 0.00 / 0.04 |
| L0H14 zero (control) | 0.986 | +0.001 [-0.004, +0.008] | 0.947 | -0.003 [-0.010, +0.004] | 0.995 | 0.00 / 0.04 |
| L0H25 zero (control) | 0.500 | -0.495 [-0.535, -0.455] | 0.465 | -0.485 [-0.518, -0.453] | 0.960 | 0.39 / 0.44 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
| L0H1 zero | 0.83 | 0.86 | 0.89 | 0.90 | 0.99 | 0.64 | 0.64 | 0.78 | 0.64 |
| L0H1 mean | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.94 | 0.95 | 0.96 | 0.93 |
| L0H13 zero (control) | 0.98 | 0.99 | 0.98 | 0.99 | 1.00 | 0.96 | 0.95 | 0.96 | 0.93 |
| L0H14 zero (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.99 | 0.95 | 0.95 | 0.96 | 0.93 |
| L0H25 zero (control) | 0.55 | 0.66 | 0.45 | 0.29 | 0.96 | 0.44 | 0.46 | 0.49 | 0.46 |
