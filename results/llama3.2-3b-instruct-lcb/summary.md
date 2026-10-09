# llama3.2-3b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.993 |  | 0.911 |  | 1.000 | 0.00 / 0.05 |
| L0H2 zero | 0.993 | +0.001 [-0.006, +0.009] | 0.909 | -0.003 [-0.013, +0.008] | 0.995 | 0.00 / 0.05 |
| L0H2 mean | 0.991 | +0.000 [-0.007, +0.007] | 0.911 | +0.000 [-0.010, +0.010] | 0.995 | 0.00 / 0.05 |
| L2H17 zero | 0.997 | +0.004 [-0.001, +0.010] | 0.905 | -0.006 [-0.014, +0.002] | 0.995 | 0.00 / 0.05 |
| L2H17 mean | 0.987 | -0.004 [-0.010, +0.001] | 0.908 | -0.003 [-0.010, +0.003] | 0.995 | 0.00 / 0.05 |
| L0H13 zero (control) | 0.994 | +0.003 [-0.003, +0.007] | 0.912 | +0.001 [-0.004, +0.007] | 0.995 | 0.00 / 0.05 |
| L2H13 zero (control) | 0.997 | +0.004 [+0.000, +0.009] | 0.911 | -0.001 [-0.007, +0.005] | 1.000 | 0.00 / 0.05 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 0.92 | 0.92 | 0.91 | 0.90 |
| L0H2 zero | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 0.91 | 0.90 | 0.92 | 0.91 |
| L0H2 mean | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 0.91 | 0.92 | 0.91 | 0.91 |
| L2H17 zero | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 | 0.91 | 0.90 | 0.90 | 0.90 |
| L2H17 mean | 0.99 | 1.00 | 0.98 | 0.99 | 0.99 | 0.91 | 0.91 | 0.91 | 0.90 |
| L0H13 zero (control) | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.92 | 0.92 | 0.91 | 0.90 |
| L2H13 zero (control) | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 0.91 | 0.91 | 0.90 |
