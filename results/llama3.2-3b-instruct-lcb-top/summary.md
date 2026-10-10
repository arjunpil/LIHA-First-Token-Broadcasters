# llama3.2-3b-instruct: Language Confusion Benchmark

Greedy 100 tokens, chat template. LPR = share of replies whose lines (5+ words) are all in the expected language, averaged over sources as in the benchmark. Δ = paired change against base on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, zero-ablated.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |
|---|---|---|---|---|---|---|
| base | 0.993 |  | 0.911 |  | 1.000 | 0.00 / 0.05 |
| L13H19 zero | 0.992 | +0.000 [-0.006, +0.006] | 0.572 | -0.335 [-0.363, -0.308] | 0.995 | 0.00 / 0.36 |
| L13H19 mean | 0.987 | -0.005 [-0.011, +0.001] | 0.658 | -0.250 [-0.277, -0.225] | 0.990 | 0.00 / 0.28 |
| L13H1 zero (control) | 0.993 | +0.000 [-0.004, +0.004] | 0.911 | +0.000 [-0.008, +0.008] | 0.990 | 0.00 / 0.05 |
| L13H12 zero (control) | 0.997 | +0.005 [+0.000, +0.011] | 0.909 | -0.003 [-0.011, +0.006] | 1.000 | 0.00 / 0.05 |
| L13H13 zero (control) | 0.999 | +0.006 [+0.001, +0.013] | 0.909 | -0.003 [-0.008, +0.003] | 1.000 | 0.00 / 0.05 |

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 0.92 | 0.92 | 0.91 | 0.90 |
| L13H19 zero | 1.00 | 1.00 | 0.98 | 1.00 | 0.99 | 0.58 | 0.53 | 0.58 | 0.60 |
| L13H19 mean | 0.99 | 0.99 | 0.98 | 1.00 | 0.99 | 0.65 | 0.65 | 0.66 | 0.67 |
| L13H1 zero (control) | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 | 0.92 | 0.92 | 0.92 | 0.89 |
| L13H12 zero (control) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | 0.91 | 0.92 | 0.90 |
| L13H13 zero (control) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.92 | 0.92 | 0.89 |
