# gemma3-4b-instruct L24H0: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.990 |  | 0.133 |  | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.004 | 0.000 / 0.007 |
| L24H0 steer | 0.982 | -0.007 [-0.016, +0.001] | 0.213 | +0.079 [+0.064, +0.095] | 0.000 / 0.000 | 0.00 / 0.39 | 0.985 / 0.000 | 0.003 | 0.000 / 0.009 |
| L24H0 swap | 0.003 | -0.989 [-0.995, -0.981] | 0.000 | -0.131 [-0.150, -0.112] | 0.985 / 0.137 | 0.00 / 0.43 | 0.995 / 0.000 | 0.006 | 0.003 / 0.008 |
| L24H0 add steer | 0.993 | +0.003 [-0.004, +0.009] | 0.173 | +0.040 [+0.029, +0.053] | 0.000 / 0.000 | 0.00 / 0.41 | 0.985 / 0.000 | 0.003 | 0.001 / 0.008 |
| L24H0 add swap | 0.803 | -0.180 [-0.208, -0.153] | 0.089 | -0.043 [-0.059, -0.027] | 0.045 / 0.023 | 0.01 / 0.45 | 0.990 / 0.000 | 0.004 | 0.001 / 0.006 |
| L24H4 steer (control) | 0.991 | +0.001 [-0.006, +0.009] | 0.119 | -0.014 [-0.023, -0.006] | 0.000 / 0.000 | 0.00 / 0.44 | 0.990 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H4 swap (control) | 0.989 | +0.000 [-0.007, +0.007] | 0.119 | -0.014 [-0.023, -0.006] | 0.000 / 0.000 | 0.00 / 0.44 | 0.990 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H4 add steer (control) | 0.993 | +0.003 [+0.000, +0.006] | 0.133 | +0.000 [+0.000, +0.000] | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.004 | 0.000 / 0.007 |
| L24H4 add swap (control) | 0.992 | +0.001 [+0.000, +0.004] | 0.132 | -0.002 [-0.004, +0.000] | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H6 steer (control) | 0.989 | -0.001 [-0.009, +0.006] | 0.142 | +0.008 [-0.003, +0.020] | 0.000 / 0.000 | 0.00 / 0.43 | 0.980 / 0.000 | 0.003 | 0.001 / 0.007 |
| L24H6 swap (control) | 0.987 | -0.003 [-0.011, +0.006] | 0.141 | +0.008 [-0.003, +0.019] | 0.000 / 0.000 | 0.00 / 0.43 | 0.980 / 0.000 | 0.003 | 0.001 / 0.007 |
| L24H6 add steer (control) | 0.990 | +0.000 [+0.000, +0.000] | 0.137 | +0.004 [+0.000, +0.009] | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H6 add swap (control) | 0.990 | +0.000 [-0.004, +0.004] | 0.137 | +0.004 [+0.000, +0.009] | 0.000 / 0.000 | 0.00 / 0.43 | 0.985 / 0.000 | 0.004 | 0.000 / 0.007 |
| L24H7 steer (control) | 0.986 | -0.004 [-0.011, +0.003] | 0.110 | -0.023 [-0.032, -0.013] | 0.000 / 0.000 | 0.00 / 0.45 | 0.990 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H7 swap (control) | 0.986 | -0.004 [-0.011, +0.003] | 0.114 | -0.019 [-0.028, -0.011] | 0.000 / 0.000 | 0.00 / 0.45 | 0.990 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H7 add steer (control) | 0.993 | +0.003 [+0.000, +0.006] | 0.132 | -0.002 [-0.004, +0.000] | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.003 | 0.000 / 0.007 |
| L24H7 add swap (control) | 0.992 | +0.001 [+0.000, +0.004] | 0.131 | -0.003 [-0.006, +0.000] | 0.000 / 0.000 | 0.00 / 0.44 | 0.985 / 0.000 | 0.003 | 0.000 / 0.007 |

Baseline of the earlier LCB run: mono 0.990, cross 0.133.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.13 | 0.11 | 0.15 |
| L24H0 steer | 0.97 | 1.00 | 0.99 | 0.99 | 0.98 | 0.22 | 0.23 | 0.18 | 0.23 |
| L24H0 swap | 0.01 | 0.00 | 0.00 | 0.00 | 0.99 | 0.00 | 0.00 | 0.00 | 0.00 |
| L24H0 add steer | 0.99 | 1.00 | 1.00 | 0.99 | 0.98 | 0.19 | 0.19 | 0.12 | 0.19 |
| L24H0 add swap | 0.77 | 1.00 | 0.82 | 0.74 | 0.99 | 0.06 | 0.19 | 0.03 | 0.07 |
| L24H4 steer (control) | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 0.13 | 0.13 | 0.08 | 0.13 |
| L24H4 swap (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.13 | 0.13 | 0.08 | 0.13 |
| L24H4 add steer (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.13 | 0.11 | 0.15 |
| L24H4 add swap (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.13 | 0.11 | 0.14 |
| L24H6 steer (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.98 | 0.17 | 0.14 | 0.09 | 0.17 |
| L24H6 swap (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.98 | 0.17 | 0.14 | 0.09 | 0.17 |
| L24H6 add steer (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.14 | 0.11 | 0.15 |
| L24H6 add swap (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.15 | 0.14 | 0.11 | 0.15 |
| L24H7 steer (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.99 | 0.12 | 0.10 | 0.09 | 0.13 |
| L24H7 swap (control) | 0.98 | 1.00 | 0.99 | 0.99 | 0.99 | 0.13 | 0.10 | 0.09 | 0.13 |
| L24H7 add steer (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.14 | 0.13 | 0.11 | 0.15 |
| L24H7 add swap (control) | 0.99 | 1.00 | 0.99 | 0.99 | 0.98 | 0.14 | 0.13 | 0.11 | 0.15 |
