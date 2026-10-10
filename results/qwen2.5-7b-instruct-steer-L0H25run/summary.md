# qwen2.5-7b-instruct L0H25: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.950 |  | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H25 steer | 0.471 | -0.529 [-0.568, -0.488] | 0.434 | -0.514 [-0.546, -0.481] | 0.002 / 0.000 | 0.42 / 0.46 | 0.919 / 0.000 | 0.068 | 0.260 / 0.168 |
| L0H25 swap | 0.358 | -0.625 [-0.666, -0.586] | 0.368 | -0.575 [-0.605, -0.544] | 0.002 / 0.000 | 0.53 / 0.53 | 0.924 / 0.000 | 0.081 | 0.294 / 0.186 |
| L0H25 add steer | 0.985 | +0.001 [-0.004, +0.008] | 0.949 | -0.001 [-0.006, +0.004] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H25 add swap | 0.988 | +0.005 [-0.001, +0.011] | 0.952 | +0.002 [-0.004, +0.008] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.000 / 0.013 |
| L0H12 steer (control) | 0.983 | -0.001 [-0.008, +0.004] | 0.946 | -0.003 [-0.010, +0.005] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.003 | 0.001 / 0.010 |
| L0H12 swap (control) | 0.983 | -0.001 [-0.006, +0.003] | 0.947 | -0.002 [-0.008, +0.005] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.011 |
| L0H12 add steer (control) | 0.983 | +0.000 [-0.005, +0.005] | 0.952 | +0.002 [-0.003, +0.007] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.002 | 0.001 / 0.013 |
| L0H12 add swap (control) | 0.982 | -0.003 [-0.006, +0.000] | 0.953 | +0.003 [-0.002, +0.008] | 0.000 / 0.000 | 0.00 / 0.03 | 1.000 / 0.000 | 0.003 | 0.001 / 0.014 |
| L0H13 steer (control) | 0.982 | -0.001 [-0.008, +0.005] | 0.949 | -0.001 [-0.008, +0.006] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H13 swap (control) | 0.976 | -0.006 [-0.014, +0.001] | 0.945 | -0.003 [-0.010, +0.003] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H13 add steer (control) | 0.985 | +0.001 [-0.003, +0.006] | 0.951 | +0.002 [-0.003, +0.007] | 0.000 / 0.000 | 0.00 / 0.03 | 0.995 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H13 add swap (control) | 0.983 | +0.000 [-0.006, +0.006] | 0.952 | +0.003 [-0.002, +0.008] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.003 | 0.001 / 0.013 |
| L0H24 steer (control) | 0.951 | -0.049 [-0.068, -0.030] | 0.734 | -0.206 [-0.231, -0.180] | 0.000 / 0.000 | 0.04 / 0.21 | 0.990 / 0.000 | 0.047 | 0.006 / 0.038 |
| L0H24 swap (control) | 0.689 | -0.270 [-0.305, -0.237] | 0.632 | -0.306 [-0.334, -0.276] | 0.000 / 0.000 | 0.23 / 0.30 | 0.995 / 0.000 | 0.125 | 0.060 / 0.066 |
| L0H24 add steer (control) | 0.983 | -0.001 [-0.008, +0.005] | 0.951 | +0.001 [-0.005, +0.007] | 0.000 / 0.000 | 0.00 / 0.03 | 1.000 / 0.000 | 0.002 | 0.001 / 0.015 |
| L0H24 add swap (control) | 0.987 | +0.003 [-0.005, +0.010] | 0.952 | +0.002 [-0.005, +0.008] | 0.000 / 0.000 | 0.00 / 0.03 | 1.000 / 0.000 | 0.003 | 0.001 / 0.014 |

Baseline of the earlier LCB run: mono 0.984, cross 0.950.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
| L0H25 steer | 0.51 | 0.39 | 0.48 | 0.28 | 0.92 | 0.47 | 0.35 | 0.50 | 0.40 |
| L0H25 swap | 0.37 | 0.60 | 0.33 | 0.18 | 0.92 | 0.36 | 0.43 | 0.37 | 0.30 |
| L0H25 add steer | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.96 | 0.95 | 0.95 | 0.94 |
| L0H25 add swap | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 0.96 | 0.96 | 0.95 | 0.94 |
| L0H12 steer (control) | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.95 | 0.95 | 0.96 | 0.93 |
| L0H12 swap (control) | 0.98 | 0.99 | 0.98 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.93 |
| L0H12 add steer (control) | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 0.96 | 0.95 | 0.96 | 0.94 |
| L0H12 add swap (control) | 0.98 | 0.99 | 0.98 | 0.99 | 1.00 | 0.96 | 0.95 | 0.96 | 0.94 |
| L0H13 steer (control) | 0.98 | 0.99 | 0.98 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.93 |
| L0H13 swap (control) | 0.97 | 0.99 | 0.98 | 0.99 | 1.00 | 0.94 | 0.95 | 0.96 | 0.93 |
| L0H13 add steer (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.99 | 0.96 | 0.95 | 0.96 | 0.94 |
| L0H13 add swap (control) | 0.98 | 0.99 | 0.98 | 1.00 | 0.99 | 0.96 | 0.96 | 0.96 | 0.93 |
| L0H24 steer (control) | 0.98 | 0.80 | 0.99 | 0.79 | 0.99 | 0.88 | 0.50 | 0.92 | 0.63 |
| L0H24 swap (control) | 0.76 | 0.99 | 0.60 | 0.62 | 0.99 | 0.56 | 0.86 | 0.60 | 0.50 |
| L0H24 add steer (control) | 0.98 | 0.99 | 0.98 | 0.98 | 1.00 | 0.96 | 0.95 | 0.96 | 0.93 |
| L0H24 add swap (control) | 0.98 | 0.99 | 0.99 | 0.98 | 1.00 | 0.96 | 0.95 | 0.96 | 0.94 |
