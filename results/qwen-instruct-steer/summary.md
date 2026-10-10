# qwen-instruct L22H6: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.982 |  | 0.704 |  | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.009 |
| L22H6 steer | 0.977 | -0.001 [-0.010, +0.008] | 0.684 | -0.020 [-0.033, -0.008] | 0.000 / 0.000 | 0.01 / 0.30 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |
| L22H6 swap | 0.699 | -0.335 [-0.369, -0.303] | 0.384 | -0.317 [-0.343, -0.289] | 0.043 / 0.019 | 0.04 / 0.43 | 0.995 / 0.000 | 0.002 | 0.005 / 0.014 |
| L22H6 add steer | 0.981 | -0.001 [-0.010, +0.009] | 0.661 | -0.042 [-0.057, -0.026] | 0.000 / 0.000 | 0.01 / 0.32 | 0.995 / 0.000 | 0.002 | 0.004 / 0.011 |
| L22H6 add swap | 0.984 | +0.003 [-0.005, +0.010] | 0.691 | -0.013 [-0.024, -0.002] | 0.000 / 0.000 | 0.00 / 0.29 | 0.985 / 0.000 | 0.002 | 0.003 / 0.008 |
| L22H4 steer (control) | 0.987 | +0.005 [-0.001, +0.013] | 0.710 | +0.007 [-0.002, +0.015] | 0.000 / 0.000 | 0.00 / 0.27 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |
| L22H4 swap (control) | 0.987 | +0.005 [-0.001, +0.013] | 0.702 | -0.001 [-0.010, +0.008] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.009 |
| L22H4 add steer (control) | 0.981 | +0.000 [-0.004, +0.004] | 0.706 | +0.003 [+0.000, +0.006] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.009 |
| L22H4 add swap (control) | 0.980 | -0.001 [-0.004, +0.000] | 0.700 | -0.004 [-0.008, -0.001] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |
| L22H8 steer (control) | 0.985 | +0.004 [-0.003, +0.011] | 0.698 | -0.006 [-0.013, +0.001] | 0.000 / 0.000 | 0.00 / 0.27 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |
| L22H8 swap (control) | 0.984 | +0.003 [-0.003, +0.009] | 0.695 | -0.008 [-0.016, -0.002] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.011 |
| L22H8 add steer (control) | 0.979 | -0.001 [-0.006, +0.003] | 0.703 | -0.001 [-0.003, +0.002] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.009 |
| L22H8 add swap (control) | 0.982 | +0.000 [+0.000, +0.000] | 0.702 | -0.002 [-0.006, +0.003] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.003 / 0.009 |
| L22H9 steer (control) | 0.976 | -0.004 [-0.009, +0.001] | 0.706 | +0.003 [-0.004, +0.009] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |
| L22H9 swap (control) | 0.979 | -0.001 [-0.008, +0.005] | 0.706 | +0.002 [-0.005, +0.009] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.011 |
| L22H9 add steer (control) | 0.982 | +0.000 [+0.000, +0.000] | 0.701 | -0.003 [-0.007, +0.002] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.009 |
| L22H9 add swap (control) | 0.982 | +0.000 [+0.000, +0.000] | 0.702 | -0.002 [-0.005, +0.002] | 0.000 / 0.000 | 0.00 / 0.28 | 0.995 / 0.000 | 0.002 | 0.004 / 0.010 |

Baseline of the earlier LCB run: mono 0.982, cross 0.704.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.62 | 0.75 | 0.69 |
| L22H6 steer | 0.98 | 1.00 | 0.97 | 1.00 | 0.99 | 0.74 | 0.58 | 0.75 | 0.67 |
| L22H6 swap | 0.64 | 0.54 | 0.91 | 0.00 | 0.99 | 0.53 | 0.40 | 0.60 | 0.00 |
| L22H6 add steer | 0.98 | 0.98 | 0.98 | 1.00 | 0.99 | 0.64 | 0.61 | 0.76 | 0.64 |
| L22H6 add swap | 0.98 | 0.99 | 0.98 | 1.00 | 0.98 | 0.74 | 0.60 | 0.75 | 0.67 |
| L22H4 steer (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.99 | 0.77 | 0.61 | 0.76 | 0.69 |
| L22H4 swap (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.99 | 0.76 | 0.62 | 0.76 | 0.67 |
| L22H4 add steer (control) | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.62 | 0.76 | 0.69 |
| L22H4 add swap (control) | 0.98 | 0.99 | 0.97 | 1.00 | 0.99 | 0.75 | 0.61 | 0.75 | 0.68 |
| L22H8 steer (control) | 0.99 | 0.99 | 0.98 | 1.00 | 0.99 | 0.75 | 0.61 | 0.75 | 0.68 |
| L22H8 swap (control) | 0.99 | 0.99 | 0.98 | 1.00 | 0.99 | 0.74 | 0.61 | 0.74 | 0.68 |
| L22H8 add steer (control) | 0.98 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.61 | 0.75 | 0.68 |
| L22H8 add swap (control) | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.75 | 0.62 | 0.75 | 0.69 |
| L22H9 steer (control) | 0.99 | 0.99 | 0.96 | 1.00 | 0.99 | 0.75 | 0.62 | 0.76 | 0.69 |
| L22H9 swap (control) | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.62 | 0.75 | 0.69 |
| L22H9 add steer (control) | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.75 | 0.61 | 0.75 | 0.69 |
| L22H9 add swap (control) | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 | 0.76 | 0.61 | 0.75 | 0.69 |
