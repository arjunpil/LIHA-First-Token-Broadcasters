# qwen2.5-3b-instruct L27H13: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.982 |  | 0.888 |  | 0.000 / 0.000 | 0.00 / 0.09 | 1.000 / 0.000 | 0.002 | 0.003 / 0.013 |
| L27H13 steer | 0.989 | +0.009 [-0.001, +0.019] | 0.894 | +0.007 [-0.005, +0.019] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.003 | 0.003 / 0.010 |
| L27H13 swap | 0.000 | -0.981 [-0.990, -0.972] | 0.000 | -0.890 [-0.907, -0.871] | 0.982 / 0.829 | 0.01 / 0.13 | 0.995 / 0.000 | 0.004 | 0.001 / 0.012 |
| L27H13 add steer | 0.991 | +0.010 [-0.001, +0.021] | 0.906 | +0.018 [+0.006, +0.030] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.003 | 0.003 / 0.010 |
| L27H13 add swap | 0.739 | -0.253 [-0.284, -0.223] | 0.755 | -0.130 [-0.151, -0.110] | 0.165 / 0.081 | 0.01 / 0.10 | 1.000 / 0.000 | 0.003 | 0.001 / 0.014 |
| L27H6 steer (control) | 0.986 | +0.004 [-0.006, +0.014] | 0.886 | -0.002 [-0.012, +0.008] | 0.000 / 0.000 | 0.00 / 0.09 | 0.995 / 0.000 | 0.003 | 0.003 / 0.013 |
| L27H6 swap (control) | 0.988 | +0.006 [-0.003, +0.015] | 0.889 | +0.001 [-0.009, +0.010] | 0.000 / 0.000 | 0.00 / 0.08 | 0.995 / 0.000 | 0.003 | 0.003 / 0.013 |
| L27H6 add steer (control) | 0.983 | +0.001 [+0.000, +0.004] | 0.888 | +0.000 [+0.000, +0.000] | 0.000 / 0.000 | 0.00 / 0.09 | 1.000 / 0.000 | 0.003 | 0.003 / 0.013 |
| L27H6 add swap (control) | 0.982 | +0.000 [+0.000, +0.000] | 0.886 | -0.002 [-0.004, +0.000] | 0.000 / 0.000 | 0.00 / 0.09 | 1.000 / 0.000 | 0.002 | 0.003 / 0.013 |
| L27H12 steer (control) | 0.978 | -0.003 [-0.010, +0.005] | 0.901 | +0.012 [+0.003, +0.021] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.002 | 0.003 / 0.008 |
| L27H12 swap (control) | 0.976 | -0.004 [-0.013, +0.004] | 0.899 | +0.010 [+0.000, +0.019] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.003 | 0.003 / 0.009 |
| L27H12 add steer (control) | 0.982 | +0.000 [-0.004, +0.004] | 0.890 | +0.002 [-0.002, +0.006] | 0.000 / 0.000 | 0.00 / 0.08 | 1.000 / 0.000 | 0.002 | 0.003 / 0.012 |
| L27H12 add swap (control) | 0.983 | +0.001 [+0.000, +0.004] | 0.891 | +0.003 [-0.003, +0.008] | 0.000 / 0.000 | 0.00 / 0.08 | 1.000 / 0.000 | 0.002 | 0.003 / 0.012 |
| L27H14 steer (control) | 0.987 | +0.005 [-0.003, +0.013] | 0.906 | +0.019 [+0.008, +0.030] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.003 | 0.003 / 0.011 |
| L27H14 swap (control) | 0.984 | +0.004 [-0.004, +0.011] | 0.909 | +0.022 [+0.013, +0.032] | 0.000 / 0.000 | 0.00 / 0.07 | 1.000 / 0.000 | 0.003 | 0.003 / 0.012 |
| L27H14 add steer (control) | 0.984 | +0.001 [-0.004, +0.006] | 0.897 | +0.008 [+0.000, +0.015] | 0.000 / 0.000 | 0.00 / 0.08 | 1.000 / 0.000 | 0.003 | 0.003 / 0.010 |
| L27H14 add swap (control) | 0.982 | +0.000 [-0.005, +0.005] | 0.897 | +0.008 [+0.002, +0.015] | 0.000 / 0.000 | 0.00 / 0.08 | 1.000 / 0.000 | 0.002 | 0.003 / 0.013 |

Baseline of the earlier LCB run: mono 0.982, cross 0.888.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 0.91 | 0.87 | 0.92 | 0.86 |
| L27H13 steer | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 | 0.91 | 0.88 | 0.92 | 0.86 |
| L27H13 swap | 0.00 | 0.00 | 0.00 | 0.00 | 0.99 | 0.00 | 0.00 | 0.00 | 0.00 |
| L27H13 add steer | 0.98 | 1.00 | 0.99 | 1.00 | 1.00 | 0.90 | 0.90 | 0.93 | 0.88 |
| L27H13 add swap | 0.74 | 0.99 | 0.74 | 0.39 | 1.00 | 0.77 | 0.88 | 0.85 | 0.52 |
| L27H6 steer (control) | 0.98 | 1.00 | 0.98 | 0.98 | 0.99 | 0.91 | 0.87 | 0.92 | 0.85 |
| L27H6 swap (control) | 0.99 | 1.00 | 0.99 | 0.98 | 0.99 | 0.91 | 0.87 | 0.92 | 0.86 |
| L27H6 add steer (control) | 0.99 | 0.99 | 0.98 | 0.97 | 1.00 | 0.91 | 0.87 | 0.92 | 0.86 |
| L27H6 add swap (control) | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 0.90 | 0.87 | 0.92 | 0.86 |
| L27H12 steer (control) | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 0.92 | 0.88 | 0.94 | 0.86 |
| L27H12 swap (control) | 0.99 | 0.99 | 0.97 | 0.97 | 1.00 | 0.92 | 0.88 | 0.92 | 0.87 |
| L27H12 add steer (control) | 0.99 | 0.99 | 0.98 | 0.97 | 1.00 | 0.90 | 0.87 | 0.93 | 0.86 |
| L27H12 add swap (control) | 0.99 | 0.99 | 0.98 | 0.97 | 1.00 | 0.91 | 0.87 | 0.92 | 0.86 |
| L27H14 steer (control) | 0.99 | 0.99 | 0.98 | 0.99 | 1.00 | 0.91 | 0.89 | 0.95 | 0.88 |
| L27H14 swap (control) | 0.99 | 1.00 | 0.98 | 0.99 | 1.00 | 0.92 | 0.89 | 0.94 | 0.88 |
| L27H14 add steer (control) | 0.99 | 0.99 | 0.98 | 0.97 | 1.00 | 0.90 | 0.88 | 0.93 | 0.87 |
| L27H14 add swap (control) | 0.99 | 0.99 | 0.98 | 0.97 | 1.00 | 0.92 | 0.88 | 0.92 | 0.87 |
