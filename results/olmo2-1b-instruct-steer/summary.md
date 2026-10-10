# olmo2-1b-instruct L12H8: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.986 |  | 0.931 |  | 0.000 / 0.000 | 0.00 / 0.05 | 0.990 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H8 steer | 0.987 | +0.004 [-0.005, +0.013] | 0.932 | +0.000 [-0.010, +0.010] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.005 | 0.001 / 0.013 |
| L12H8 swap | 0.010 | -0.977 [-0.987, -0.966] | 0.010 | -0.923 [-0.937, -0.906] | 0.844 / 0.828 | 0.06 / 0.08 | 0.990 / 0.000 | 0.007 | 0.009 / 0.013 |
| L12H8 add steer | 0.994 | +0.010 [+0.003, +0.019] | 0.935 | +0.003 [-0.006, +0.012] | 0.000 / 0.000 | 0.00 / 0.05 | 1.000 / 0.000 | 0.006 | 0.003 / 0.013 |
| L12H8 add swap | 0.974 | -0.013 [-0.024, -0.003] | 0.875 | -0.054 [-0.070, -0.038] | 0.009 / 0.018 | 0.01 / 0.06 | 0.995 / 0.000 | 0.004 | 0.003 / 0.013 |
| L12H0 steer (control) | 0.987 | +0.001 [+0.000, +0.004] | 0.935 | +0.003 [-0.005, +0.012] | 0.000 / 0.000 | 0.01 / 0.05 | 1.000 / 0.000 | 0.004 | 0.003 / 0.012 |
| L12H0 swap (control) | 0.987 | +0.001 [+0.000, +0.004] | 0.935 | +0.003 [-0.005, +0.012] | 0.000 / 0.000 | 0.01 / 0.05 | 1.000 / 0.000 | 0.004 | 0.003 / 0.012 |
| L12H0 add steer (control) | 0.986 | +0.000 [+0.000, +0.000] | 0.931 | -0.001 [-0.003, +0.000] | 0.000 / 0.000 | 0.00 / 0.05 | 0.990 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H0 add swap (control) | 0.984 | -0.001 [-0.004, +0.000] | 0.930 | -0.001 [-0.004, +0.002] | 0.000 / 0.000 | 0.00 / 0.05 | 0.995 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H6 steer (control) | 0.987 | +0.003 [+0.000, +0.006] | 0.935 | +0.003 [-0.004, +0.009] | 0.000 / 0.000 | 0.00 / 0.05 | 1.000 / 0.000 | 0.004 | 0.003 / 0.013 |
| L12H6 swap (control) | 0.987 | +0.003 [+0.000, +0.006] | 0.934 | +0.002 [-0.005, +0.009] | 0.000 / 0.000 | 0.00 / 0.05 | 1.000 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H6 add steer (control) | 0.984 | -0.001 [-0.004, +0.000] | 0.930 | -0.002 [-0.004, +0.000] | 0.000 / 0.000 | 0.00 / 0.05 | 0.995 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H6 add swap (control) | 0.986 | +0.000 [+0.000, +0.000] | 0.931 | +0.000 [-0.003, +0.003] | 0.000 / 0.000 | 0.00 / 0.05 | 0.990 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H15 steer (control) | 0.983 | +0.000 [-0.006, +0.006] | 0.930 | -0.002 [-0.009, +0.006] | 0.000 / 0.000 | 0.01 / 0.05 | 0.995 / 0.000 | 0.005 | 0.001 / 0.017 |
| L12H15 swap (control) | 0.983 | +0.000 [-0.006, +0.006] | 0.932 | +0.001 [-0.007, +0.009] | 0.000 / 0.000 | 0.01 / 0.05 | 0.995 / 0.000 | 0.005 | 0.001 / 0.017 |
| L12H15 add steer (control) | 0.986 | +0.000 [+0.000, +0.000] | 0.931 | +0.000 [-0.003, +0.003] | 0.000 / 0.000 | 0.00 / 0.05 | 0.990 / 0.000 | 0.005 | 0.003 / 0.013 |
| L12H15 add swap (control) | 0.987 | +0.001 [+0.000, +0.004] | 0.931 | +0.000 [-0.003, +0.003] | 0.000 / 0.000 | 0.00 / 0.05 | 0.995 / 0.000 | 0.005 | 0.003 / 0.013 |

Baseline of the earlier LCB run: mono 0.986, cross 0.931.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.95 | 0.92 | 0.94 | 0.92 |
| L12H8 steer | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.94 | 0.93 | 0.95 | 0.91 |
| L12H8 swap | 0.01 | 0.00 | 0.01 | 0.00 | 0.99 | 0.02 | 0.00 | 0.02 | 0.00 |
| L12H8 add steer | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 0.94 | 0.92 | 0.95 | 0.93 |
| L12H8 add swap | 0.98 | 0.94 | 0.98 | 0.94 | 0.99 | 0.90 | 0.92 | 0.89 | 0.80 |
| L12H0 steer (control) | 0.99 | 0.98 | 0.99 | 0.96 | 1.00 | 0.93 | 0.93 | 0.95 | 0.92 |
| L12H0 swap (control) | 0.99 | 0.98 | 0.99 | 0.96 | 1.00 | 0.94 | 0.93 | 0.95 | 0.92 |
| L12H0 add steer (control) | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.95 | 0.92 | 0.94 | 0.92 |
| L12H0 add swap (control) | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.94 | 0.92 | 0.94 | 0.92 |
| L12H6 steer (control) | 0.99 | 0.98 | 0.99 | 0.96 | 1.00 | 0.94 | 0.93 | 0.95 | 0.92 |
| L12H6 swap (control) | 0.99 | 0.98 | 0.99 | 0.96 | 1.00 | 0.94 | 0.93 | 0.95 | 0.92 |
| L12H6 add steer (control) | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.94 | 0.92 | 0.94 | 0.92 |
| L12H6 add swap (control) | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.95 | 0.92 | 0.94 | 0.92 |
| L12H15 steer (control) | 0.99 | 0.97 | 0.99 | 0.96 | 0.99 | 0.95 | 0.92 | 0.94 | 0.90 |
| L12H15 swap (control) | 0.99 | 0.97 | 0.99 | 0.96 | 0.99 | 0.94 | 0.93 | 0.94 | 0.91 |
| L12H15 add steer (control) | 0.99 | 0.98 | 0.99 | 0.95 | 0.99 | 0.94 | 0.93 | 0.94 | 0.92 |
| L12H15 add swap (control) | 0.99 | 0.98 | 0.99 | 0.96 | 0.99 | 0.95 | 0.92 | 0.94 | 0.92 |
