# qwen3-1.7b-instruct L18H12: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.985 |  | 0.823 |  | 0.000 / 0.000 | 0.00 / 0.14 | 0.985 / 0.000 | 0.008 | 0.004 / 0.016 |
| L18H12 steer | 0.980 | -0.004 [-0.013, +0.004] | 0.813 | -0.013 [-0.029, +0.004] | 0.000 / 0.000 | 0.00 / 0.15 | 0.985 / 0.000 | 0.008 | 0.001 / 0.017 |
| L18H12 swap | 0.004 | -0.982 [-0.991, -0.972] | 0.003 | -0.826 [-0.848, -0.804] | 0.926 / 0.701 | 0.04 / 0.26 | 0.960 / 0.040 | 0.014 | 0.004 / 0.021 |
| L18H12 add steer | 0.987 | +0.001 [-0.008, +0.010] | 0.866 | +0.041 [+0.028, +0.055] | 0.000 / 0.000 | 0.00 / 0.10 | 0.990 / 0.000 | 0.010 | 0.003 / 0.021 |
| L18H12 add swap | 0.944 | -0.036 [-0.051, -0.021] | 0.772 | -0.049 [-0.067, -0.032] | 0.015 / 0.048 | 0.00 / 0.14 | 0.995 / 0.000 | 0.009 | 0.001 / 0.018 |
| L18H6 steer (control) | 0.990 | +0.004 [-0.003, +0.011] | 0.837 | +0.014 [+0.003, +0.024] | 0.000 / 0.000 | 0.00 / 0.14 | 0.980 / 0.000 | 0.011 | 0.001 / 0.018 |
| L18H6 swap (control) | 0.992 | +0.005 [+0.000, +0.011] | 0.836 | +0.012 [+0.002, +0.023] | 0.000 / 0.000 | 0.00 / 0.14 | 0.980 / 0.000 | 0.011 | 0.001 / 0.018 |
| L18H6 add steer (control) | 0.985 | +0.000 [+0.000, +0.000] | 0.825 | +0.001 [-0.003, +0.004] | 0.000 / 0.000 | 0.00 / 0.14 | 0.990 / 0.000 | 0.008 | 0.003 / 0.017 |
| L18H6 add swap (control) | 0.985 | +0.000 [+0.000, +0.000] | 0.824 | +0.000 [-0.003, +0.003] | 0.000 / 0.000 | 0.00 / 0.14 | 0.985 / 0.000 | 0.008 | 0.004 / 0.017 |
| L18H13 steer (control) | 0.981 | -0.003 [-0.010, +0.004] | 0.866 | +0.041 [+0.028, +0.054] | 0.000 / 0.000 | 0.00 / 0.11 | 0.985 / 0.000 | 0.008 | 0.001 / 0.016 |
| L18H13 swap (control) | 0.978 | -0.005 [-0.014, +0.003] | 0.859 | +0.034 [+0.022, +0.046] | 0.000 / 0.000 | 0.00 / 0.11 | 0.990 / 0.000 | 0.008 | 0.001 / 0.017 |
| L18H13 add steer (control) | 0.982 | -0.003 [-0.006, +0.000] | 0.839 | +0.015 [+0.008, +0.024] | 0.000 / 0.000 | 0.00 / 0.13 | 0.980 / 0.000 | 0.008 | 0.004 / 0.016 |
| L18H13 add swap (control) | 0.985 | +0.000 [-0.006, +0.006] | 0.838 | +0.013 [+0.006, +0.020] | 0.000 / 0.000 | 0.00 / 0.13 | 0.990 / 0.000 | 0.008 | 0.004 / 0.017 |
| L18H14 steer (control) | 0.981 | -0.003 [-0.011, +0.006] | 0.841 | +0.018 [+0.004, +0.032] | 0.000 / 0.000 | 0.00 / 0.12 | 0.980 / 0.000 | 0.010 | 0.004 / 0.016 |
| L18H14 swap (control) | 0.979 | -0.004 [-0.013, +0.005] | 0.839 | +0.016 [+0.003, +0.030] | 0.000 / 0.000 | 0.00 / 0.13 | 0.980 / 0.000 | 0.010 | 0.004 / 0.016 |
| L18H14 add steer (control) | 0.985 | +0.000 [+0.000, +0.000] | 0.826 | +0.002 [+0.000, +0.004] | 0.000 / 0.000 | 0.00 / 0.14 | 0.985 / 0.000 | 0.008 | 0.003 / 0.018 |
| L18H14 add swap (control) | 0.985 | +0.000 [+0.000, +0.000] | 0.827 | +0.003 [-0.001, +0.007] | 0.000 / 0.000 | 0.00 / 0.14 | 0.985 / 0.000 | 0.008 | 0.004 / 0.017 |

Baseline of the earlier LCB run: mono 0.985, cross 0.823.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.78 | 0.85 | 0.81 |
| L18H12 steer | 0.98 | 1.00 | 0.98 | 0.99 | 0.98 | 0.85 | 0.74 | 0.87 | 0.79 |
| L18H12 swap | 0.01 | 0.00 | 0.00 | 0.00 | 0.96 | 0.01 | 0.00 | 0.00 | 0.00 |
| L18H12 add steer | 0.99 | 1.00 | 0.98 | 0.99 | 0.99 | 0.87 | 0.84 | 0.91 | 0.85 |
| L18H12 add swap | 0.95 | 0.97 | 0.94 | 0.95 | 0.99 | 0.81 | 0.71 | 0.80 | 0.77 |
| L18H6 steer (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.78 | 0.88 | 0.83 |
| L18H6 swap (control) | 0.99 | 1.00 | 0.99 | 1.00 | 0.98 | 0.85 | 0.78 | 0.88 | 0.83 |
| L18H6 add steer (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.99 | 0.85 | 0.78 | 0.85 | 0.82 |
| L18H6 add swap (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.78 | 0.85 | 0.81 |
| L18H13 steer (control) | 0.99 | 1.00 | 0.97 | 1.00 | 0.98 | 0.88 | 0.83 | 0.90 | 0.85 |
| L18H13 swap (control) | 0.98 | 0.99 | 0.98 | 1.00 | 0.99 | 0.87 | 0.83 | 0.89 | 0.85 |
| L18H13 add steer (control) | 0.99 | 1.00 | 0.97 | 1.00 | 0.98 | 0.85 | 0.81 | 0.87 | 0.82 |
| L18H13 add swap (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.99 | 0.86 | 0.81 | 0.86 | 0.82 |
| L18H14 steer (control) | 0.99 | 1.00 | 0.97 | 1.00 | 0.98 | 0.85 | 0.79 | 0.87 | 0.85 |
| L18H14 swap (control) | 0.99 | 1.00 | 0.97 | 1.00 | 0.98 | 0.85 | 0.79 | 0.87 | 0.85 |
| L18H14 add steer (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.79 | 0.85 | 0.81 |
| L18H14 add swap (control) | 0.99 | 1.00 | 0.98 | 1.00 | 0.98 | 0.85 | 0.79 | 0.85 | 0.82 |
