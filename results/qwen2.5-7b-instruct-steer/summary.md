# qwen2.5-7b-instruct L19H1: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.950 |  | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H1 steer | 0.987 | +0.003 [-0.004, +0.010] | 0.948 | -0.003 [-0.011, +0.006] | 0.000 / 0.000 | 0.00 / 0.03 | 1.000 / 0.000 | 0.003 | 0.001 / 0.012 |
| L19H1 swap | 0.725 | -0.280 [-0.314, -0.247] | 0.684 | -0.269 [-0.295, -0.244] | 0.122 / 0.128 | 0.00 / 0.06 | 1.000 / 0.000 | 0.003 | 0.003 / 0.008 |
| L19H1 add steer | 0.982 | -0.004 [-0.013, +0.004] | 0.953 | +0.004 [-0.003, +0.013] | 0.000 / 0.000 | 0.00 / 0.03 | 0.995 / 0.000 | 0.003 | 0.001 / 0.012 |
| L19H1 add swap | 0.991 | +0.006 [-0.001, +0.014] | 0.943 | -0.008 [-0.016, +0.001] | 0.000 / 0.001 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.003 / 0.014 |
| L19H13 steer (control) | 0.981 | -0.001 [-0.008, +0.005] | 0.951 | +0.001 [-0.006, +0.007] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H13 swap (control) | 0.983 | +0.000 [-0.008, +0.008] | 0.951 | +0.001 [-0.005, +0.007] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.003 | 0.000 / 0.013 |
| L19H13 add steer (control) | 0.986 | +0.001 [-0.003, +0.005] | 0.951 | +0.001 [-0.004, +0.006] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H13 add swap (control) | 0.982 | -0.001 [-0.005, +0.003] | 0.947 | -0.002 [-0.006, +0.003] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.012 |
| L19H14 steer (control) | 0.986 | +0.003 [-0.005, +0.010] | 0.947 | -0.003 [-0.009, +0.003] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.003 | 0.000 / 0.015 |
| L19H14 swap (control) | 0.986 | +0.001 [-0.005, +0.008] | 0.947 | -0.003 [-0.009, +0.003] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.002 | 0.000 / 0.014 |
| L19H14 add steer (control) | 0.984 | +0.000 [+0.000, +0.000] | 0.951 | +0.001 [-0.004, +0.007] | 0.000 / 0.000 | 0.00 / 0.03 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H14 add swap (control) | 0.984 | +0.000 [+0.000, +0.000] | 0.951 | +0.000 [-0.004, +0.004] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.014 |
| L19H25 steer (control) | 0.988 | +0.004 [-0.003, +0.010] | 0.951 | +0.002 [-0.003, +0.008] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H25 swap (control) | 0.984 | +0.000 [-0.006, +0.008] | 0.950 | +0.000 [-0.007, +0.007] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.015 |
| L19H25 add steer (control) | 0.986 | +0.001 [-0.003, +0.005] | 0.952 | +0.002 [-0.002, +0.005] | 0.000 / 0.000 | 0.00 / 0.04 | 0.995 / 0.000 | 0.003 | 0.001 / 0.013 |
| L19H25 add swap (control) | 0.984 | +0.000 [-0.004, +0.004] | 0.948 | -0.002 [-0.006, +0.002] | 0.000 / 0.000 | 0.00 / 0.04 | 1.000 / 0.000 | 0.003 | 0.001 / 0.013 |

Baseline of the earlier LCB run: mono 0.984, cross 0.950.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
| L19H1 steer | 0.99 | 0.99 | 0.98 | 1.00 | 1.00 | 0.95 | 0.96 | 0.95 | 0.94 |
| L19H1 swap | 0.76 | 0.00 | 0.86 | 0.79 | 1.00 | 0.84 | 0.24 | 0.91 | 0.75 |
| L19H1 add steer | 0.98 | 0.99 | 0.98 | 0.98 | 0.99 | 0.94 | 0.95 | 0.97 | 0.95 |
| L19H1 add swap | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 0.93 | 0.95 | 0.96 | 0.94 |
| L19H13 steer (control) | 0.98 | 0.99 | 0.98 | 1.00 | 1.00 | 0.95 | 0.96 | 0.96 | 0.93 |
| L19H13 swap (control) | 0.98 | 0.99 | 0.98 | 1.00 | 0.99 | 0.95 | 0.95 | 0.96 | 0.94 |
| L19H13 add steer (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.96 | 0.95 | 0.96 | 0.93 |
| L19H13 add swap (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.93 |
| L19H14 steer (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.99 | 0.94 | 0.95 | 0.96 | 0.93 |
| L19H14 swap (control) | 0.97 | 0.99 | 1.00 | 1.00 | 0.99 | 0.94 | 0.95 | 0.96 | 0.94 |
| L19H14 add steer (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.96 | 0.96 | 0.93 |
| L19H14 add swap (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.93 |
| L19H25 steer (control) | 0.98 | 0.99 | 1.00 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.93 |
| L19H25 swap (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.97 | 0.93 |
| L19H25 add steer (control) | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.96 | 0.96 | 0.96 | 0.93 |
| L19H25 add swap (control) | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 0.95 | 0.95 | 0.96 | 0.94 |
