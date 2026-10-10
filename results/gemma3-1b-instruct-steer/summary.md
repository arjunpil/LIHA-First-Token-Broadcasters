# gemma3-1b-instruct L11H3: steering on the Language Confusion Benchmark

Greedy 100 tokens, chat template. The head's output is replaced by its mean output on the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds the language mean minus the mean over all languages instead of replacing. Controls are the same-layer heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose lines are all in the swap language. The head is replaced or shifted at every position, template and prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR leaves out.

| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.984 |  | 0.118 |  | 0.000 / 0.000 | 0.00 / 0.55 | 0.985 / 0.000 | 0.004 | 0.001 / 0.019 |
| L11H3 steer | 0.996 | +0.013 [+0.004, +0.021] | 0.560 | +0.440 [+0.410, +0.469] | 0.000 / 0.000 | 0.00 / 0.17 | 1.000 / 0.000 | 0.006 | 0.000 / 0.007 |
| L11H3 swap | 0.000 | -0.984 [-0.991, -0.975] | 0.000 | -0.116 [-0.133, -0.098] | 0.994 / 0.594 | 0.00 / 0.17 | 0.030 / 0.585 | 0.010 | 0.000 / 0.008 |
| L11H3 add steer | 0.966 | -0.015 [-0.028, -0.004] | 0.165 | +0.047 [+0.031, +0.064] | 0.000 / 0.000 | 0.00 / 0.48 | 0.995 / 0.000 | 0.004 | 0.001 / 0.021 |
| L11H3 add swap | 0.080 | -0.867 [-0.890, -0.843] | 0.045 | -0.070 [-0.090, -0.051] | 0.828 / 0.123 | 0.00 / 0.47 | 0.995 / 0.000 | 0.012 | 0.003 / 0.019 |
| L11H0 steer (control) | 0.981 | -0.003 [-0.010, +0.005] | 0.154 | +0.034 [+0.020, +0.048] | 0.000 / 0.000 | 0.00 / 0.49 | 0.995 / 0.000 | 0.005 | 0.000 / 0.005 |
| L11H0 swap (control) | 0.982 | -0.001 [-0.010, +0.008] | 0.150 | +0.032 [+0.019, +0.044] | 0.000 / 0.000 | 0.00 / 0.50 | 0.990 / 0.000 | 0.005 | 0.000 / 0.005 |
| L11H0 add steer (control) | 0.982 | -0.001 [-0.004, +0.000] | 0.137 | +0.019 [+0.010, +0.027] | 0.000 / 0.000 | 0.00 / 0.52 | 0.985 / 0.000 | 0.004 | 0.001 / 0.019 |
| L11H0 add swap (control) | 0.984 | +0.000 [-0.004, +0.004] | 0.133 | +0.014 [+0.007, +0.022] | 0.000 / 0.000 | 0.00 / 0.53 | 0.985 / 0.000 | 0.004 | 0.001 / 0.019 |
| L11H1 steer (control) | 0.988 | +0.004 [-0.005, +0.013] | 0.112 | -0.003 [-0.013, +0.008] | 0.000 / 0.000 | 0.00 / 0.55 | 0.980 / 0.000 | 0.004 | 0.001 / 0.017 |
| L11H1 swap (control) | 0.990 | +0.005 [-0.003, +0.013] | 0.112 | -0.003 [-0.012, +0.008] | 0.000 / 0.000 | 0.00 / 0.55 | 0.980 / 0.000 | 0.004 | 0.001 / 0.017 |
| L11H1 add steer (control) | 0.984 | +0.000 [+0.000, +0.000] | 0.116 | -0.003 [-0.006, +0.000] | 0.000 / 0.000 | 0.00 / 0.55 | 0.985 / 0.000 | 0.004 | 0.001 / 0.019 |
| L11H1 add swap (control) | 0.985 | +0.000 [-0.004, +0.004] | 0.118 | -0.001 [-0.003, +0.000] | 0.000 / 0.000 | 0.00 / 0.55 | 0.985 / 0.000 | 0.004 | 0.001 / 0.019 |
| L11H2 steer (control) | 0.991 | +0.006 [-0.001, +0.015] | 0.084 | -0.032 [-0.045, -0.020] | 0.000 / 0.000 | 0.00 / 0.61 | 0.995 / 0.000 | 0.004 | 0.001 / 0.010 |
| L11H2 swap (control) | 0.991 | +0.006 [-0.001, +0.015] | 0.082 | -0.034 [-0.047, -0.021] | 0.000 / 0.000 | 0.00 / 0.61 | 0.995 / 0.000 | 0.004 | 0.001 / 0.010 |
| L11H2 add steer (control) | 0.986 | +0.001 [-0.004, +0.006] | 0.113 | -0.006 [-0.011, -0.001] | 0.000 / 0.000 | 0.00 / 0.56 | 0.990 / 0.000 | 0.004 | 0.001 / 0.020 |
| L11H2 add swap (control) | 0.984 | +0.000 [+0.000, +0.000] | 0.113 | -0.006 [-0.011, -0.001] | 0.000 / 0.000 | 0.00 / 0.56 | 0.980 / 0.000 | 0.004 | 0.001 / 0.020 |

Baseline of the earlier LCB run: mono 0.984, cross 0.118.

LPR per task and language

| condition | monolingual/fr | monolingual/de | monolingual/es | monolingual/it | monolingual/en | crosslingual/fr | crosslingual/de | crosslingual/es | crosslingual/it |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.97 | 0.99 | 0.99 | 1.00 | 0.98 | 0.16 | 0.08 | 0.11 | 0.14 |
| L11H3 steer | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.47 | 0.82 | 0.46 | 0.49 |
| L11H3 swap | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| L11H3 add steer | 0.93 | 1.00 | 0.99 | 1.00 | 0.99 | 0.11 | 0.13 | 0.20 | 0.22 |
| L11H3 add swap | 0.00 | 0.89 | 0.01 | 0.00 | 0.99 | 0.01 | 0.14 | 0.01 | 0.02 |
| L11H0 steer (control) | 0.96 | 0.99 | 0.99 | 1.00 | 0.99 | 0.18 | 0.13 | 0.15 | 0.16 |
| L11H0 swap (control) | 0.96 | 1.00 | 0.99 | 1.00 | 0.99 | 0.19 | 0.11 | 0.14 | 0.16 |
| L11H0 add steer (control) | 0.97 | 0.99 | 0.98 | 1.00 | 0.98 | 0.17 | 0.11 | 0.13 | 0.15 |
| L11H0 add swap (control) | 0.97 | 1.00 | 0.99 | 1.00 | 0.98 | 0.17 | 0.09 | 0.13 | 0.14 |
| L11H1 steer (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.98 | 0.14 | 0.07 | 0.11 | 0.14 |
| L11H1 swap (control) | 0.98 | 0.99 | 0.99 | 1.00 | 0.98 | 0.14 | 0.07 | 0.11 | 0.13 |
| L11H1 add steer (control) | 0.97 | 0.99 | 0.99 | 1.00 | 0.98 | 0.15 | 0.07 | 0.10 | 0.14 |
| L11H1 add swap (control) | 0.97 | 0.99 | 0.99 | 1.00 | 0.98 | 0.15 | 0.08 | 0.11 | 0.14 |
| L11H2 steer (control) | 0.99 | 0.99 | 0.99 | 1.00 | 0.99 | 0.10 | 0.06 | 0.08 | 0.10 |
| L11H2 swap (control) | 0.98 | 1.00 | 0.99 | 1.00 | 0.99 | 0.09 | 0.06 | 0.09 | 0.09 |
| L11H2 add steer (control) | 0.97 | 1.00 | 0.99 | 1.00 | 0.99 | 0.14 | 0.07 | 0.11 | 0.13 |
| L11H2 add swap (control) | 0.97 | 0.99 | 0.99 | 1.00 | 0.98 | 0.14 | 0.08 | 0.10 | 0.14 |
