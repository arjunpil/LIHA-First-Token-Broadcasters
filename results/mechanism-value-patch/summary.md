# Value patching and the larger mechanism sample

Reading fixed in experiments/mechanism_value_patch.md. Counts over baseline-passing replies unless said otherwise; exact two-sided McNemar tests, uncorrected. Sample 1 = the 96 earlier prompts (its nearby and control-head masks come from the earlier run, whose clean replies sample 1 reproduces exactly); sample 2 = 384 new prompts.

## Qwen2.5-1.5B L22H6

sample 1: 62 of 96 scorable replies pass at baseline (96 prompts)
sample 2: 261 of 378 scorable replies pass at baseline (384 prompts)
pooled: 323 of 474 scorable replies pass at baseline (480 prompts)

| condition | sample | pass to fail | fail to pass | newly unscorable | all lines in donor language (of scorable) | mean output change |
|---|---|---|---|---|---|---|
| target_word_mask | 1 | 6 | 2 | 1 | 0 / 95 | |
| target_word_mask | 2 | 28 | 2 | 1 | 0 / 378 | |
| target_word_mask | pooled | 34 | 4 | 2 | 0 / 473 | |
| nearby_word_mask | 1 | 0 | 0 | 0 | 0 / 96 | |
| nearby_word_mask | 2 | 0 | 0 | 0 | 0 / 378 | |
| nearby_word_mask | pooled | 0 | 0 | 0 | 0 / 474 | |
| control_head_word_mask | 1 | 0 | 0 | 0 | 0 / 96 | |
| control_head_word_mask | 2 | 0 | 0 | 0 | 0 / 378 | |
| control_head_word_mask | pooled | 0 | 0 | 0 | 0 / 474 | |
| head_zero_decode | 1 | 20 | 2 | 0 | 0 / 96 | |
| head_zero_decode | 2 | 85 | 2 | 1 | 0 / 378 | |
| head_zero_decode | pooled | 105 | 4 | 1 | 0 / 474 | |
| value_patch_same | 1 | 0 | 2 | 0 | 0 / 96 | 0.401 |
| value_patch_same | 2 | 1 | 1 | 0 | 0 / 378 | 0.389 |
| value_patch_same | pooled | 1 | 3 | 0 | 0 / 474 | 0.391 |
| value_patch_other | 1 | 17 | 1 | 1 | 8 / 95 | 1.953 |
| value_patch_other | 2 | 73 | 2 | 1 | 36 / 378 | 2.021 |
| value_patch_other | pooled | 90 | 3 | 2 | 44 / 473 | 2.007 |

Baseline replies already all in the donor language: 0

Readings:
- Guard, pooled: value_patch_same against nearby mask, 1 vs 0 discordant, p = 1; read as a general perturbation: no
- Content, pooled: value_patch_other against value_patch_same, 89 vs 0 discordant failures, p = 3.231e-27
- Direction, pooled: replies all in the donor language, other vs same, 44 vs 0 discordant of 472 scorable in both, p = 1.137e-13
- Replication, sample 2: name mask against nearby mask 28 vs 0, p = 7.451e-09; against control head's mask 28 vs 0, p = 7.451e-09

Pass to fail by language, pooled (baseline-passing n in brackets):

| condition | de | es | fr | it |
|---|---|---|---|---|
| target_word_mask | 4 (68) | 6 (88) | 4 (88) | 20 (79) |
| nearby_word_mask | 0 (68) | 0 (88) | 0 (88) | 0 (79) |
| control_head_word_mask | 0 (68) | 0 (88) | 0 (88) | 0 (79) |
| head_zero_decode | 5 (68) | 11 (88) | 10 (88) | 79 (79) |
| value_patch_same | 1 (68) | 0 (88) | 0 (88) | 0 (79) |
| value_patch_other | 8 (68) | 16 (88) | 15 (88) | 51 (79) |

## Gemma-3-1B L11H3

sample 1: 12 of 95 scorable replies pass at baseline (96 prompts)
sample 2: 39 of 377 scorable replies pass at baseline (384 prompts)
pooled: 51 of 472 scorable replies pass at baseline (480 prompts)

| condition | sample | pass to fail | fail to pass | newly unscorable | all lines in donor language (of scorable) | mean output change |
|---|---|---|---|---|---|---|
| target_word_mask | 1 | 7 | 0 | 0 | 0 / 95 | |
| target_word_mask | 2 | 25 | 0 | 1 | 0 / 376 | |
| target_word_mask | pooled | 32 | 0 | 1 | 0 / 471 | |
| nearby_word_mask | 1 | 0 | 0 | 0 | 0 / 95 | |
| nearby_word_mask | 2 | 0 | 0 | 0 | 0 / 377 | |
| nearby_word_mask | pooled | 0 | 0 | 0 | 0 / 472 | |
| control_head_word_mask | 1 | 0 | 0 | 0 | 0 / 95 | |
| control_head_word_mask | 2 | 2 | 2 | 0 | 0 / 377 | |
| control_head_word_mask | pooled | 2 | 2 | 0 | 0 / 472 | |
| head_zero_decode | 1 | 8 | 0 | 0 | 0 / 95 | |
| head_zero_decode | 2 | 28 | 1 | 1 | 0 / 376 | |
| head_zero_decode | pooled | 36 | 1 | 1 | 0 / 471 | |
| value_patch_same | 1 | 0 | 1 | 0 | 0 / 95 | 0.101 |
| value_patch_same | 2 | 0 | 1 | 0 | 0 / 377 | 0.102 |
| value_patch_same | pooled | 0 | 2 | 0 | 0 / 472 | 0.102 |
| value_patch_other | 1 | 10 | 0 | 0 | 7 / 95 | 0.357 |
| value_patch_other | 2 | 33 | 0 | 1 | 17 / 376 | 0.361 |
| value_patch_other | pooled | 43 | 0 | 1 | 24 / 471 | 0.360 |

Baseline replies already all in the donor language: 0

Readings:
- Guard, pooled: value_patch_same against nearby mask, 0 vs 0 discordant, p = 1; read as a general perturbation: no
- Content, pooled: value_patch_other against value_patch_same, 43 vs 0 discordant failures, p = 2.274e-13
- Direction, pooled: replies all in the donor language, other vs same, 24 vs 0 discordant of 471 scorable in both, p = 1.192e-07
- Replication, sample 2: name mask against nearby mask 25 vs 0, p = 5.96e-08; against control head's mask 23 vs 0, p = 2.384e-07

Pass to fail by language, pooled (baseline-passing n in brackets):

| condition | de | es | fr | it |
|---|---|---|---|---|
| target_word_mask | 4 (6) | 12 (14) | 7 (18) | 9 (13) |
| nearby_word_mask | 0 (6) | 0 (14) | 0 (18) | 0 (13) |
| control_head_word_mask | 1 (6) | 0 (14) | 0 (18) | 1 (13) |
| head_zero_decode | 5 (6) | 11 (14) | 10 (18) | 10 (13) |
| value_patch_same | 0 (6) | 0 (14) | 0 (18) | 0 (13) |
| value_patch_other | 5 (6) | 13 (14) | 14 (18) | 11 (13) |

