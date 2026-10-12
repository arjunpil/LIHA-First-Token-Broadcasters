# Controls for the value patch

Reading fixed in experiments/mechanism_value_controls.md (79dc4e6). Samples 1 and 2 pooled; counts over baseline-passing replies unless said otherwise; exact two-sided McNemar tests, uncorrected. The clean replies of the new runs reproduce the earlier ones exactly (checked).

## Qwen2.5-1.5B L22H6

323 of 474 scorable replies pass at baseline (480 prompts)

| condition | pass to fail | fail to pass | newly unscorable | all lines in first donor language | all lines in second donor language | mean output change |
|---|---|---|---|---|---|---|
| value_patch_same | 1 | 3 | 0 | 0 / 474 | 0 / 474 | 0.391 |
| value_patch_other | 90 | 3 | 2 | 44 / 473 | 2 / 473 | 2.007 |
| target_word_mask | 34 | 4 | 2 | 0 / 473 | 4 / 473 |  |
| value_patch_other2 | 217 | 0 | 2 | 0 / 476 | 208 / 476 | 1.976 |
| value_patch_random1 | 5 | 3 | 0 | 0 / 474 | 0 / 474 | 2.034 |
| value_patch_random2 | 8 | 7 | 0 | 0 / 474 | 0 / 474 | 2.028 |
| value_patch_random3 | 4 | 5 | 0 | 0 / 474 | 0 / 474 | 2.043 |

Readings:
- Size, other-language against random-direction failures, per seed: 87 vs 2, p = 1.29e-23; 83 vs 1, p = 8.79e-24; 88 vs 2, p = 6.62e-24. Read as depending on direction: yes
- Direction, replies all in the first donor language, other-language against random, per seed: 44 vs 0, p = 1.14e-13; 44 vs 0, p = 1.14e-13; 44 vs 0, p = 1.14e-13. Read as specific to its values: yes
- Second language, replies all in the second donor language, other2 against same: 205 vs 0, p = 3.89e-62. Shown: yes

## Gemma-3-1B L11H3

51 of 472 scorable replies pass at baseline (480 prompts)

| condition | pass to fail | fail to pass | newly unscorable | all lines in first donor language | all lines in second donor language | mean output change |
|---|---|---|---|---|---|---|
| value_patch_same | 0 | 2 | 0 | 0 / 472 | 0 / 472 | 0.102 |
| value_patch_other | 43 | 0 | 1 | 24 / 471 | 0 / 471 | 0.360 |
| target_word_mask | 32 | 0 | 1 | 0 / 471 | 0 / 471 |  |
| value_patch_other2 | 25 | 0 | 1 | 0 / 472 | 16 / 472 | 0.335 |
| value_patch_random1 | 1 | 4 | 0 | 0 / 472 | 0 / 472 | 0.367 |
| value_patch_random2 | 3 | 10 | 0 | 0 / 472 | 0 / 472 | 0.367 |
| value_patch_random3 | 6 | 3 | 0 | 0 / 472 | 0 / 472 | 0.368 |
| l11h2_word_mask | 2 | 5 | 0 | 0 / 472 | 0 / 472 |  |

Readings:
- Size, other-language against random-direction failures, per seed: 42 vs 0, p = 4.55e-13; 40 vs 0, p = 1.82e-12; 37 vs 0, p = 1.46e-11. Read as depending on direction: yes
- Direction, replies all in the first donor language, other-language against random, per seed: 24 vs 0, p = 1.19e-07; 24 vs 0, p = 1.19e-07; 24 vs 0, p = 1.19e-07. Read as specific to its values: yes
- Second language, replies all in the second donor language, other2 against same: 15 vs 0, p = 6.1e-05. Shown: yes
- L11H2, L11H3 name mask against L11H2 name mask failures: 30 vs 0, p = 1.86e-09. Shown: yes

