# Mechanism checks

The reading fixed in experiments/mechanism_checks.md, from the saved outputs (python experiments/mechanism_check_reading.py). Failures count replies that pass the line check without intervention and fail with it.

## Gemma-3-1B, L11H3 (control head L11H1)

12 of 95 scorable replies pass without intervention (de 1, es 2, fr 5, it 4).

| condition | fail | by language | fail without intervention, pass with it |
|---|---|---|---|
| head's attention to the name masked | 7 of 12 | es 2, fr 3, it 2 | 0 |
| head's attention to nearby tokens masked | 0 of 12 |  | 0 |
| control head's attention to the name masked | 0 of 12 |  | 0 |
| head zeroed during generation | 8 of 12 | de 1, es 2, fr 3, it 2 | 0 |

Failing under both the head's mask and zeroing: 4.

- Head's mask against the control head's mask: 7 vs 0 discordant, exact McNemar p = 0.0156
- Head's mask against the nearby mask: 7 vs 0 discordant, exact McNemar p = 0.0156

Last-prompt-token attention to the language name, mean over the prompts: L11H3 0.136, L11H1 0.095, L11H2 0.066, L11H0 0.025.

Reading: the head's mask fails more replies than the control head's mask and than the nearby mask, each with p < 0.05: yes.

## Qwen2.5-1.5B, L22H6 (control head L22H7)

62 of 96 scorable replies pass without intervention (de 12, es 17, fr 18, it 15).

| condition | fail | by language | fail without intervention, pass with it |
|---|---|---|---|
| head's attention to the name masked | 6 of 62 | es 3, fr 1, it 2 | 2 |
| head's attention to nearby tokens masked | 0 of 62 |  | 0 |
| control head's attention to the name masked | 0 of 62 |  | 0 |
| head zeroed during generation | 20 of 62 | de 1, es 2, fr 2, it 15 | 2 |

Failing under both the head's mask and zeroing: 5.

- Head's mask against the control head's mask: 6 vs 0 discordant, exact McNemar p = 0.0312
- Head's mask against the nearby mask: 6 vs 0 discordant, exact McNemar p = 0.0312

Last-prompt-token attention to the language name, mean over the prompts: L22H6 0.774, L22H7 0.042, L22H11 0.020, L22H9 0.018, L22H1 0.005, L22H3 0.005, L22H5 0.004, L22H4 0.003, L22H8 0.002, L22H10 0.001, L22H0 0.001, L22H2 0.000.

Reading: the head's mask fails more replies than the control head's mask and than the nearby mask, each with p < 0.05: yes.
