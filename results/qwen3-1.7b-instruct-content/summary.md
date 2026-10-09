# Does the content stay when L18H12 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L18H8, L18H15, L18H7. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 1987 | 0.679 [0.669, 0.690] |
| baseline, drifts to English | 4 | 0.680 [0.468, 0.818] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.391] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L18H12 flips to English, before | 739 | 0.635 [0.615, 0.654] |
| L18H12 flips to English, after | 739 | 0.683 [0.671, 0.693] |
| L18H12: before vs after continuation, flipped | 739 | 0.600 [0.586, 0.612] |
| L18H12: before vs after continuation, stayed | 1176 | 0.780 [0.768, 0.791] |
| L18H8: only 1 prompts flip to English | | |
| L18H15: only 1 prompts flip to English | | |
| L18H7: only 1 prompts flip to English | | |
