# Does the content stay when L27H13 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L27H8, L27H15, L27H7. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 1983 | 0.663 [0.655, 0.671] |
| baseline, drifts to English | 11 | 0.659 [0.558, 0.751] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.391] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L27H13 flips to English, before | 514 | 0.590 [0.570, 0.607] |
| L27H13 flips to English, after | 514 | 0.657 [0.644, 0.670] |
| L27H13: before vs after continuation, flipped | 514 | 0.582 [0.569, 0.596] |
| L27H13: before vs after continuation, stayed | 695 | 0.701 [0.689, 0.714] |
| L27H8: only 2 prompts flip to English | | |
| L27H15: only 2 prompts flip to English | | |
| L27H7: only 0 prompts flip to English | | |
