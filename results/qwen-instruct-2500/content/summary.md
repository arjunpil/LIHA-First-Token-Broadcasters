# Does the content stay when L22H6 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L22H9, L22H11, L22H8. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 1804 | 0.507 [0.497, 0.518] |
| baseline, drifts to English | 194 | 0.580 [0.546, 0.613] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.392] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L22H6 flips to English, before | 918 | 0.475 [0.461, 0.490] |
| L22H6 flips to English, after | 918 | 0.558 [0.545, 0.572] |
| L22H6: before vs after continuation, flipped | 918 | 0.486 [0.476, 0.497] |
| L22H6: before vs after continuation, stayed | 554 | 0.672 [0.655, 0.689] |
| L22H9: only 3 prompts flip to English | | |
| L22H11: only 9 prompts flip to English | | |
| L22H8: only 5 prompts flip to English | | |
