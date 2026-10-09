# Does the content stay when L12H8 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L12H7, L12H14, L12H6. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 1993 | 0.706 [0.700, 0.713] |
| baseline, drifts to English | 4 | 0.606 [0.450, 0.763] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.391] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L12H8 flips to English, before | 101 | 0.702 [0.670, 0.734] |
| L12H8 flips to English, after | 101 | 0.668 [0.643, 0.695] |
| L12H8: before vs after continuation, flipped | 101 | 0.607 [0.575, 0.637] |
| L12H8: before vs after continuation, stayed | 1792 | 0.719 [0.711, 0.727] |
| L12H7: only 0 prompts flip to English | | |
| L12H14: only 2 prompts flip to English | | |
| L12H6: only 1 prompts flip to English | | |
