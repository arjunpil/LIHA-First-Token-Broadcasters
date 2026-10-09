# Does the content stay when L11H3 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L11H1, L11H2, L11H0. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 1993 | 0.656 [0.647, 0.666] |
| baseline, drifts to English | 5 | 0.594 [0.320, 0.794] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.391] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L11H3 flips to English, before | 873 | 0.639 [0.624, 0.654] |
| L11H3 flips to English, after | 873 | 0.559 [0.546, 0.574] |
| L11H3: before vs after continuation, flipped | 873 | 0.586 [0.575, 0.597] |
| L11H3: before vs after continuation, stayed | 961 | 0.691 [0.679, 0.702] |
| L11H1: only 1 prompts flip to English | | |
| L11H2: only 0 prompts flip to English | | |
| L11H0: only 0 prompts flip to English | | |
