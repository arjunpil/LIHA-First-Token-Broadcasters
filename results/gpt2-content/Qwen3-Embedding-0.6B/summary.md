# Does the content stay when L6H10 is removed? (Qwen/Qwen3-Embedding-0.6B)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L7H2, L8H1, L10H11. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 592 | 0.471 [0.447, 0.496] |
| baseline, drifts to English | 1302 | 0.219 [0.212, 0.227] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.384 [0.377, 0.392] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.179 [0.175, 0.183] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.370 [0.363, 0.377] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.172 [0.169, 0.176] |
| L6H10 flips to English, before | 511 | 0.453 [0.428, 0.477] |
| L6H10 flips to English, after | 511 | 0.195 [0.185, 0.206] |
| L6H10: before vs after continuation, flipped | 511 | 0.220 [0.210, 0.229] |
| L6H10: before vs after continuation, stayed | 68 | 0.865 [0.804, 0.919] |
| L7H2 flips to English, before | 34 | 0.478 [0.381, 0.573] |
| L7H2 flips to English, after | 34 | 0.203 [0.165, 0.249] |
| L7H2: before vs after continuation, flipped | 34 | 0.329 [0.265, 0.399] |
| L7H2: before vs after continuation, stayed | 539 | 0.916 [0.899, 0.931] |
| L8H1: only 8 prompts flip to English | | |
| L10H11 flips to English, before | 15 | 0.477 [0.351, 0.608] |
| L10H11 flips to English, after | 15 | 0.209 [0.169, 0.249] |
| L10H11: before vs after continuation, flipped | 15 | 0.282 [0.232, 0.333] |
| L10H11: before vs after continuation, stayed | 568 | 0.905 [0.888, 0.921] |
