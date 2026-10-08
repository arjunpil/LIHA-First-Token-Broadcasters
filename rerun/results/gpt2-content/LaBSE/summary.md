# Does the content stay when L6H10 is removed? (sentence-transformers/LaBSE)

Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a multilingual encoder. Controls: L7H2, L8H1, L10H11. Means with bootstrap 95% CIs.

| group | n | prompt vs continuation |
|---|---|---|
| baseline, continues in the prompt language | 592 | 0.419 [0.392, 0.445] |
| baseline, drifts to English | 1302 | 0.157 [0.151, 0.164] |
| reference: FLORES sentence vs same language, next sentence | 1976 | 0.426 [0.421, 0.431] |
| reference: FLORES sentence vs same language, random sentence | 1980 | 0.300 [0.296, 0.303] |
| reference: FLORES sentence vs English, next sentence | 1976 | 0.333 [0.327, 0.339] |
| reference: FLORES sentence vs English, random sentence | 1980 | 0.198 [0.195, 0.202] |
| L6H10 flips to English, before | 511 | 0.395 [0.367, 0.422] |
| L6H10 flips to English, after | 511 | 0.112 [0.103, 0.122] |
| L6H10: before vs after continuation, flipped | 511 | 0.183 [0.172, 0.194] |
| L6H10: before vs after continuation, stayed | 68 | 0.873 [0.816, 0.926] |
| L7H2 flips to English, before | 34 | 0.425 [0.322, 0.524] |
| L7H2 flips to English, after | 34 | 0.175 [0.131, 0.224] |
| L7H2: before vs after continuation, flipped | 34 | 0.272 [0.195, 0.358] |
| L7H2: before vs after continuation, stayed | 539 | 0.922 [0.906, 0.936] |
| L8H1: only 8 prompts flip to English | | |
| L10H11 flips to English, before | 15 | 0.443 [0.275, 0.618] |
| L10H11 flips to English, after | 15 | 0.157 [0.108, 0.210] |
| L10H11: before vs after continuation, flipped | 15 | 0.294 [0.212, 0.388] |
| L10H11: before vs after continuation, stayed | 568 | 0.912 [0.895, 0.926] |
