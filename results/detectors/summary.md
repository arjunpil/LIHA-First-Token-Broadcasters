# Detector check and CIs for the instruct heads

The 2,500-prompt continuations of each model relabeled with langid and fastText (lid.176); vote = at least two of langdetect, langid and fastText agree, otherwise unknown. c->w with a bootstrap 95% CI over prompts; rank and the largest other head are among the heads run on 2,500 prompts; Spearman compares c->w per head with langdetect's.

| model | head | detector | baseline non-English retention | c->w [95% CI] | rank | largest other head | Spearman vs langdetect |
|---|---|---|---|---|---|---|---|
| qwen-instruct-full | L22H6 | langdetect | 0.902 | 0.500 [0.481, 0.520] | 1 of 336 | L17H7 0.325 | 1.000 |
| qwen-instruct-full | L22H6 | langid | 0.902 | 0.502 [0.481, 0.521] | 1 of 336 | L17H7 0.324 | 0.996 |
| qwen-instruct-full | L22H6 | fasttext | 0.903 | 0.501 [0.481, 0.519] | 1 of 336 | L17H7 0.322 | 0.998 |
| qwen-instruct-full | L22H6 | vote | 0.903 | 0.500 [0.481, 0.520] | 1 of 336 | L17H7 0.323 | 0.998 |
| qwen2.5-3b-instruct | L27H13 | langdetect | 0.992 | 0.515 [0.495, 0.536] | 1 of 32 | L0H12 0.020 | 1.000 |
| qwen2.5-3b-instruct | L27H13 | langid | 0.992 | 0.531 [0.511, 0.551] | 1 of 32 | L0H12 0.021 | 0.900 |
| qwen2.5-3b-instruct | L27H13 | fasttext | 0.994 | 0.518 [0.499, 0.539] | 1 of 32 | L0H12 0.020 | 0.900 |
| qwen2.5-3b-instruct | L27H13 | vote | 0.993 | 0.518 [0.499, 0.539] | 1 of 32 | L0H12 0.020 | 0.940 |
| qwen3-1.7b-instruct | L18H12 | langdetect | 0.994 | 0.324 [0.306, 0.343] | 2 of 48 | L0H3 0.466 | 1.000 |
| qwen3-1.7b-instruct | L18H12 | langid | 0.995 | 0.326 [0.307, 0.344] | 2 of 48 | L0H3 0.441 | 0.869 |
| qwen3-1.7b-instruct | L18H12 | fasttext | 0.997 | 0.332 [0.315, 0.351] | 2 of 48 | L0H3 0.458 | 0.834 |
| qwen3-1.7b-instruct | L18H12 | vote | 0.995 | 0.326 [0.308, 0.345] | 2 of 48 | L0H3 0.453 | 0.941 |
| gemma3-1b-instruct | L11H3 | langdetect | 0.997 | 0.413 [0.394, 0.431] | 1 of 8 | L5H0 0.064 | 1.000 |
| gemma3-1b-instruct | L11H3 | langid | 0.997 | 0.413 [0.394, 0.433] | 1 of 8 | L5H0 0.060 | 0.961 |
| gemma3-1b-instruct | L11H3 | fasttext | 0.998 | 0.399 [0.380, 0.418] | 1 of 8 | L5H0 0.061 | 0.858 |
| gemma3-1b-instruct | L11H3 | vote | 0.997 | 0.410 [0.392, 0.429] | 1 of 8 | L5H0 0.062 | 0.961 |
| gemma3-4b-instruct | L24H0 | langdetect | 0.997 | 0.214 [0.198, 0.230] | 1 of 16 | L0H3 0.002 | 1.000 |
| gemma3-4b-instruct | L24H0 | langid | 0.998 | 0.223 [0.208, 0.240] | 1 of 16 | L0H3 0.002 | 0.863 |
| gemma3-4b-instruct | L24H0 | fasttext | 0.999 | 0.215 [0.200, 0.230] | 1 of 16 | L0H3 0.002 | 0.845 |
| gemma3-4b-instruct | L24H0 | vote | 0.997 | 0.218 [0.201, 0.234] | 1 of 16 | L0H3 0.001 | 0.942 |
| olmo2-1b-instruct | L12H8 | langdetect | 0.997 | 0.080 [0.070, 0.091] | 1 of 48 | L0H10 0.041 | 1.000 |
| olmo2-1b-instruct | L12H8 | langid | 0.998 | 0.088 [0.076, 0.098] | 1 of 48 | L0H10 0.041 | 0.789 |
| olmo2-1b-instruct | L12H8 | fasttext | 0.998 | 0.085 [0.074, 0.096] | 1 of 48 | L0H10 0.040 | 0.816 |
| olmo2-1b-instruct | L12H8 | vote | 0.998 | 0.083 [0.072, 0.094] | 1 of 48 | L0H10 0.041 | 0.843 |
