# Detector check and CIs for the instruct heads

The 2,500-prompt continuations of each model relabeled with langid and fastText (lid.176); vote = at least two of langdetect, langid and fastText agree, otherwise unknown. c->w with a bootstrap 95% CI over prompts; rank and the largest other head are among the heads run on 2,500 prompts; Spearman compares c->w per head with langdetect's.

| model | head | detector | baseline non-English retention | c->w [95% CI] | rank | largest other head | Spearman vs langdetect |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b-instruct | L19H1 | langdetect | 0.996 | 0.200 [0.184, 0.216] | 3 of 56 | L0H25 0.629 | 1.000 |
| qwen2.5-7b-instruct | L19H1 | langid | 0.994 | 0.208 [0.192, 0.223] | 3 of 56 | L0H25 0.553 | 0.889 |
| qwen2.5-7b-instruct | L19H1 | fasttext | 0.997 | 0.200 [0.185, 0.217] | 3 of 56 | L0H25 0.586 | 0.888 |
| qwen2.5-7b-instruct | L19H1 | vote | 0.997 | 0.202 [0.186, 0.218] | 3 of 56 | L0H25 0.586 | 0.921 |
