# Detector and prompt-split checks for the GPT-2 head sweep

Same 144 head-hook generations as results/gpt2, relabeled with langid and fastText (lid.176). vote = at least two of langdetect, langid and fastText agree, otherwise unknown. Ranks are among the 137 heads with dNLL <= 0.1.

| detector | baseline acc | Spearman c->w vs langdetect | L6H10 c->w (rank) | top 5 c->w |
|---|---|---|---|---|
| langdetect | 0.435 | 1.000 | 0.210 (1) | L6H10, L2H5, L4H8, L2H3, L8H6 |
| langid | 0.425 | 0.995 | 0.201 (1) | L6H10, L2H5, L4H8, L2H3, L8H6 |
| fasttext | 0.446 | 0.994 | 0.216 (1) | L6H10, L2H5, L4H8, L8H6, L2H3 |
| vote | 0.438 | 0.997 | 0.211 (1) | L6H10, L2H5, L4H8, L8H6, L2H3 |

Split halves: 200 random splits, stratified by language, c->w per head on each half.

| detector | Spearman between halves, mean [5th pct] | L6H10 rank 1 in both halves | top-5 overlap, mean |
|---|---|---|---|
| langdetect | 0.976 [0.969] | 200/200 | 4.3 / 5 |
| vote | 0.978 [0.973] | 200/200 | 3.9 / 5 |
