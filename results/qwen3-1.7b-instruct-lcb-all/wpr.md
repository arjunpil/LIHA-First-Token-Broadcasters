# WPR for qwen3-1.7b-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 0.97 (300) | 0.99 (100) | 0.97 (97) | 1.00 (100) | 0.99 (99) | 1.00 (178) | 0.986 |
| base | crosslingual | 0.89 (244) | 0.96 (249) | 0.90 (221) | 0.93 (211) | 0.90 (245) | 0.90 (230) | 0.914 |
| L18H12 zero | monolingual | 0.61 (35) | 0.85 (78) | 0.97 (40) | 1.00 (30) | 0.91 (69) | 0.99 (175) | 0.890 |
| L18H12 zero | crosslingual | 0.85 (20) | 0.94 (83) | 0.73 (16) | 0.00 (1) | 0.77 (81) | 0.82 (12) | 0.685 |
| L18H12 mean | monolingual | 0.86 (23) | 0.87 (84) | 0.93 (69) | 1.00 (14) | 0.83 (54) | 0.99 (169) | 0.915 |
| L18H12 mean | crosslingual | 0.82 (20) | 0.91 (81) | 0.73 (15) | 1.00 (2) | 0.74 (69) | 0.43 (7) | 0.772 |
| L18H6 zero (control) | monolingual | 0.98 (300) | 0.99 (100) | 0.98 (97) | 1.00 (100) | 0.99 (99) | 0.99 (181) | 0.988 |
| L18H6 zero (control) | crosslingual | 0.89 (249) | 0.96 (249) | 0.88 (222) | 0.92 (217) | 0.91 (248) | 0.89 (231) | 0.908 |
| L18H13 zero (control) | monolingual | 0.97 (300) | 0.99 (98) | 0.98 (95) | 1.00 (99) | 0.99 (99) | 1.00 (175) | 0.988 |
| L18H13 zero (control) | crosslingual | 0.88 (245) | 0.93 (249) | 0.88 (222) | 0.90 (224) | 0.88 (246) | 0.95 (229) | 0.904 |
| L18H14 zero (control) | monolingual | 0.99 (298) | 0.98 (100) | 0.97 (98) | 1.00 (99) | 0.97 (99) | 0.99 (176) | 0.983 |
| L18H14 zero (control) | crosslingual | 0.88 (247) | 0.95 (249) | 0.88 (223) | 0.87 (203) | 0.87 (250) | 0.90 (230) | 0.892 |
