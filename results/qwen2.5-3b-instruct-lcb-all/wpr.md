# WPR for qwen2.5-3b-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 0.95 (297) | 0.99 (99) | 1.00 (100) | 0.99 (97) | 0.99 (100) | 0.99 (196) | 0.985 |
| base | crosslingual | 0.88 (271) | 0.98 (262) | 0.92 (255) | 0.94 (259) | 0.95 (265) | 0.95 (255) | 0.936 |
| L27H13 zero | monolingual | 0.79 (72) | 0.88 (59) | 0.97 (30) |  | 0.96 (54) | 0.99 (197) | 0.919 |
| L27H13 zero | crosslingual | 0.75 (126) | 0.91 (184) | 0.92 (152) | 1.00 (14) | 0.91 (158) | 0.93 (205) | 0.904 |
| L27H13 mean | monolingual | 0.78 (14) | 0.91 (35) | 0.96 (56) |  | 0.60 (5) | 0.98 (197) | 0.848 |
| L27H13 mean | crosslingual | 0.80 (72) | 0.85 (146) | 0.89 (182) | 1.00 (2) | 0.66 (51) | 0.94 (210) | 0.856 |
| L27H6 zero (control) | monolingual | 0.96 (299) | 0.99 (100) | 1.00 (100) | 1.00 (98) | 0.99 (100) | 0.99 (198) | 0.989 |
| L27H6 zero (control) | crosslingual | 0.89 (269) | 0.96 (266) | 0.94 (262) | 0.94 (255) | 0.96 (265) | 0.95 (261) | 0.941 |
| L27H12 zero (control) | monolingual | 0.97 (299) | 0.99 (98) | 1.00 (97) | 0.99 (97) | 0.98 (100) | 0.99 (194) | 0.986 |
| L27H12 zero (control) | crosslingual | 0.91 (268) | 0.98 (265) | 0.93 (265) | 0.94 (259) | 0.95 (272) | 0.97 (257) | 0.947 |
| L27H14 zero (control) | monolingual | 0.96 (299) | 0.98 (99) | 1.00 (99) | 0.99 (96) | 0.97 (99) | 0.99 (197) | 0.982 |
| L27H14 zero (control) | crosslingual | 0.86 (259) | 0.96 (246) | 0.93 (244) | 0.95 (246) | 0.97 (250) | 0.96 (252) | 0.938 |
