# WPR for llama3.2-1b-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 0.95 (291) | 1.00 (100) | 0.96 (100) | 0.91 (99) | 0.54 (98) | 0.60 (194) | 0.825 |
| base | crosslingual | 0.80 (201) | 0.98 (258) | 0.74 (200) | 0.56 (169) | 0.36 (220) | 0.43 (192) | 0.645 |
| L8H25 zero | monolingual | 0.85 (261) | 1.00 (100) | 0.80 (82) | 0.71 (59) | 0.47 (96) | 0.70 (175) | 0.757 |
| L8H25 zero | crosslingual | 0.75 (9) | 0.91 (45) | 1.00 (1) |  | 0.00 (4) |  | 0.666 |
| L8H25 mean | monolingual | 0.86 (233) | 1.00 (100) | 0.80 (85) | 0.70 (37) | 0.47 (90) | 0.68 (169) | 0.752 |
| L8H25 mean | crosslingual | 0.56 (26) | 0.84 (104) | 0.50 (2) | 0.50 (4) | 0.30 (41) | 0.50 (8) | 0.532 |
| L8H12 zero (control) | monolingual | 0.93 (287) | 1.00 (100) | 0.91 (100) | 0.92 (100) | 0.55 (98) | 0.68 (195) | 0.832 |
| L8H12 zero (control) | crosslingual | 0.79 (191) | 0.97 (251) | 0.73 (182) | 0.56 (157) | 0.33 (209) | 0.53 (170) | 0.653 |
| L8H24 zero (control) | monolingual | 0.95 (290) | 1.00 (99) | 0.95 (99) | 0.89 (100) | 0.50 (98) | 0.66 (193) | 0.825 |
| L8H24 zero (control) | crosslingual | 0.77 (209) | 0.98 (257) | 0.74 (206) | 0.60 (173) | 0.35 (224) | 0.52 (187) | 0.663 |
| L8H28 zero (control) | monolingual | 0.92 (286) | 1.00 (100) | 0.89 (100) | 0.85 (99) | 0.42 (99) | 0.67 (188) | 0.792 |
| L8H28 zero (control) | crosslingual | 0.79 (202) | 0.98 (253) | 0.72 (202) | 0.58 (168) | 0.32 (219) | 0.47 (185) | 0.644 |
