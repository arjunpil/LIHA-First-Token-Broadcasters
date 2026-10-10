# WPR for olmo2-1b-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.00 (281) | 1.00 (91) | 0.99 (95) | 0.98 (100) | 0.98 (99) | 0.98 (186) | 0.989 |
| base | crosslingual | 0.97 (243) | 0.95 (257) | 0.88 (250) | 0.86 (222) | 0.87 (267) | 0.89 (257) | 0.904 |
| L12H8 zero | monolingual | 0.90 (152) | 1.00 (90) | 0.88 (40) | 0.90 (51) | 0.73 (75) | 0.94 (115) | 0.891 |
| L12H8 zero | crosslingual | 0.85 (75) | 0.72 (118) | 0.73 (89) | 0.68 (89) | 0.65 (185) | 0.76 (216) | 0.732 |
| L12H8 mean | monolingual | 0.88 (28) | 0.96 (77) | 0.68 (25) | 0.65 (17) | 0.25 (4) | 0.98 (85) | 0.731 |
| L12H8 mean | crosslingual | 0.00 (1) | 0.37 (39) | 0.64 (60) | 0.40 (21) | 0.42 (15) | 0.73 (170) | 0.426 |
| L12H6 zero (control) | monolingual | 1.00 (278) | 1.00 (92) | 0.99 (96) | 0.96 (100) | 0.98 (99) | 0.98 (183) | 0.986 |
| L12H6 zero (control) | crosslingual | 0.96 (245) | 0.95 (256) | 0.88 (252) | 0.83 (221) | 0.88 (264) | 0.88 (254) | 0.896 |
| L12H13 zero (control) | monolingual | 1.00 (286) | 1.00 (92) | 0.98 (95) | 0.97 (100) | 0.97 (100) | 0.97 (189) | 0.982 |
| L12H13 zero (control) | crosslingual | 0.96 (246) | 0.95 (257) | 0.86 (252) | 0.84 (230) | 0.87 (272) | 0.87 (262) | 0.893 |
| L12H14 zero (control) | monolingual | 1.00 (282) | 1.00 (91) | 1.00 (94) | 0.99 (98) | 0.97 (98) | 1.00 (186) | 0.993 |
| L12H14 zero (control) | crosslingual | 0.97 (244) | 0.94 (262) | 0.89 (250) | 0.84 (228) | 0.88 (271) | 0.88 (261) | 0.901 |
