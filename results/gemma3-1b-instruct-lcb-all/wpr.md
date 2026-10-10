# WPR for gemma3-1b-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.00 (299) | 0.99 (100) | 1.00 (97) | 1.00 (100) | 0.99 (100) | 0.98 (185) | 0.993 |
| base | crosslingual | 0.93 (19) | 0.86 (36) | 0.89 (41) | 0.93 (46) | 0.90 (27) | 0.96 (22) | 0.912 |
| L11H3 zero | monolingual | 0.78 (39) | 1.00 (8) | 0.95 (19) | 0.96 (26) | 0.91 (45) | 0.86 (17) | 0.910 |
| L11H3 zero | crosslingual | 0.88 (5) | 1.00 (1) | 0.80 (7) |  | 0.90 (7) | 1.00 (3) | 0.915 |
| L11H3 mean | monolingual | 0.77 (25) | 0.95 (21) | 0.98 (66) | 1.00 (67) | 0.80 (20) | 0.98 (81) | 0.914 |
| L11H3 mean | crosslingual |  | 0.50 (3) | 1.00 (3) |  | 1.00 (3) | 1.00 (1) | 0.875 |
| L11H0 zero (control) | monolingual | 1.00 (300) | 0.98 (99) | 0.98 (98) | 1.00 (100) | 0.99 (100) | 0.99 (183) | 0.989 |
| L11H0 zero (control) | crosslingual | 0.79 (32) | 0.88 (40) | 0.92 (51) | 0.97 (43) | 0.90 (35) | 0.93 (26) | 0.900 |
| L11H1 zero (control) | monolingual | 1.00 (300) | 0.99 (100) | 0.99 (99) | 1.00 (100) | 0.99 (100) | 0.99 (186) | 0.994 |
| L11H1 zero (control) | crosslingual | 0.78 (30) | 0.89 (62) | 0.89 (62) | 0.97 (67) | 0.83 (37) | 0.94 (38) | 0.882 |
| L11H2 zero (control) | monolingual | 1.00 (299) | 0.99 (99) | 0.99 (98) | 0.99 (99) | 0.99 (100) | 0.99 (189) | 0.991 |
| L11H2 zero (control) | crosslingual | 0.85 (13) | 0.81 (30) | 0.90 (40) | 0.93 (37) | 0.86 (22) | 1.00 (14) | 0.891 |
