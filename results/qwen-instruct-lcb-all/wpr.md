# WPR for qwen-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages. In brackets: the number of such replies, which is small where the head's removal leaves few replies in the language.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 0.96 (295) | 0.98 (99) | 0.98 (96) | 0.99 (92) | 0.99 (100) | 0.99 (196) | 0.983 |
| base | crosslingual | 0.93 (215) | 0.99 (222) | 0.94 (160) | 0.97 (147) | 0.96 (214) | 0.98 (242) | 0.959 |
| L22H6 zero | monolingual | 0.95 (287) | 0.89 (38) | 0.98 (94) | 0.99 (73) | 0.99 (99) | 0.99 (196) | 0.965 |
| L22H6 zero | crosslingual | 0.93 (188) | 0.83 (39) | 0.93 (152) | 0.99 (83) | 0.94 (208) | 0.99 (234) | 0.935 |
| L22H6 mean | monolingual | 0.98 (286) | 0.82 (38) | 0.97 (92) | 0.99 (67) | 1.00 (99) | 0.99 (195) | 0.958 |
| L22H6 mean | crosslingual | 0.94 (186) | 0.83 (22) | 0.96 (148) | 1.00 (67) | 0.95 (205) | 0.99 (232) | 0.945 |
| L22H0 zero (control) | monolingual | 0.96 (294) | 0.99 (100) | 0.96 (97) | 0.99 (95) | 0.99 (99) | 0.99 (199) | 0.980 |
| L22H0 zero (control) | crosslingual | 0.93 (218) | 0.99 (217) | 0.91 (174) | 0.96 (148) | 0.96 (213) | 0.98 (241) | 0.953 |
| L22H7 zero (control) | monolingual | 0.95 (296) | 0.99 (100) | 0.98 (94) | 0.96 (91) | 0.97 (99) | 0.99 (198) | 0.974 |
| L22H7 zero (control) | crosslingual | 0.92 (210) | 1.00 (226) | 0.95 (169) | 0.97 (151) | 0.91 (213) | 0.98 (242) | 0.955 |
| L22H11 zero (control) | monolingual | 0.95 (296) | 0.99 (99) | 0.98 (96) | 0.98 (92) | 0.99 (100) | 1.00 (200) | 0.981 |
| L22H11 zero (control) | crosslingual | 0.92 (212) | 1.00 (222) | 0.95 (163) | 0.97 (148) | 0.94 (213) | 0.98 (244) | 0.959 |
