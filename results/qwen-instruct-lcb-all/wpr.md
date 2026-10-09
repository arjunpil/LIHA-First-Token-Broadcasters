# WPR for qwen-instruct-lcb-all

Share of the replies without line errors that contain no English word, averaged over sources per language, then over languages.

| condition | task | ar | hi | ja | ko | ru | zh | mean |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 0.96 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.983 |
| base | crosslingual | 0.93 | 0.99 | 0.94 | 0.97 | 0.96 | 0.98 | 0.959 |
| L22H6 zero | monolingual | 0.95 | 0.89 | 0.98 | 0.99 | 0.99 | 0.99 | 0.965 |
| L22H6 zero | crosslingual | 0.93 | 0.83 | 0.93 | 0.99 | 0.94 | 0.99 | 0.935 |
| L22H6 mean | monolingual | 0.98 | 0.82 | 0.97 | 0.99 | 1.00 | 0.99 | 0.958 |
| L22H6 mean | crosslingual | 0.94 | 0.83 | 0.96 | 1.00 | 0.95 | 0.99 | 0.945 |
| L22H0 zero (control) | monolingual | 0.96 | 0.99 | 0.96 | 0.99 | 0.99 | 0.99 | 0.980 |
| L22H0 zero (control) | crosslingual | 0.93 | 0.99 | 0.91 | 0.96 | 0.96 | 0.98 | 0.953 |
| L22H7 zero (control) | monolingual | 0.95 | 0.99 | 0.98 | 0.96 | 0.97 | 0.99 | 0.974 |
| L22H7 zero (control) | crosslingual | 0.92 | 1.00 | 0.95 | 0.97 | 0.91 | 0.98 | 0.955 |
| L22H11 zero (control) | monolingual | 0.95 | 0.99 | 0.98 | 0.98 | 0.99 | 1.00 | 0.981 |
| L22H11 zero (control) | crosslingual | 0.92 | 1.00 | 0.95 | 0.97 | 0.94 | 0.98 | 0.959 |
