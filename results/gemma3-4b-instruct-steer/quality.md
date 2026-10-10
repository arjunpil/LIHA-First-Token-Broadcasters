# gemma3-4b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.207 | 1.000 |  /  (0) |  (0) | 2.9 | 2.9 |
| base | crosslingual | 1.001 | 0.224 | 1.000 |  /  (0) |  (0) | 4.7 | 8.3 |
| L24H0 steer | monolingual | 0.959 | 0.207 | 1.000 | 0.888 / 0.193 (14) | 0.959 (4) | 2.9 | 2.9 |
| L24H0 steer | crosslingual | 0.923 | 0.225 | 0.998 | 0.841 / 0.205 (102) | 0.841 (98) | 4.9 | 7.0 |
| L24H0 swap | monolingual | 0.717 | 0.152 | 0.996 | 0.719 / 0.152 (789) |  (0) | 3.3 | 2.4 |
| L24H0 swap | crosslingual | 0.834 | 0.201 | 0.997 | 0.788 / 0.192 (155) |  (0) | 6.8 |  |
| L24H0 add steer | monolingual | 0.945 | 0.206 | 1.000 | 0.934 / 0.197 (6) | 0.923 (4) | 3.0 | 3.0 |
| L24H0 add steer | crosslingual | 0.927 | 0.224 | 0.999 | 0.852 / 0.216 (54) | 0.851 (51) | 4.8 | 7.4 |
| L24H0 add swap | monolingual | 0.900 | 0.204 | 0.999 | 0.853 / 0.198 (152) | 0.803 (4) | 3.6 | 3.4 |
| L24H0 add swap | crosslingual | 0.904 | 0.221 | 0.996 | 0.817 / 0.194 (95) | 0.839 (22) | 5.8 | 8.1 |
| L24H4 steer (control) | monolingual | 0.944 | 0.207 | 1.000 | 0.921 / 0.218 (9) | 0.881 (5) | 2.9 | 2.9 |
| L24H4 steer (control) | crosslingual | 0.937 | 0.224 | 0.999 | 0.835 / 0.184 (27) | 0.858 (5) | 4.8 | 8.6 |
| L24H4 swap (control) | monolingual | 0.944 | 0.207 | 1.000 | 0.925 / 0.203 (10) | 0.896 (5) | 2.9 | 2.9 |
| L24H4 swap (control) | crosslingual | 0.937 | 0.224 | 0.999 | 0.835 / 0.187 (27) | 0.858 (5) | 4.8 | 8.7 |
| L24H4 add steer (control) | monolingual | 0.998 | 0.207 | 1.000 | 0.965 / 0.193 (2) | 0.965 (2) | 2.9 | 2.9 |
| L24H4 add steer (control) | crosslingual | 0.996 | 0.224 | 1.000 |  /  (0) |  (0) | 4.7 | 8.2 |
| L24H4 add swap (control) | monolingual | 0.997 | 0.207 | 1.000 | 0.968 / 0.171 (1) | 0.968 (1) | 2.9 | 2.9 |
| L24H4 add swap (control) | crosslingual | 0.997 | 0.224 | 1.000 | 0.802 / 0.271 (2) |  (0) | 4.7 | 8.3 |
| L24H6 steer (control) | monolingual | 0.927 | 0.206 | 0.999 | 0.907 / 0.208 (11) | 0.903 (5) | 3.0 | 3.0 |
| L24H6 steer (control) | crosslingual | 0.926 | 0.224 | 0.997 | 0.835 / 0.221 (42) | 0.824 (26) | 4.8 | 7.8 |
| L24H6 swap (control) | monolingual | 0.928 | 0.205 | 0.999 | 0.912 / 0.206 (12) | 0.902 (5) | 3.0 | 3.0 |
| L24H6 swap (control) | crosslingual | 0.927 | 0.224 | 0.997 | 0.834 / 0.219 (41) | 0.825 (25) | 4.8 | 8.0 |
| L24H6 add steer (control) | monolingual | 0.996 | 0.207 | 1.000 |  /  (0) |  (0) | 2.9 | 2.9 |
| L24H6 add steer (control) | crosslingual | 0.992 | 0.224 | 0.999 | 0.855 / 0.226 (7) | 0.852 (6) | 4.7 | 8.2 |
| L24H6 add swap (control) | monolingual | 0.994 | 0.207 | 1.000 | 0.977 / 0.230 (2) | 0.962 (1) | 2.9 | 2.9 |
| L24H6 add swap (control) | crosslingual | 0.992 | 0.224 | 0.999 | 0.864 / 0.222 (7) | 0.852 (6) | 4.7 | 8.3 |
| L24H7 steer (control) | monolingual | 0.965 | 0.207 | 1.000 | 0.938 / 0.214 (7) | 0.952 (2) | 2.9 | 2.9 |
| L24H7 steer (control) | crosslingual | 0.958 | 0.223 | 0.998 | 0.827 / 0.190 (31) | 0.848 (2) | 4.7 | 8.3 |
| L24H7 swap (control) | monolingual | 0.964 | 0.207 | 1.000 | 0.938 / 0.214 (7) | 0.952 (2) | 2.9 | 2.9 |
| L24H7 swap (control) | crosslingual | 0.959 | 0.224 | 0.998 | 0.835 / 0.187 (27) | 0.965 (2) | 4.7 | 8.4 |
| L24H7 add steer (control) | monolingual | 0.997 | 0.207 | 1.000 | 0.965 / 0.193 (2) | 0.965 (2) | 2.9 | 2.9 |
| L24H7 add steer (control) | crosslingual | 0.995 | 0.224 | 1.000 | 0.790 / 0.238 (2) |  (0) | 4.7 | 8.3 |
| L24H7 add swap (control) | monolingual | 0.996 | 0.207 | 1.000 | 0.950 / 0.170 (1) | 0.950 (1) | 2.9 | 2.9 |
| L24H7 add swap (control) | crosslingual | 0.993 | 0.224 | 1.000 | 0.825 / 0.210 (3) |  (0) | 4.7 | 8.3 |
