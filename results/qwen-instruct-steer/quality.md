# qwen-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.209 | 1.000 |  /  (0) |  (0) | 3.5 | 3.5 |
| base | crosslingual | 1.001 | 0.171 | 1.000 |  /  (0) |  (0) | 4.0 | 3.9 |
| L22H6 steer | monolingual | 0.940 | 0.210 | 1.000 | 0.899 / 0.183 (13) | 0.909 (6) | 3.5 | 3.5 |
| L22H6 steer | crosslingual | 0.900 | 0.170 | 0.998 | 0.709 / 0.149 (56) | 0.716 (16) | 4.1 | 3.9 |
| L22H6 swap | monolingual | 0.781 | 0.197 | 0.994 | 0.711 / 0.186 (279) | 0.880 (6) | 4.2 | 4.1 |
| L22H6 swap | crosslingual | 0.782 | 0.163 | 0.987 | 0.673 / 0.157 (391) | 0.572 (9) | 4.3 | 4.2 |
| L22H6 add steer | monolingual | 0.901 | 0.210 | 0.999 | 0.926 / 0.221 (15) | 0.940 (7) | 3.5 | 3.5 |
| L22H6 add steer | crosslingual | 0.878 | 0.170 | 0.999 | 0.698 / 0.141 (84) | 0.762 (17) | 4.0 | 3.9 |
| L22H6 add swap | monolingual | 0.919 | 0.211 | 1.000 | 0.918 / 0.197 (10) | 0.916 (6) | 3.5 | 3.5 |
| L22H6 add swap | crosslingual | 0.902 | 0.171 | 0.997 | 0.685 / 0.156 (43) | 0.682 (14) | 4.1 | 4.0 |
| L22H4 steer (control) | monolingual | 0.921 | 0.210 | 0.999 | 0.907 / 0.239 (8) | 0.889 (6) | 3.5 | 3.5 |
| L22H4 steer (control) | crosslingual | 0.908 | 0.173 | 0.997 | 0.754 / 0.154 (26) | 0.744 (17) | 4.1 | 4.0 |
| L22H4 swap (control) | monolingual | 0.921 | 0.210 | 0.999 | 0.897 / 0.248 (8) | 0.885 (6) | 3.5 | 3.5 |
| L22H4 swap (control) | crosslingual | 0.906 | 0.172 | 0.997 | 0.721 / 0.167 (31) | 0.737 (15) | 4.1 | 4.0 |
| L22H4 add steer (control) | monolingual | 0.988 | 0.210 | 1.000 | 0.936 / 0.128 (2) | 0.994 (1) | 3.5 | 3.5 |
| L22H4 add steer (control) | crosslingual | 0.988 | 0.172 | 0.999 | 0.693 / 0.150 (3) | 0.693 (3) | 4.0 | 3.9 |
| L22H4 add swap (control) | monolingual | 0.986 | 0.209 | 1.000 | 0.877 / 0.126 (1) |  (0) | 3.5 | 3.5 |
| L22H4 add swap (control) | crosslingual | 0.987 | 0.171 | 0.999 | 0.823 / 0.175 (5) |  (0) | 4.1 | 3.9 |
| L22H8 steer (control) | monolingual | 0.964 | 0.209 | 1.000 | 0.948 / 0.201 (7) | 0.947 (5) | 3.5 | 3.5 |
| L22H8 steer (control) | crosslingual | 0.947 | 0.172 | 0.998 | 0.830 / 0.185 (19) | 0.762 (6) | 4.1 | 3.9 |
| L22H8 swap (control) | monolingual | 0.958 | 0.209 | 1.000 | 0.939 / 0.221 (6) | 0.933 (4) | 3.5 | 3.5 |
| L22H8 swap (control) | crosslingual | 0.945 | 0.173 | 0.999 | 0.841 / 0.188 (20) | 0.794 (5) | 4.1 | 3.9 |
| L22H8 add steer (control) | monolingual | 0.984 | 0.209 | 1.000 | 0.882 / 0.238 (3) | 0.994 (1) | 3.5 | 3.5 |
| L22H8 add steer (control) | crosslingual | 0.985 | 0.172 | 0.999 | 0.951 / 0.196 (3) | 0.921 (1) | 4.1 | 3.9 |
| L22H8 add swap (control) | monolingual | 0.983 | 0.209 | 1.000 |  /  (0) |  (0) | 3.5 | 3.5 |
| L22H8 add swap (control) | crosslingual | 0.983 | 0.171 | 0.999 | 0.796 / 0.181 (6) | 0.843 (2) | 4.1 | 4.0 |
| L22H9 steer (control) | monolingual | 0.964 | 0.211 | 1.000 | 0.929 / 0.184 (5) | 0.975 (1) | 3.5 | 3.5 |
| L22H9 steer (control) | crosslingual | 0.951 | 0.172 | 0.999 | 0.735 / 0.171 (15) | 0.711 (9) | 4.1 | 3.9 |
| L22H9 swap (control) | monolingual | 0.964 | 0.211 | 1.000 | 0.926 / 0.160 (7) | 0.951 (3) | 3.5 | 3.5 |
| L22H9 swap (control) | crosslingual | 0.947 | 0.172 | 0.999 | 0.764 / 0.159 (18) | 0.732 (10) | 4.1 | 3.9 |
| L22H9 add steer (control) | monolingual | 0.984 | 0.209 | 1.000 |  /  (0) |  (0) | 3.5 | 3.5 |
| L22H9 add steer (control) | crosslingual | 0.986 | 0.171 | 0.999 | 0.758 / 0.154 (7) | 0.964 (2) | 4.0 | 3.9 |
| L22H9 add swap (control) | monolingual | 0.986 | 0.209 | 1.000 |  /  (0) |  (0) | 3.5 | 3.5 |
| L22H9 add swap (control) | crosslingual | 0.987 | 0.172 | 0.998 | 0.841 / 0.140 (4) | 0.776 (1) | 4.1 | 4.0 |
