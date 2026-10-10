# qwen2.5-7b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.197 | 1.000 |  /  (0) |  (0) | 2.6 | 2.6 |
| base | crosslingual | 1.001 | 0.192 | 1.000 |  /  (0) |  (0) | 3.3 | 3.3 |
| L0H25 steer | monolingual | 0.346 | 0.187 | 0.725 | 0.241 / 0.171 (321) | 0.560 (4) | 4.6 | 4.9 |
| L0H25 steer | crosslingual | 0.508 | 0.189 | 0.878 | 0.421 / 0.177 (524) | 0.530 (8) | 5.8 | 6.0 |
| L0H25 swap | monolingual | 0.295 | 0.185 | 0.671 | 0.229 / 0.173 (357) | 0.704 (2) | 4.3 | 4.6 |
| L0H25 swap | crosslingual | 0.455 | 0.184 | 0.855 | 0.384 / 0.176 (568) | 0.463 (7) | 5.7 | 6.3 |
| L0H25 add steer | monolingual | 0.965 | 0.196 | 1.000 | 0.964 / 0.180 (5) | 0.956 (3) | 2.6 | 2.6 |
| L0H25 add steer | crosslingual | 0.957 | 0.192 | 0.997 | 0.887 / 0.194 (9) | 0.900 (4) | 3.4 | 3.3 |
| L0H25 add swap | monolingual | 0.962 | 0.197 | 1.000 | 0.935 / 0.205 (6) | 0.922 (5) | 2.6 | 2.6 |
| L0H25 add swap | crosslingual | 0.950 | 0.191 | 0.997 | 0.880 / 0.147 (12) | 0.893 (7) | 3.4 | 3.3 |
| L0H12 steer (control) | monolingual | 0.948 | 0.197 | 1.000 | 0.941 / 0.181 (5) | 0.965 (2) | 2.6 | 2.6 |
| L0H12 steer (control) | crosslingual | 0.936 | 0.191 | 0.997 | 0.876 / 0.213 (23) | 0.879 (10) | 3.3 | 3.3 |
| L0H12 swap (control) | monolingual | 0.947 | 0.197 | 1.000 | 0.977 / 0.181 (3) | 0.997 (1) | 2.6 | 2.6 |
| L0H12 swap (control) | crosslingual | 0.935 | 0.191 | 0.997 | 0.837 / 0.186 (16) | 0.856 (7) | 3.3 | 3.3 |
| L0H12 add steer (control) | monolingual | 0.968 | 0.196 | 1.000 | 0.936 / 0.186 (4) | 0.952 (2) | 2.7 | 2.7 |
| L0H12 add steer (control) | crosslingual | 0.958 | 0.192 | 0.997 | 0.860 / 0.179 (10) | 0.857 (6) | 3.3 | 3.3 |
| L0H12 add swap (control) | monolingual | 0.966 | 0.197 | 1.000 | 0.984 / 0.219 (2) |  (0) | 2.6 | 2.7 |
| L0H12 add swap (control) | crosslingual | 0.956 | 0.191 | 0.997 | 0.906 / 0.158 (9) | 0.912 (6) | 3.3 | 3.3 |
| L0H13 steer (control) | monolingual | 0.957 | 0.197 | 0.999 | 0.947 / 0.177 (7) | 0.953 (3) | 2.7 | 2.7 |
| L0H13 steer (control) | crosslingual | 0.946 | 0.191 | 0.998 | 0.853 / 0.252 (17) | 0.869 (8) | 3.3 | 3.3 |
| L0H13 swap (control) | monolingual | 0.957 | 0.197 | 1.000 | 0.960 / 0.179 (9) | 0.963 (2) | 2.6 | 2.6 |
| L0H13 swap (control) | crosslingual | 0.949 | 0.192 | 0.996 | 0.861 / 0.181 (16) | 0.867 (6) | 3.3 | 3.3 |
| L0H13 add steer (control) | monolingual | 0.966 | 0.196 | 0.999 | 0.974 / 0.214 (3) | 0.977 (2) | 2.6 | 2.6 |
| L0H13 add steer (control) | crosslingual | 0.960 | 0.191 | 0.999 | 0.900 / 0.175 (10) | 0.934 (6) | 3.4 | 3.3 |
| L0H13 add swap (control) | monolingual | 0.968 | 0.197 | 0.999 | 0.948 / 0.213 (6) | 0.965 (3) | 2.7 | 2.7 |
| L0H13 add swap (control) | crosslingual | 0.961 | 0.191 | 0.998 | 0.872 / 0.147 (10) | 0.893 (7) | 3.3 | 3.3 |
| L0H24 steer (control) | monolingual | 0.787 | 0.197 | 0.987 | 0.569 / 0.174 (57) | 0.815 (9) | 3.4 | 3.3 |
| L0H24 steer (control) | crosslingual | 0.702 | 0.189 | 0.978 | 0.533 / 0.176 (273) | 0.761 (19) | 4.9 | 4.4 |
| L0H24 swap (control) | monolingual | 0.626 | 0.198 | 0.929 | 0.388 / 0.181 (217) | 0.817 (7) | 5.0 | 4.7 |
| L0H24 swap (control) | crosslingual | 0.661 | 0.188 | 0.965 | 0.525 / 0.186 (368) | 0.714 (14) | 5.8 | 5.3 |
| L0H24 add steer (control) | monolingual | 0.962 | 0.197 | 1.000 | 0.936 / 0.195 (7) | 0.952 (3) | 2.6 | 2.6 |
| L0H24 add steer (control) | crosslingual | 0.955 | 0.192 | 0.997 | 0.874 / 0.175 (13) | 0.877 (7) | 3.3 | 3.3 |
| L0H24 add swap (control) | monolingual | 0.962 | 0.197 | 1.000 | 0.912 / 0.193 (8) | 0.903 (5) | 2.6 | 2.6 |
| L0H24 add swap (control) | crosslingual | 0.950 | 0.191 | 0.998 | 0.877 / 0.149 (16) | 0.886 (9) | 3.3 | 3.3 |
