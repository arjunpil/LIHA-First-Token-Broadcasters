# qwen2.5-7b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.197 | 1.000 |  /  (0) |  (0) | 2.6 | 2.6 |
| base | crosslingual | 1.001 | 0.192 | 1.000 |  /  (0) |  (0) | 3.3 | 3.3 |
| L19H1 steer | monolingual | 0.952 | 0.197 | 1.000 | 0.951 / 0.182 (8) | 0.945 (5) | 2.7 | 2.7 |
| L19H1 steer | crosslingual | 0.921 | 0.192 | 0.997 | 0.864 / 0.203 (25) | 0.861 (11) | 3.3 | 3.3 |
| L19H1 swap | monolingual | 0.856 | 0.190 | 1.000 | 0.796 / 0.173 (237) | 0.887 (7) | 3.3 | 3.1 |
| L19H1 swap | crosslingual | 0.835 | 0.189 | 0.997 | 0.780 / 0.165 (329) | 0.865 (6) | 4.6 | 4.1 |
| L19H1 add steer | monolingual | 0.935 | 0.197 | 1.000 | 0.925 / 0.206 (11) | 0.945 (4) | 2.6 | 2.6 |
| L19H1 add steer | crosslingual | 0.915 | 0.191 | 0.995 | 0.860 / 0.145 (23) | 0.862 (14) | 3.4 | 3.3 |
| L19H1 add swap | monolingual | 0.921 | 0.197 | 0.999 | 0.916 / 0.198 (9) | 0.913 (7) | 2.7 | 2.7 |
| L19H1 add swap | crosslingual | 0.901 | 0.191 | 0.997 | 0.863 / 0.211 (27) | 0.886 (9) | 3.4 | 3.3 |
| L19H13 steer (control) | monolingual | 0.951 | 0.197 | 1.000 | 0.925 / 0.207 (7) | 0.882 (3) | 2.6 | 2.6 |
| L19H13 steer (control) | crosslingual | 0.936 | 0.191 | 1.000 | 0.858 / 0.180 (15) | 0.856 (8) | 3.4 | 3.3 |
| L19H13 swap (control) | monolingual | 0.949 | 0.198 | 1.000 | 0.942 / 0.206 (8) | 0.928 (4) | 2.6 | 2.6 |
| L19H13 swap (control) | crosslingual | 0.939 | 0.191 | 0.998 | 0.868 / 0.164 (15) | 0.859 (8) | 3.4 | 3.3 |
| L19H13 add steer (control) | monolingual | 0.974 | 0.196 | 1.000 | 0.899 / 0.224 (3) | 0.929 (2) | 2.6 | 2.6 |
| L19H13 add steer (control) | crosslingual | 0.970 | 0.191 | 0.997 | 0.861 / 0.141 (9) | 0.915 (5) | 3.3 | 3.3 |
| L19H13 add swap (control) | monolingual | 0.974 | 0.196 | 1.000 | 0.942 / 0.210 (3) | 0.997 (1) | 2.7 | 2.7 |
| L19H13 add swap (control) | crosslingual | 0.971 | 0.191 | 1.000 | 0.828 / 0.133 (6) | 0.820 (2) | 3.3 | 3.3 |
| L19H14 steer (control) | monolingual | 0.952 | 0.196 | 1.000 | 0.898 / 0.220 (8) | 0.899 (5) | 2.6 | 2.6 |
| L19H14 steer (control) | crosslingual | 0.933 | 0.190 | 0.997 | 0.847 / 0.134 (14) | 0.885 (5) | 3.3 | 3.3 |
| L19H14 swap (control) | monolingual | 0.954 | 0.197 | 1.000 | 0.886 / 0.238 (7) | 0.888 (4) | 2.7 | 2.7 |
| L19H14 swap (control) | crosslingual | 0.931 | 0.190 | 0.997 | 0.841 / 0.156 (14) | 0.859 (5) | 3.3 | 3.3 |
| L19H14 add steer (control) | monolingual | 0.974 | 0.196 | 1.000 |  /  (0) |  (0) | 2.6 | 2.7 |
| L19H14 add steer (control) | crosslingual | 0.970 | 0.191 | 0.999 | 0.849 / 0.173 (11) | 0.869 (6) | 3.3 | 3.3 |
| L19H14 add swap (control) | monolingual | 0.978 | 0.196 | 1.000 |  /  (0) |  (0) | 2.6 | 2.6 |
| L19H14 add swap (control) | crosslingual | 0.969 | 0.191 | 0.999 | 0.873 / 0.184 (6) | 0.839 (3) | 3.3 | 3.3 |
| L19H25 steer (control) | monolingual | 0.951 | 0.197 | 1.000 | 0.917 / 0.207 (7) | 0.930 (5) | 2.6 | 2.6 |
| L19H25 steer (control) | crosslingual | 0.938 | 0.191 | 0.999 | 0.868 / 0.167 (12) | 0.894 (7) | 3.4 | 3.3 |
| L19H25 swap (control) | monolingual | 0.950 | 0.197 | 1.000 | 0.941 / 0.211 (8) | 0.916 (4) | 2.6 | 2.6 |
| L19H25 swap (control) | crosslingual | 0.937 | 0.191 | 0.998 | 0.873 / 0.164 (16) | 0.912 (8) | 3.4 | 3.3 |
| L19H25 add steer (control) | monolingual | 0.975 | 0.197 | 1.000 | 0.934 / 0.226 (3) | 0.982 (2) | 2.6 | 2.6 |
| L19H25 add steer (control) | crosslingual | 0.970 | 0.191 | 0.999 | 0.854 / 0.133 (4) | 0.911 (3) | 3.3 | 3.3 |
| L19H25 add swap (control) | monolingual | 0.975 | 0.197 | 1.000 | 0.869 / 0.252 (2) | 0.897 (1) | 2.6 | 2.6 |
| L19H25 add swap (control) | crosslingual | 0.971 | 0.191 | 0.999 | 0.882 / 0.141 (6) | 0.820 (2) | 3.3 | 3.3 |
