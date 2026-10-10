# qwen3-1.7b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.193 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| base | crosslingual | 1.001 | 0.183 | 1.000 |  /  (0) |  (0) | 4.2 | 3.9 |
| L18H12 steer | monolingual | 0.929 | 0.192 | 1.000 | 0.890 / 0.192 (11) | 0.917 (4) | 3.1 | 3.1 |
| L18H12 steer | crosslingual | 0.897 | 0.184 | 0.996 | 0.778 / 0.161 (101) | 0.780 (43) | 4.2 | 3.9 |
| L18H12 swap | monolingual | 0.694 | 0.147 | 0.995 | 0.695 / 0.146 (781) |  (0) | 4.0 | 1.4 |
| L18H12 swap | crosslingual | 0.739 | 0.151 | 0.997 | 0.719 / 0.154 (962) |  (0) | 4.4 | 2.5 |
| L18H12 add steer | monolingual | 0.927 | 0.194 | 1.000 | 0.938 / 0.216 (13) | 0.950 (7) | 3.1 | 3.1 |
| L18H12 add steer | crosslingual | 0.910 | 0.184 | 0.997 | 0.768 / 0.162 (66) | 0.756 (57) | 4.2 | 3.9 |
| L18H12 add swap | monolingual | 0.892 | 0.192 | 1.000 | 0.841 / 0.164 (41) | 0.897 (6) | 3.3 | 3.2 |
| L18H12 add swap | crosslingual | 0.883 | 0.182 | 0.998 | 0.761 / 0.148 (108) | 0.755 (25) | 4.4 | 4.0 |
| L18H6 steer (control) | monolingual | 0.918 | 0.193 | 1.000 | 0.880 / 0.238 (7) | 0.869 (5) | 3.1 | 3.1 |
| L18H6 steer (control) | crosslingual | 0.915 | 0.184 | 0.998 | 0.809 / 0.154 (40) | 0.808 (28) | 4.1 | 3.9 |
| L18H6 swap (control) | monolingual | 0.918 | 0.193 | 1.000 | 0.875 / 0.267 (6) | 0.876 (5) | 3.1 | 3.1 |
| L18H6 swap (control) | crosslingual | 0.916 | 0.184 | 0.998 | 0.815 / 0.155 (42) | 0.822 (28) | 4.2 | 3.9 |
| L18H6 add steer (control) | monolingual | 0.995 | 0.193 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L18H6 add steer (control) | crosslingual | 0.993 | 0.183 | 1.000 | 0.866 / 0.181 (5) | 0.891 (3) | 4.2 | 3.9 |
| L18H6 add swap (control) | monolingual | 0.992 | 0.193 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L18H6 add swap (control) | crosslingual | 0.993 | 0.183 | 1.000 | 0.922 / 0.160 (4) | 0.932 (2) | 4.3 | 3.9 |
| L18H13 steer (control) | monolingual | 0.935 | 0.194 | 1.000 | 0.891 / 0.187 (8) | 0.897 (3) | 3.1 | 3.1 |
| L18H13 steer (control) | crosslingual | 0.927 | 0.185 | 0.999 | 0.785 / 0.172 (60) | 0.773 (54) | 4.3 | 4.0 |
| L18H13 swap (control) | monolingual | 0.927 | 0.194 | 1.000 | 0.867 / 0.227 (10) | 0.919 (3) | 3.1 | 3.1 |
| L18H13 swap (control) | crosslingual | 0.921 | 0.184 | 1.000 | 0.770 / 0.160 (54) | 0.754 (47) | 4.2 | 4.0 |
| L18H13 add steer (control) | monolingual | 0.973 | 0.193 | 1.000 | 0.940 / 0.174 (2) |  (0) | 3.1 | 3.1 |
| L18H13 add steer (control) | crosslingual | 0.967 | 0.184 | 0.999 | 0.814 / 0.176 (24) | 0.808 (21) | 4.2 | 3.9 |
| L18H13 add swap (control) | monolingual | 0.967 | 0.193 | 1.000 | 0.919 / 0.199 (6) | 0.876 (3) | 3.1 | 3.1 |
| L18H13 add swap (control) | crosslingual | 0.966 | 0.183 | 1.000 | 0.833 / 0.176 (19) | 0.824 (17) | 4.2 | 3.9 |
| L18H14 steer (control) | monolingual | 0.910 | 0.194 | 1.000 | 0.900 / 0.210 (12) | 0.915 (5) | 3.0 | 3.0 |
| L18H14 steer (control) | crosslingual | 0.892 | 0.184 | 0.995 | 0.778 / 0.155 (67) | 0.761 (44) | 4.2 | 3.9 |
| L18H14 swap (control) | monolingual | 0.910 | 0.193 | 1.000 | 0.899 / 0.201 (13) | 0.906 (5) | 3.0 | 3.0 |
| L18H14 swap (control) | crosslingual | 0.891 | 0.185 | 0.995 | 0.773 / 0.157 (65) | 0.752 (42) | 4.2 | 3.9 |
| L18H14 add steer (control) | monolingual | 0.995 | 0.192 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L18H14 add steer (control) | crosslingual | 0.995 | 0.183 | 1.000 | 0.920 / 0.136 (2) | 0.920 (2) | 4.2 | 3.9 |
| L18H14 add swap (control) | monolingual | 0.996 | 0.193 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L18H14 add swap (control) | crosslingual | 0.993 | 0.183 | 1.000 | 0.898 / 0.126 (5) | 0.915 (4) | 4.2 | 3.9 |
