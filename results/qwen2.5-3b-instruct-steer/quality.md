# qwen2.5-3b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.209 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| base | crosslingual | 1.001 | 0.190 | 0.999 |  /  (0) |  (0) | 3.9 | 3.7 |
| L27H13 steer | monolingual | 0.923 | 0.209 | 1.000 | 0.881 / 0.196 (17) | 0.853 (12) | 3.0 | 3.0 |
| L27H13 steer | crosslingual | 0.890 | 0.191 | 0.997 | 0.719 / 0.152 (54) | 0.692 (31) | 3.9 | 3.8 |
| L27H13 swap | monolingual | 0.695 | 0.165 | 0.997 | 0.695 / 0.165 (782) |  (0) | 3.9 |  |
| L27H13 swap | crosslingual | 0.702 | 0.162 | 0.993 | 0.690 / 0.162 (1045) |  (0) | 4.5 |  |
| L27H13 add steer | monolingual | 0.912 | 0.209 | 1.000 | 0.875 / 0.216 (20) | 0.849 (14) | 3.0 | 3.0 |
| L27H13 add steer | crosslingual | 0.871 | 0.192 | 0.997 | 0.710 / 0.165 (55) | 0.707 (38) | 4.0 | 3.8 |
| L27H13 add swap | monolingual | 0.840 | 0.201 | 1.000 | 0.731 / 0.192 (212) | 0.865 (5) | 3.4 | 3.1 |
| L27H13 add swap | crosslingual | 0.835 | 0.186 | 0.996 | 0.715 / 0.169 (183) | 0.817 (15) | 4.3 | 4.0 |
| L27H6 steer (control) | monolingual | 0.926 | 0.208 | 1.000 | 0.880 / 0.202 (15) | 0.904 (9) | 3.0 | 3.0 |
| L27H6 steer (control) | crosslingual | 0.905 | 0.191 | 0.997 | 0.779 / 0.188 (34) | 0.762 (16) | 4.0 | 3.8 |
| L27H6 swap (control) | monolingual | 0.926 | 0.208 | 1.000 | 0.870 / 0.210 (13) | 0.904 (9) | 3.0 | 3.0 |
| L27H6 swap (control) | crosslingual | 0.906 | 0.191 | 0.997 | 0.779 / 0.191 (35) | 0.757 (18) | 4.0 | 3.8 |
| L27H6 add steer (control) | monolingual | 0.994 | 0.210 | 1.000 | 0.957 / 0.098 (1) | 0.957 (1) | 3.1 | 3.1 |
| L27H6 add steer (control) | crosslingual | 0.994 | 0.190 | 0.999 |  /  (0) |  (0) | 3.9 | 3.8 |
| L27H6 add swap (control) | monolingual | 0.994 | 0.210 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L27H6 add swap (control) | crosslingual | 0.994 | 0.190 | 0.999 | 0.880 / 0.220 (2) |  (0) | 3.9 | 3.7 |
| L27H12 steer (control) | monolingual | 0.940 | 0.209 | 1.000 | 0.884 / 0.195 (10) | 0.881 (4) | 3.1 | 3.1 |
| L27H12 steer (control) | crosslingual | 0.910 | 0.191 | 0.997 | 0.779 / 0.168 (32) | 0.763 (23) | 3.9 | 3.8 |
| L27H12 swap (control) | monolingual | 0.940 | 0.210 | 1.000 | 0.899 / 0.230 (11) | 0.913 (4) | 3.1 | 3.1 |
| L27H12 swap (control) | crosslingual | 0.907 | 0.192 | 0.996 | 0.774 / 0.169 (34) | 0.758 (23) | 3.9 | 3.8 |
| L27H12 add steer (control) | monolingual | 0.987 | 0.209 | 1.000 | 0.967 / 0.184 (2) | 0.938 (1) | 3.1 | 3.1 |
| L27H12 add steer (control) | crosslingual | 0.982 | 0.190 | 0.999 | 0.804 / 0.169 (6) | 0.750 (4) | 3.9 | 3.8 |
| L27H12 add swap (control) | monolingual | 0.983 | 0.210 | 1.000 | 0.955 / 0.127 (1) | 0.955 (1) | 3.1 | 3.1 |
| L27H12 add swap (control) | crosslingual | 0.979 | 0.190 | 0.999 | 0.754 / 0.189 (11) | 0.736 (7) | 3.9 | 3.8 |
| L27H14 steer (control) | monolingual | 0.918 | 0.208 | 0.999 | 0.908 / 0.252 (10) | 0.907 (7) | 3.0 | 3.0 |
| L27H14 steer (control) | crosslingual | 0.898 | 0.192 | 0.996 | 0.761 / 0.173 (40) | 0.749 (31) | 3.8 | 3.7 |
| L27H14 swap (control) | monolingual | 0.916 | 0.207 | 1.000 | 0.897 / 0.227 (11) | 0.885 (7) | 3.0 | 3.0 |
| L27H14 swap (control) | crosslingual | 0.893 | 0.192 | 0.996 | 0.746 / 0.162 (38) | 0.740 (32) | 3.9 | 3.8 |
| L27H14 add steer (control) | monolingual | 0.974 | 0.209 | 1.000 | 0.906 / 0.171 (5) | 0.920 (3) | 3.1 | 3.1 |
| L27H14 add steer (control) | crosslingual | 0.964 | 0.191 | 0.999 | 0.766 / 0.181 (23) | 0.749 (16) | 3.9 | 3.8 |
| L27H14 add swap (control) | monolingual | 0.972 | 0.209 | 1.000 | 0.897 / 0.173 (4) | 0.911 (2) | 3.1 | 3.1 |
| L27H14 add swap (control) | crosslingual | 0.966 | 0.191 | 0.999 | 0.796 / 0.169 (18) | 0.774 (14) | 3.9 | 3.8 |
