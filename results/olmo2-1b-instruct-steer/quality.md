# olmo2-1b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.198 | 1.000 |  /  (0) |  (0) | 5.7 | 5.7 |
| base | crosslingual | 1.001 | 0.193 | 1.000 |  /  (0) |  (0) | 8.3 | 7.8 |
| L12H8 steer | monolingual | 0.899 | 0.198 | 1.000 | 0.775 / 0.203 (13) | 0.719 (8) | 5.7 | 5.7 |
| L12H8 steer | crosslingual | 0.874 | 0.197 | 0.997 | 0.799 / 0.189 (40) | 0.807 (20) | 8.2 | 7.8 |
| L12H8 swap | monolingual | 0.667 | 0.163 | 0.996 | 0.666 / 0.162 (775) |  (0) | 7.5 | 20.3 |
| L12H8 swap | crosslingual | 0.684 | 0.167 | 0.991 | 0.678 / 0.168 (1085) |  (0) | 10.1 | 69.5 |
| L12H8 add steer | monolingual | 0.876 | 0.200 | 1.000 | 0.681 / 0.216 (10) | 0.667 (9) | 5.9 | 5.9 |
| L12H8 add steer | crosslingual | 0.860 | 0.194 | 0.997 | 0.820 / 0.142 (27) | 0.871 (15) | 8.3 | 8.0 |
| L12H8 add swap | monolingual | 0.836 | 0.197 | 1.000 | 0.742 / 0.175 (20) | 0.723 (5) | 6.3 | 6.3 |
| L12H8 add swap | crosslingual | 0.813 | 0.194 | 0.991 | 0.739 / 0.181 (102) | 0.829 (19) | 9.2 | 8.3 |
| L12H0 steer (control) | monolingual | 0.963 | 0.198 | 1.000 | 0.493 / 0.256 (1) | 0.493 (1) | 5.7 | 5.7 |
| L12H0 steer (control) | crosslingual | 0.911 | 0.194 | 0.997 | 0.787 / 0.183 (28) | 0.821 (16) | 8.2 | 7.8 |
| L12H0 swap (control) | monolingual | 0.962 | 0.198 | 1.000 | 0.493 / 0.256 (1) | 0.493 (1) | 5.7 | 5.7 |
| L12H0 swap (control) | crosslingual | 0.911 | 0.194 | 0.997 | 0.786 / 0.181 (26) | 0.810 (15) | 8.1 | 7.8 |
| L12H0 add steer (control) | monolingual | 0.996 | 0.197 | 1.000 |  /  (0) |  (0) | 5.7 | 5.7 |
| L12H0 add steer (control) | crosslingual | 0.995 | 0.193 | 1.000 | 0.983 / 0.126 (1) |  (0) | 8.2 | 7.9 |
| L12H0 add swap (control) | monolingual | 0.994 | 0.197 | 1.000 | 0.928 / 0.170 (1) |  (0) | 5.7 | 5.7 |
| L12H0 add swap (control) | crosslingual | 0.993 | 0.193 | 1.000 | 0.786 / 0.213 (3) | 0.827 (1) | 8.2 | 7.8 |
| L12H6 steer (control) | monolingual | 0.954 | 0.197 | 1.000 | 0.764 / 0.268 (2) | 0.764 (2) | 5.8 | 5.8 |
| L12H6 steer (control) | crosslingual | 0.939 | 0.194 | 0.998 | 0.840 / 0.166 (17) | 0.834 (10) | 8.2 | 7.8 |
| L12H6 swap (control) | monolingual | 0.950 | 0.197 | 1.000 | 0.764 / 0.268 (2) | 0.764 (2) | 5.8 | 5.8 |
| L12H6 swap (control) | crosslingual | 0.937 | 0.194 | 0.997 | 0.846 / 0.169 (18) | 0.834 (10) | 8.5 | 7.9 |
| L12H6 add steer (control) | monolingual | 0.997 | 0.198 | 1.000 | 0.928 / 0.170 (1) |  (0) | 5.7 | 5.7 |
| L12H6 add steer (control) | crosslingual | 0.994 | 0.193 | 0.999 | 0.765 / 0.288 (2) |  (0) | 8.2 | 7.8 |
| L12H6 add swap (control) | monolingual | 0.994 | 0.198 | 1.000 |  /  (0) |  (0) | 5.7 | 5.7 |
| L12H6 add swap (control) | crosslingual | 0.995 | 0.193 | 1.000 | 0.900 / 0.115 (2) | 0.946 (1) | 8.3 | 7.9 |
| L12H15 steer (control) | monolingual | 0.969 | 0.198 | 1.000 | 0.931 / 0.224 (6) | 0.890 (3) | 5.6 | 5.6 |
| L12H15 steer (control) | crosslingual | 0.957 | 0.194 | 1.000 | 0.849 / 0.191 (18) | 0.883 (8) | 8.3 | 7.8 |
| L12H15 swap (control) | monolingual | 0.964 | 0.198 | 1.000 | 0.931 / 0.224 (6) | 0.890 (3) | 5.6 | 5.6 |
| L12H15 swap (control) | crosslingual | 0.954 | 0.194 | 1.000 | 0.848 / 0.195 (21) | 0.851 (11) | 8.2 | 7.8 |
| L12H15 add steer (control) | monolingual | 0.992 | 0.197 | 1.000 |  /  (0) |  (0) | 5.7 | 5.7 |
| L12H15 add steer (control) | crosslingual | 0.990 | 0.193 | 0.999 | 0.782 / 0.165 (4) | 0.799 (2) | 8.3 | 7.8 |
| L12H15 add swap (control) | monolingual | 0.988 | 0.198 | 1.000 | 0.795 / 0.304 (1) | 0.795 (1) | 5.7 | 5.7 |
| L12H15 add swap (control) | crosslingual | 0.989 | 0.193 | 1.000 | 0.904 / 0.129 (4) | 0.889 (2) | 8.2 | 7.9 |
