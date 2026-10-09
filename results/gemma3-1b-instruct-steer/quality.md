# gemma3-1b-instruct: quality of the steered replies

Content: cosine similarity (Qwen/Qwen3-Embedding-0.6B) between a reply and the baseline reply to the same prompt, against the baseline reply to another prompt of the same task and language (one fixed pairing for every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored non-English replies and over those entirely in the expected language.

| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | fixed: same (n) | PPL, all | PPL, in expected language |
|---|---|---|---|---|---|---|---|---|
| base | monolingual | 1.001 | 0.207 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| base | crosslingual | 1.001 | 0.201 | 1.000 |  /  (0) |  (0) | 5.1 | 5.8 |
| L11H3 steer | monolingual | 0.882 | 0.205 | 1.000 | 0.838 / 0.224 (14) | 0.826 (12) | 3.3 | 3.3 |
| L11H3 steer | crosslingual | 0.809 | 0.208 | 0.993 | 0.763 / 0.199 (544) | 0.763 (530) | 4.7 | 4.9 |
| L11H3 swap | monolingual | 0.703 | 0.151 | 0.995 | 0.704 / 0.152 (786) |  (0) | 3.4 |  |
| L11H3 swap | crosslingual | 0.752 | 0.182 | 0.997 | 0.739 / 0.174 (136) |  (0) | 6.3 |  |
| L11H3 add steer | monolingual | 0.931 | 0.207 | 0.999 | 0.858 / 0.227 (22) | 0.886 (5) | 3.3 | 3.3 |
| L11H3 add steer | crosslingual | 0.887 | 0.203 | 0.998 | 0.803 / 0.194 (109) | 0.797 (82) | 5.4 | 5.7 |
| L11H3 add swap | monolingual | 0.735 | 0.164 | 0.986 | 0.710 / 0.157 (692) |  (0) | 4.1 | 2.8 |
| L11H3 add swap | crosslingual | 0.866 | 0.195 | 0.999 | 0.787 / 0.187 (140) | 0.792 (29) | 6.5 | 5.0 |
| L11H0 steer (control) | monolingual | 0.942 | 0.207 | 0.999 | 0.817 / 0.232 (10) | 0.841 (4) | 3.1 | 3.1 |
| L11H0 steer (control) | crosslingual | 0.902 | 0.204 | 1.000 | 0.816 / 0.200 (66) | 0.811 (53) | 5.1 | 5.7 |
| L11H0 swap (control) | monolingual | 0.944 | 0.207 | 0.999 | 0.845 / 0.212 (13) | 0.875 (6) | 3.1 | 3.1 |
| L11H0 swap (control) | crosslingual | 0.904 | 0.203 | 0.999 | 0.816 / 0.204 (59) | 0.814 (48) | 5.1 | 5.4 |
| L11H0 add steer (control) | monolingual | 0.986 | 0.206 | 1.000 | 0.936 / 0.188 (1) |  (0) | 3.1 | 3.1 |
| L11H0 add steer (control) | crosslingual | 0.967 | 0.202 | 0.999 | 0.838 / 0.214 (28) | 0.836 (25) | 5.1 | 5.9 |
| L11H0 add swap (control) | monolingual | 0.985 | 0.207 | 1.000 | 0.879 / 0.236 (2) | 0.941 (1) | 3.1 | 3.1 |
| L11H0 add swap (control) | crosslingual | 0.965 | 0.202 | 1.000 | 0.842 / 0.200 (23) | 0.828 (20) | 5.1 | 5.6 |
| L11H1 steer (control) | monolingual | 0.933 | 0.207 | 1.000 | 0.849 / 0.189 (13) | 0.810 (8) | 3.2 | 3.1 |
| L11H1 steer (control) | crosslingual | 0.884 | 0.204 | 0.998 | 0.783 / 0.200 (37) | 0.787 (17) | 5.2 | 5.1 |
| L11H1 swap (control) | monolingual | 0.933 | 0.206 | 1.000 | 0.841 / 0.197 (12) | 0.812 (8) | 3.2 | 3.2 |
| L11H1 swap (control) | crosslingual | 0.884 | 0.204 | 0.997 | 0.773 / 0.198 (37) | 0.776 (17) | 5.2 | 5.3 |
| L11H1 add steer (control) | monolingual | 0.994 | 0.206 | 1.000 |  /  (0) |  (0) | 3.1 | 3.1 |
| L11H1 add steer (control) | crosslingual | 0.986 | 0.201 | 1.000 | 0.873 / 0.226 (3) |  (0) | 5.0 | 5.9 |
| L11H1 add swap (control) | monolingual | 0.994 | 0.207 | 1.000 | 0.875 / 0.272 (2) | 0.934 (1) | 3.1 | 3.1 |
| L11H1 add swap (control) | crosslingual | 0.985 | 0.200 | 1.000 | 0.625 / 0.041 (1) |  (0) | 5.1 | 5.8 |
| L11H2 steer (control) | monolingual | 0.890 | 0.206 | 1.000 | 0.804 / 0.200 (11) | 0.777 (8) | 3.3 | 3.3 |
| L11H2 steer (control) | crosslingual | 0.824 | 0.195 | 0.997 | 0.768 / 0.193 (66) | 0.776 (14) | 5.4 | 5.3 |
| L11H2 swap (control) | monolingual | 0.887 | 0.205 | 1.000 | 0.791 / 0.207 (11) | 0.757 (8) | 3.3 | 3.3 |
| L11H2 swap (control) | crosslingual | 0.827 | 0.194 | 0.997 | 0.770 / 0.192 (64) | 0.804 (12) | 5.4 | 5.6 |
| L11H2 add steer (control) | monolingual | 0.984 | 0.206 | 1.000 | 0.894 / 0.200 (5) | 0.940 (3) | 3.1 | 3.1 |
| L11H2 add steer (control) | crosslingual | 0.964 | 0.200 | 1.000 | 0.830 / 0.182 (9) | 0.946 (1) | 5.1 | 5.8 |
| L11H2 add swap (control) | monolingual | 0.981 | 0.206 | 1.000 |  /  (0) |  (0) | 3.2 | 3.1 |
| L11H2 add swap (control) | crosslingual | 0.963 | 0.201 | 1.000 | 0.877 / 0.224 (11) | 0.870 (2) | 5.1 | 5.7 |
