# Script switches in the 14-language LCB runs

From the saved replies; nothing regenerated. Switched: more Han/kana characters than characters of the expected script. Mostly Han/kana: Han/kana is the largest script in the reply. Pass rates are simple shares (not averaged over sources like LPR): the first leaves skipped replies out as LPR does, the second counts them as failures. Rows: all Korean rows, and other languages where the ablated condition has at least 3 switched replies (zh, ja left out).

| model | head | lang | task | condition | skipped | switched | switched: skipped / passing / failing | mostly Han/kana | pass rate (skipped left out) | pass rate (skipped = fail) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-instruct | L22H6 | ko | monolingual | base | 3/100 | 1/100 | 0 / 0 / 1 | 1 | 0.95 | 0.92 |
| qwen-instruct | L22H6 | ko | monolingual | L22H6 zero | 16/100 | 17/100 | 11 / 3 / 3 | 17 | 0.87 | 0.73 |
| qwen-instruct | L22H6 | ko | crosslingual | base | 8/299 | 5/299 | 5 / 0 / 0 | 3 | 0.51 | 0.49 |
| qwen-instruct | L22H6 | ko | crosslingual | L22H6 zero | 51/299 | 59/299 | 50 / 5 / 4 | 57 | 0.33 | 0.28 |
| qwen-instruct | L22H6 | ar | monolingual | base | 1/300 | 4/300 | 1 / 2 / 1 | 4 | 0.99 | 0.98 |
| qwen-instruct | L22H6 | ar | monolingual | L22H6 zero | 2/300 | 5/300 | 2 / 2 / 1 | 5 | 0.96 | 0.96 |
| qwen-instruct | L22H6 | ar | crosslingual | base | 10/299 | 6/299 | 6 / 0 / 0 | 6 | 0.74 | 0.72 |
| qwen-instruct | L22H6 | ar | crosslingual | L22H6 zero | 10/299 | 5/299 | 5 / 0 / 0 | 5 | 0.65 | 0.63 |
| qwen-instruct | L22H6 | hi | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 0.99 | 0.99 |
| qwen-instruct | L22H6 | hi | monolingual | L22H6 zero | 2/100 | 8/100 | 2 / 1 / 5 | 4 | 0.39 | 0.38 |
| qwen-instruct | L22H6 | hi | crosslingual | base | 0/299 | 0/299 | 0 / 0 / 0 | 0 | 0.74 | 0.74 |
| qwen-instruct | L22H6 | hi | crosslingual | L22H6 zero | 6/299 | 5/299 | 1 / 0 / 4 | 5 | 0.13 | 0.13 |
| qwen-instruct | L22H6 | tr | monolingual | base | 0/100 | 1/100 | 0 / 0 / 1 | 1 | 0.95 | 0.95 |
| qwen-instruct | L22H6 | tr | monolingual | L22H6 zero | 4/100 | 6/100 | 3 / 0 / 3 | 4 | 0.39 | 0.37 |
| qwen-instruct | L22H6 | tr | crosslingual | base | 8/299 | 1/299 | 1 / 0 / 0 | 1 | 0.56 | 0.55 |
| qwen-instruct | L22H6 | tr | crosslingual | L22H6 zero | 7/299 | 6/299 | 3 / 0 / 3 | 6 | 0.22 | 0.21 |
| qwen-instruct | L22H6 | vi | crosslingual | base | 4/299 | 1/299 | 0 / 0 / 1 | 1 | 0.61 | 0.61 |
| qwen-instruct | L22H6 | vi | crosslingual | L22H6 zero | 6/299 | 8/299 | 4 / 1 / 3 | 5 | 0.10 | 0.10 |
| qwen2.5-3b-instruct | L27H13 | ko | monolingual | base | 3/100 | 1/100 | 1 / 0 / 0 | 1 | 1.00 | 0.97 |
| qwen2.5-3b-instruct | L27H13 | ko | monolingual | L27H13 zero | 91/100 | 95/100 | 90 / 0 / 5 | 95 | 0.00 | 0.00 |
| qwen2.5-3b-instruct | L27H13 | ko | crosslingual | base | 4/299 | 0/299 | 0 / 0 / 0 | 0 | 0.88 | 0.87 |
| qwen2.5-3b-instruct | L27H13 | ko | crosslingual | L27H13 zero | 162/299 | 205/299 | 160 / 14 / 31 | 193 | 0.10 | 0.05 |
| qwen2.5-3b-instruct | L27H13 | ar | monolingual | base | 0/300 | 1/300 | 0 / 0 / 1 | 1 | 0.99 | 0.99 |
| qwen2.5-3b-instruct | L27H13 | ar | monolingual | L27H13 zero | 80/300 | 101/300 | 78 / 12 / 11 | 98 | 0.33 | 0.24 |
| qwen2.5-3b-instruct | L27H13 | ar | crosslingual | base | 8/299 | 2/299 | 2 / 0 / 0 | 2 | 0.93 | 0.91 |
| qwen2.5-3b-instruct | L27H13 | ar | crosslingual | L27H13 zero | 8/299 | 15/299 | 4 / 4 / 7 | 15 | 0.43 | 0.42 |
| qwen2.5-3b-instruct | L27H13 | de | monolingual | base | 0/100 | 1/100 | 0 / 0 / 1 | 1 | 0.99 | 0.99 |
| qwen2.5-3b-instruct | L27H13 | de | monolingual | L27H13 zero | 4/100 | 4/100 | 4 / 0 / 0 | 4 | 0.48 | 0.46 |
| qwen2.5-3b-instruct | L27H13 | es | monolingual | base | 1/300 | 0/300 | 0 / 0 / 0 | 0 | 0.97 | 0.97 |
| qwen2.5-3b-instruct | L27H13 | es | monolingual | L27H13 zero | 8/300 | 7/300 | 7 / 0 / 0 | 7 | 0.83 | 0.81 |
| qwen2.5-3b-instruct | L27H13 | fr | monolingual | base | 1/300 | 0/300 | 0 / 0 / 0 | 0 | 0.99 | 0.99 |
| qwen2.5-3b-instruct | L27H13 | fr | monolingual | L27H13 zero | 8/300 | 8/300 | 7 / 0 / 1 | 7 | 0.20 | 0.19 |
| qwen2.5-3b-instruct | L27H13 | hi | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 0.99 | 0.99 |
| qwen2.5-3b-instruct | L27H13 | hi | monolingual | L27H13 zero | 5/100 | 13/100 | 5 / 1 / 7 | 12 | 0.62 | 0.59 |
| qwen2.5-3b-instruct | L27H13 | hi | crosslingual | base | 3/299 | 0/299 | 0 / 0 / 0 | 0 | 0.89 | 0.88 |
| qwen2.5-3b-instruct | L27H13 | hi | crosslingual | L27H13 zero | 11/299 | 13/299 | 7 / 1 / 5 | 12 | 0.64 | 0.62 |
| qwen2.5-3b-instruct | L27H13 | id | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 0.97 | 0.97 |
| qwen2.5-3b-instruct | L27H13 | id | monolingual | L27H13 zero | 9/100 | 12/100 | 8 / 2 / 2 | 12 | 0.26 | 0.24 |
| qwen2.5-3b-instruct | L27H13 | ru | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| qwen2.5-3b-instruct | L27H13 | ru | monolingual | L27H13 zero | 14/100 | 17/100 | 14 / 2 / 1 | 16 | 0.63 | 0.54 |
| qwen2.5-3b-instruct | L27H13 | ru | crosslingual | base | 7/299 | 0/299 | 0 / 0 / 0 | 0 | 0.91 | 0.89 |
| qwen2.5-3b-instruct | L27H13 | ru | crosslingual | L27H13 zero | 8/299 | 10/299 | 4 / 1 / 5 | 9 | 0.54 | 0.53 |
| qwen2.5-3b-instruct | L27H13 | tr | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| qwen2.5-3b-instruct | L27H13 | tr | monolingual | L27H13 zero | 16/100 | 46/100 | 16 / 11 / 19 | 46 | 0.35 | 0.29 |
| qwen2.5-3b-instruct | L27H13 | tr | crosslingual | base | 7/299 | 1/299 | 1 / 0 / 0 | 1 | 0.83 | 0.81 |
| qwen2.5-3b-instruct | L27H13 | tr | crosslingual | L27H13 zero | 15/299 | 27/299 | 12 / 1 / 14 | 27 | 0.17 | 0.16 |
| qwen2.5-3b-instruct | L27H13 | vi | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| qwen2.5-3b-instruct | L27H13 | vi | monolingual | L27H13 zero | 41/100 | 46/100 | 41 / 1 / 4 | 46 | 0.03 | 0.02 |
| qwen2.5-3b-instruct | L27H13 | vi | crosslingual | base | 5/299 | 0/299 | 0 / 0 / 0 | 0 | 0.82 | 0.80 |
| qwen2.5-3b-instruct | L27H13 | vi | crosslingual | L27H13 zero | 19/299 | 18/299 | 15 / 2 / 1 | 17 | 0.02 | 0.02 |
| qwen3-1.7b-instruct | L18H12 | ko | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| qwen3-1.7b-instruct | L18H12 | ko | monolingual | L18H12 zero | 24/100 | 63/100 | 23 / 17 / 23 | 61 | 0.39 | 0.30 |
| qwen3-1.7b-instruct | L18H12 | ko | crosslingual | base | 8/299 | 0/299 | 0 / 0 / 0 | 0 | 0.73 | 0.71 |
| qwen3-1.7b-instruct | L18H12 | ko | crosslingual | L18H12 zero | 10/299 | 9/299 | 4 / 0 / 5 | 7 | 0.00 | 0.00 |
| qwen3-1.7b-instruct | L18H12 | ar | crosslingual | base | 5/299 | 1/299 | 0 / 0 / 1 | 0 | 0.83 | 0.82 |
| qwen3-1.7b-instruct | L18H12 | ar | crosslingual | L18H12 zero | 5/299 | 3/299 | 1 / 0 / 2 | 2 | 0.07 | 0.07 |
| olmo2-1b-instruct | L12H8 | ko | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| olmo2-1b-instruct | L12H8 | ko | monolingual | L12H8 zero | 9/100 | 13/100 | 3 / 6 / 4 | 12 | 0.56 | 0.51 |
| olmo2-1b-instruct | L12H8 | ko | crosslingual | base | 8/299 | 1/299 | 1 / 0 / 0 | 1 | 0.76 | 0.74 |
| olmo2-1b-instruct | L12H8 | ko | crosslingual | L12H8 zero | 20/299 | 36/299 | 9 / 8 / 19 | 21 | 0.32 | 0.30 |
| gemma3-1b-instruct | L11H3 | ko | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 1.00 | 1.00 |
| gemma3-1b-instruct | L11H3 | ko | monolingual | L11H3 zero | 0/100 | 5/100 | 0 / 0 / 5 | 2 | 0.26 | 0.26 |
| gemma3-1b-instruct | L11H3 | ko | crosslingual | base | 6/299 | 0/299 | 0 / 0 / 0 | 0 | 0.16 | 0.15 |
| gemma3-1b-instruct | L11H3 | ko | crosslingual | L11H3 zero | 8/299 | 6/299 | 0 / 0 / 6 | 0 | 0.00 | 0.00 |
| llama3.2-1b-instruct | L8H25 | ko | monolingual | base | 0/100 | 0/100 | 0 / 0 / 0 | 0 | 0.99 | 0.99 |
| llama3.2-1b-instruct | L8H25 | ko | monolingual | L8H25 zero | 2/100 | 5/100 | 1 / 1 / 3 | 5 | 0.60 | 0.59 |
| llama3.2-1b-instruct | L8H25 | ko | crosslingual | base | 7/299 | 0/299 | 0 / 0 / 0 | 0 | 0.58 | 0.57 |
| llama3.2-1b-instruct | L8H25 | ko | crosslingual | L8H25 zero | 0/299 | 0/299 | 0 / 0 / 0 | 0 | 0.00 | 0.00 |
