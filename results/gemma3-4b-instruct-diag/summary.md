# gemma3-4b-instruct L24H0: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 4 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 14.57, rank 2 of 8 in its layer (layer median 6.92).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 197528 | 0.56 | 0.73 |
| continuation only | 99291 | 0.56 | 0.80 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.99, distance 0.13 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero | 0.729 | 0.214 | 0.000 | 0.041 | 0.001 |
| mean of all prompt tokens (follow-up) | 0.983 | 0.012 | 0.001 | 0.003 | 0.001 |
| mean of continuation tokens | 0.984 | 0.011 | 0.001 | 0.003 | 0.002 |
| minus the continuation mean | 0.993 | 0.003 | 0.000 | 0.005 | 0.000 |
| own-language mean | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| English mean | 0.720 | 0.222 | 0.000 | 0.092 | 0.000 |
| other-language mean | 0.002 | 0.796 | 0.000 | 0.005 | 0.989 |
| random, same norm | 0.632 | 0.291 | 0.000 | 0.148 | 0.003 |
| x0.5 | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero, norm frozen | 0.705 | 0.234 | 0.000 | 0.052 | 0.000 |
