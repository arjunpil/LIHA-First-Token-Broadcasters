# qwen2.5-7b-instruct L19H1: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 24 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 7.48, rank 2 of 28 in its layer (layer median 3.48).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 211332 | 0.38 | 0.77 |
| continuation only | 99931 | 0.39 | 0.84 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.95, distance 0.44 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero | 0.745 | 0.202 | 0.000 | 0.018 | 0.002 |
| mean of all prompt tokens (follow-up) | 0.972 | 0.021 | 0.001 | 0.006 | 0.001 |
| mean of continuation tokens | 0.990 | 0.008 | 0.001 | 0.005 | 0.002 |
| minus the continuation mean | 0.911 | 0.070 | 0.001 | 0.008 | 0.001 |
| own-language mean | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| English mean | 0.763 | 0.188 | 0.001 | 0.021 | 0.002 |
| other-language mean | 0.562 | 0.352 | 0.001 | 0.013 | 0.417 |
| random, same norm | 0.712 | 0.228 | 0.000 | 0.029 | 0.002 |
| x0.5 | 0.977 | 0.017 | 0.001 | 0.004 | 0.000 |
