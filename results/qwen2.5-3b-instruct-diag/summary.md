# qwen2.5-3b-instruct L27H13: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 24 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 9.76, rank 2 of 16 in its layer (layer median 4.73).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 211367 | 0.38 | 0.71 |
| continuation only | 99966 | 0.48 | 0.81 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.88, distance 0.55 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.992 | 0.000 | 0.000 | 0.005 | 0.001 |
| zero | 0.349 | 0.515 | 0.001 | 0.262 | 0.005 |
| mean of all prompt tokens (follow-up) | 0.571 | 0.338 | 0.002 | 0.045 | 0.054 |
| mean of continuation tokens | 0.528 | 0.372 | 0.002 | 0.011 | 0.072 |
| minus the continuation mean | 0.919 | 0.059 | 0.000 | 0.062 | 0.000 |
| own-language mean | 0.997 | 0.001 | 0.005 | 0.003 | 0.000 |
| English mean | 0.248 | 0.595 | 0.000 | 0.433 | 0.006 |
| other-language mean | 0.000 | 0.794 | 0.000 | 0.004 | 0.996 |
| random, same norm | 0.197 | 0.636 | 0.000 | 0.342 | 0.006 |
| x0.5 | 0.934 | 0.047 | 0.002 | 0.019 | 0.001 |
