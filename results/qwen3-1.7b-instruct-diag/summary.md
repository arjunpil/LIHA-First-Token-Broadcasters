# qwen3-1.7b-instruct L18H12: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 3 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 55.81, rank 1 of 16 in its layer (layer median 22.27).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 220975 | 0.34 | 0.56 |
| continuation only | 99574 | 0.46 | 0.79 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.64, distance 0.77 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.994 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero | 0.591 | 0.324 | 0.002 | 0.372 | 0.000 |
| mean of all prompt tokens (follow-up) | 0.962 | 0.028 | 0.003 | 0.009 | 0.001 |
| mean of continuation tokens | 0.804 | 0.155 | 0.003 | 0.002 | 0.002 |
| minus the continuation mean | 0.822 | 0.138 | 0.001 | 0.169 | 0.000 |
| own-language mean | 0.997 | 0.000 | 0.003 | 0.001 | 0.000 |
| English mean | 0.160 | 0.667 | 0.000 | 0.834 | 0.000 |
| other-language mean | 0.021 | 0.780 | 0.000 | 0.017 | 0.904 |
| random, same norm | 0.524 | 0.377 | 0.002 | 0.421 | 0.003 |
| x0.5 | 0.984 | 0.009 | 0.001 | 0.010 | 0.000 |
