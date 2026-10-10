# qwen2.5-7b-instruct L0H25: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 24 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 0.47, rank 25 of 28 in its layer (layer median 0.98).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 211332 | 0.02 | 0.11 |
| continuation only | 99931 | 0.03 | 0.15 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.87, distance 0.54 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero | 0.335 | 0.629 | 0.000 | 0.394 | 0.001 |
| mean of all prompt tokens (follow-up) | 0.254 | 0.704 | 0.000 | 0.411 | 0.004 |
| mean of continuation tokens | 0.316 | 0.636 | 0.000 | 0.393 | 0.003 |
| minus the continuation mean | 0.996 | 0.001 | 0.000 | 0.003 | 0.000 |
| own-language mean | 0.305 | 0.653 | 0.000 | 0.388 | 0.003 |
| English mean | 0.295 | 0.661 | 0.000 | 0.439 | 0.001 |
| other-language mean | 0.269 | 0.704 | 0.000 | 0.367 | 0.003 |
| random, same norm | 0.264 | 0.707 | 0.000 | 0.262 | 0.002 |
| x0.5 | 0.009 | 0.988 | 0.000 | 0.041 | 0.004 |
