# qwen-instruct L22H6: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 24 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 7.47, rank 1 of 12 in its layer (layer median 4.15).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 207728 | 0.18 | 0.62 |
| continuation only | 96327 | 0.19 | 0.71 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.89, distance 0.46 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.902 | 0.000 | 0.000 | 0.097 | 0.000 |
| zero | 0.277 | 0.501 | 0.000 | 0.553 | 0.001 |
| mean of all prompt tokens (follow-up) | 0.341 | 0.451 | 0.001 | 0.476 | 0.002 |
| mean of continuation tokens | 0.365 | 0.431 | 0.001 | 0.447 | 0.003 |
| minus the continuation mean | 0.870 | 0.027 | 0.001 | 0.129 | 0.000 |
| own-language mean | 0.841 | 0.052 | 0.003 | 0.158 | 0.000 |
| English mean | 0.296 | 0.486 | 0.001 | 0.543 | 0.001 |
| other-language mean | 0.240 | 0.530 | 0.001 | 0.417 | 0.182 |
| random, same norm | 0.223 | 0.544 | 0.001 | 0.569 | 0.004 |
| x0.5 | 0.817 | 0.071 | 0.002 | 0.177 | 0.000 |
