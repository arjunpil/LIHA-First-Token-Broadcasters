# gemma3-1b-instruct L11H3: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 4 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 9.45, rank 1 of 4 in its layer (layer median 2.45).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 197543 | 0.76 | 0.03 |
| continuation only | 99306 | 0.86 | 0.11 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.97, distance 3.32 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.997 | 0.000 | 0.000 | 0.003 | 0.000 |
| zero | 0.481 | 0.413 | 0.001 | 0.440 | 0.043 |
| mean of all prompt tokens (follow-up) | 0.887 | 0.089 | 0.002 | 0.009 | 0.028 |
| mean of continuation tokens | 0.753 | 0.196 | 0.002 | 0.013 | 0.128 |
| minus the continuation mean | 0.910 | 0.070 | 0.000 | 0.036 | 0.025 |
| own-language mean | 1.000 | 0.000 | 0.003 | 0.000 | 0.000 |
| English mean | 0.018 | 0.782 | 0.000 | 0.981 | 0.000 |
| other-language mean | 0.000 | 0.977 | 0.000 | 0.002 | 0.997 |
| random, same norm | 0.166 | 0.666 | 0.000 | 0.608 | 0.042 |
| x0.5 | 0.988 | 0.008 | 0.001 | 0.010 | 0.001 |
| zero, norm frozen | 0.796 | 0.160 | 0.000 | 0.116 | 0.024 |
