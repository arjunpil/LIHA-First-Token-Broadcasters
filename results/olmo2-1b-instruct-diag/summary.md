# olmo2-1b-instruct L12H8: why zero and mean ablation differ

The head's contribution after the output projection, on the user's text and the baseline continuation of 2500 FLORES prompts (the 7 template tokens before the text are left out, since the head's output there is the same for every prompt): mean norm 2.52, rank 2 of 16 in its layer (layer median 0.84).

| tokens | n | mean vector's share of the energy | prompt language's share of the rest |
|---|---|---|---|
| user's text and continuation | 217171 | 0.31 | 0.67 |
| continuation only | 100541 | 0.31 | 0.74 |

The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine 0.96, distance 0.30 of the continuation mean's norm.

Greedy 40 tokens, batched per language in every condition. Language means are over the user's text and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.

| condition | non-English acc | c->w | w->c | non-English replies in English | in the other-language target |
|---|---|---|---|---|---|
| base | 0.997 | 0.000 | 0.000 | 0.002 | 0.000 |
| zero | 0.897 | 0.080 | 0.001 | 0.052 | 0.005 |
| mean of all prompt tokens (follow-up) | 0.928 | 0.056 | 0.002 | 0.006 | 0.015 |
| mean of continuation tokens | 0.930 | 0.054 | 0.002 | 0.006 | 0.014 |
| minus the continuation mean | 0.991 | 0.005 | 0.001 | 0.006 | 0.000 |
| own-language mean | 0.998 | 0.000 | 0.002 | 0.001 | 0.000 |
| English mean | 0.872 | 0.100 | 0.001 | 0.065 | 0.005 |
| other-language mean | 0.059 | 0.750 | 0.000 | 0.021 | 0.910 |
| random, same norm | 0.835 | 0.130 | 0.001 | 0.081 | 0.008 |
| x0.5 | 0.991 | 0.005 | 0.000 | 0.005 | 0.000 |
| zero, norm frozen | 0.906 | 0.074 | 0.002 | 0.040 | 0.005 |
