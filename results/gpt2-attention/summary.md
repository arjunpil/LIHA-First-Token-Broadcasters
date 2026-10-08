# L6H10 versions of the section 5 / appendix B numbers

500 prompts (100 per language), greedy, 40 steps. First-token attention and entropy are at the newest position at each generation step, averaged over steps and prompts.

Table 6 replacement:

| head | c->w | first-token attn | entropy |
|---|---|---|---|
| L6H10 | 0.210 | 0.732 | 1.391 |
| L2H5 | 0.196 | 0.069 | 1.797 |
| L4H8 | 0.189 | 0.612 | 1.914 |
| L2H3 | 0.173 | 0.066 | 1.834 |
| L8H6 | 0.173 | 0.609 | 1.602 |
| L6H1 | 0.010 | 0.751 | 0.948 |
| L0H4 | 0.037 | 0.004 | 1.544 |
| L9H9 | 0.012 | 0.776 | 0.938 |
| L8H2 | 0.090 | 0.463 | 2.307 |
| L2H9 | 0.052 | 0.077 | 1.767 |
| L7H3 | 0.040 | 0.608 | 1.636 |

L6H10 first-token attention on non-English prompts: 0.602 when the baseline output is in the right language (122 prompts), 0.757 when it isn't (278). On the prompt itself, attention to the first token across query positions is 0.44-0.96 (5th-95th percentile).

Probing accuracy for the prompt language from the last prompt token, 2,500 prompts, 5-fold logistic regression, by hidden state (0 = embeddings, k = output of layer k-1):

| hidden state | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.373 | 0.975 | 0.980 | 0.971 | 0.968 | 0.973 | 0.966 | 0.968 | 0.963 | 0.962 | 0.963 | 0.962 | 0.965 |
