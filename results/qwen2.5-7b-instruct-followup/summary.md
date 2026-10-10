# qwen2.5-7b-instruct: follow-up on the top correct->wrong heads

zero = the sweep's head ablation; mean = the head's mean over prompt tokens; xN = the head scaled by N.

| condition | accuracy | non-English acc | c->w | w->c | en | fr | de | es | it | dNLL |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.997 | 0.996 | 0.000 | 0.000 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | +0.0000 |
| L19H1:mean | 0.980 | 0.976 | 0.018 | 0.002 | 1.00 | 0.98 | 0.95 | 1.00 | 0.98 | +0.0141 |
| L19H1:zero | 0.797 | | 0.200 | 0.000 |  |  |  |  |  | +0.0260 |
| L19H1:x2 | 0.996 | 0.996 | 0.002 | 0.001 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | -0.0007 |
| L19H1:x3 | 0.572 | 0.465 | 0.426 | 0.001 | 1.00 | 0.38 | 0.58 | 0.36 | 0.54 | +0.1070 |
| L19H1:x5 | 0.235 | 0.044 | 0.762 | 0.000 | 1.00 | 0.09 | 0.02 | 0.03 | 0.04 | +3.6552 |
