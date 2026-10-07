# GPT-2 downstream attention redistribution

The redistribution test ablates each layer-6 head at the input to `c_proj` and
measures changes in first-token attention only in downstream layers (7--11).
The primary target is L6H10, selected from the corrected language-maintenance
results. The primary statistic is the mean increase among the top five
downstream heads, with top-five selection repeated inside every paired
prompt-level sign-flip permutation.

On 2,500 prompts, the final-prompt-position statistic for L6H10 was 0.07469.
None of 100,000 matched permutations reached the observed value, giving the
finite-permutation p-value 1/(100000+1) = 9.9999e-06
(Bonferroni-corrected p = 1.99998e-05 across the two attention summaries).

Matched-null permutations use seed 42; correlation permutations use seed 1000.

Across the 12 layer-6 heads, final-prompt-position redistribution correlated
with independently measured correct-to-wrong effects under corrected zero
ablation (rho = 0.701, permutation p = 0.0144) and mean ablation
(rho = 0.809, permutation p = 0.00243). The all-prompt-position correlations
were weaker.

These results support downstream attention redistribution following L6H10
ablation. They do not by themselves establish functional compensation.

`matched_null_statistics.json` contains the matched-null tests and correlation
statistics.
