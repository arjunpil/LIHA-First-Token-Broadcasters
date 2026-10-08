# GPT-2 mean-ablation validation

Mean ablation was run on the same 2,500 European prompts used by the corrected
zero-ablation sweep. Each head was replaced at the input to `c_proj` by its
clean dataset mean activation, computed over valid prompt tokens.

Language detection uses `langdetect` with seed 0.

The baseline language accuracy was 0.4352, matching the corrected zero-ablation
run. Across all 144 heads, mean and zero ablation agreed at Spearman
rho = 0.771 for switch rate and rho = 0.692 for correct-to-wrong switches.
Agreement was weaker for LM-loss change (rho = 0.098).

L6H10 remained the clearest robust language-maintenance head:
correct-to-wrong was 0.2096 under zero ablation and 0.1768 under mean
ablation. L6H1 remained near zero under both interventions
(0.0100 versus 0.0060 correct-to-wrong).

`summary.json` contains the mean-ablation table and rank correlations.
`mean_vs_zero_comparison.csv` contains side-by-side mean and corrected zero-ablation
metrics for all 144 heads.
