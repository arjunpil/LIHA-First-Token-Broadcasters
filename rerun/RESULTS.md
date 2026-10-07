# LIHA GPT-2 rerun (2026-10-06 night)

Setup: GPT-2 small, the 2,500 European prompts rebuilt with expand_dataset.py's rules from FLORES-200 devtest
(5 hand-written + 495 FLORES per language), greedy 40 tokens, langdetect with seed 0. Batched generation matches
single-prompt generation on 50/50 checked prompts. Baseline language accuracy 0.435, identical to the paper (43.5%).

Two hooks per head: "head" zeros head h's slice at the c_proj input (true head ablation), "paper" zeros the same
slice of the attention output after c_proj (what experiment.py does).

## Reproduction of the paper with its own hook (25 hand-written prompts)
L6H1 0.28 (paper 0.32), L0H4 0.24 (0.28), L3H1 0.24 (0.28), L9H9 0.24 (0.28), L7H3 0.24 (0.24); population mean
0.128 (paper 0.163). Within one prompt for the top heads.

## Paper hook on 2,500 prompts
Effects shrink: mean SR 0.055 (sd 0.013), max 0.096. L6H1 0.080 (rank 10/144), L0H4 rank 22, L3H1 rank 120,
L9H9 rank 51, L7H3 rank 34.

## True head ablation on 2,500 prompts
Mean SR 0.111 (sd 0.082). The paper's heads: L6H1 SR 0.038 (rank 136/144), L9H9 rank 113, L7H3 rank 83,
L0H4 rank 64, L3H1 rank 20, L10H4 rank 18. Spearman between the two hooks over heads: -0.10.

Switch direction matters. Some high-SR heads mostly flip wrong to correct (ablating them raises accuracy):
L0H0 (c->w 0.083, w->c 0.297), L5H1 (0.099, 0.190), L7H6 (0.014, 0.214). Layer-0 heads with high SR also raise
LM loss a lot (L0H10 dNLL +1.30, L0H0 +1.22, L0H7 +0.78), i.e. general disruption.

Heads that look like language maintenance (correct->wrong high, wrong->correct near 0, small LM loss change):
| head | c->w | w->c | SR | dNLL | first-token attn |
|---|---|---|---|---|---|
| L6H10 | 0.210 | 0.004 | 0.250 | +0.016 | 0.80 |
| L2H5 | 0.196 | 0.024 | 0.298 | +0.033 | 0.13 |
| L4H8 | 0.189 | 0.010 | 0.235 | +0.027 | 0.59 |
| L2H3 | 0.173 | 0.036 | 0.282 | +0.023 | 0.14 |
| L8H6 | 0.173 | 0.008 | 0.212 | +0.048 | 0.54 |
| L9H8 | 0.169 | 0.013 | 0.216 | +0.072 | 0.47 |

## First-token attention (attention sink check)
Across 144 heads, first-token attention is negatively correlated with the true ablation effect
(Spearman -0.52 with SR, -0.45 with correct->wrong). The 46 heads with first-token attention > 0.8 have mean SR
0.082 vs 0.125 for the rest. The paper's broadcasters attend to the first token heavily (L6H1 0.87, L9H9 0.95) but
barely affect output language. First-token attention alone does not mark language-critical heads; L6H10 is the one
clean head that combines both.

Files are written to out/<model>/.

# Qwen2.5-1.5B base and instruct rerun

Setup as in qwen_experiment.py: the first 25 prompts per language (125), fp16, greedy 40 tokens, chat template
for instruct, langdetect seed 0. Batched generation matches single-prompt generation on 25/25 (instruct) and
24/25 (base) checked prompts. Baseline accuracy: instruct 0.936, base 0.984.

## Reproduction with the paper's hook (o_proj output slice)
Instruct: L0H5 0.224 (paper 0.224), L0H11 0.160 (0.144), L1H9 0.152 (0.136), L0H7 0.128 (0.120), L1H7 0.120 (0.112).
Base: max SR 0.016 at L0H0 (paper 0.016 at L0H0). So the paper's Qwen numbers come from the o_proj-output hook.

## True head ablation (o_proj input)
Instruct, top heads by SR (all flips are correct to wrong):
| head | SR [95% CI] | c->w | w->c | dNLL | same-layer dNLL |
|---|---|---|---|---|---|
| L22H6 | 0.480 [0.392, 0.568] | 0.480 | 0.000 | +0.228 | +0.004 |
| L17H7 | 0.264 [0.192, 0.344] | 0.264 | 0.000 | +0.002 | +0.002 |
| L0H6 | 0.256 [0.176, 0.336] | 0.240 | 0.016 | +0.211 | +0.101 |
| L17H8 | 0.224 [0.144, 0.296] | 0.224 | 0.000 | +0.007 | +0.002 |
| L7H7 | 0.176 [0.104, 0.248] | 0.176 | 0.000 | +0.006 | +0.003 |
| L10H5 | 0.168 [0.104, 0.232] | 0.168 | 0.000 | +0.013 | +0.001 |
The paper's L0H5 has SR 0.056 under true ablation (rank 38/336).

Base: L22H6 SR 0.168 (c->w 0.160, dNLL +0.216); every other head is at or below 0.048.

So the base model is not flat and the instruct model is not concentrated at layer 0. The strongest head (L22H6)
is the same in both and its effect grows from 0.17 to 0.48 with instruction tuning, and mid-layer heads
(L17H7, L17H8, L7H7, L10H5) join it with almost no change in LM loss. L22H6 and L17H7 are also in the Qwen EN-ES
circuit found by activation patching in the circuit paper (16:9, 17:7, 22:6, 25:10, 27:6).

Not rerun yet: BLOOM, the zh/ru GPT-2 table, multi-head ablation (Fig 1b), redistribution with a matched null.
