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
Base: L0H0 0.016 (paper 0.016, its max). Our max is L1H9 at 0.024, one prompt more; two of its three flips involve
an empty output (prompt 83, a real end-of-text). So the paper's Qwen numbers come from the o_proj-output hook.

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


# GPT-2 multi-head ablation (fig 1b), 2026-10-07

2,500 European prompts, true head ablation, heads added one at a time. Orders: by SR, by correct->wrong, by
correct->wrong skipping heads with dNLL > 0.1 (this drops L0H7, L0H10 and the other layer-0 heads that raise LM loss
a lot), and 3 random orders. Every step is in results/gpt2-multi*/summary.md. Baseline accuracy 0.435.

| order | k=1 | k=3 | k=5 | k=10 | dNLL at k=10 |
|---|---|---|---|---|---|
| c->w, dNLL <= 0.1 | 0.229 (L6H10) | 0.226 | 0.274 | 0.308 | +0.49 |
| c->w | 0.233 (L0H7) | 0.474 | 0.384 | 0.365 | +1.21 |
| SR | 0.510 (L0H10) | 0.460 | 0.780 | 0.516 | +1.96 |
| random, 3 orders | 0.30-0.35 | 0.29-0.55 | 0.30-0.54 | 0.35-0.45 | +0.16 to +1.36 |

L6H10 alone takes accuracy from 0.435 to 0.229 with dNLL +0.016. That is the lowest of all 144 single-head ablations
(mean 0.419). By language: es 0.47 -> 0.02, de 0.45 -> 0.07, it 0.20 -> 0.03, fr 0.07 -> 0.03, en 0.99 -> 0.99, and
the flipped outputs are English. Adding more heads doesn't push accuracy lower: it stays at 0.23-0.31 up to k=10
while dNLL climbs to +0.49. Ranking by SR raises accuracy (0.78 at k=5) since those heads mostly flip wrong to
correct, with dNLL above +1. Single heads vary a lot (the first head of each random order already gives 0.30-0.35),
so it's fairer to compare L6H10 with the other heads than with 0.435. It's still the lowest of the 144.

The paper has L6H1 alone at 39.2% and the top 10 at 32.4% with monotonic degradation. Under true ablation the curve
is not monotonic, and one head gets as low as the 10-head set does.

The multi-head run also recomputes base and the k=1 condition of every order (L6H10, L0H7, L0H10, L8H2, L2H10, L1H2)
in a separate process. The detected labels match results/gpt2 on all 2,500 prompts for each, and dNLL is within 0.001.

figures/ has fig1_ablation_heatmap (SR per head, same layout as the paper's fig 1a), fig1_c2w_heatmap (correct->wrong,
x marks dNLL > 0.1) and fig3_accuracy_curve (fig 1b, accuracy and dNLL). File names match the paper's figures/.

# Detector and prompt-split checks (GPT-2), 2026-10-07

The head-hook generations relabeled with langid, fastText (lid.176) and a 2-of-3 vote give the same picture: c->w over
the 144 heads correlates with the langdetect version at Spearman 0.994 or higher, L6H10 is the top c->w head among the
dNLL <= 0.1 heads under every detector, and the top five are the same heads. Across 200 random half splits of the
prompts (stratified by language), c->w per head correlates at 0.976 between halves (5th percentile 0.969) and L6H10 is
first in both halves every time. results/gpt2/robustness.md, from robustness.py.

# GPT-2 zh/ru, 2026-10-07

100 prompts per language: the 5 hand-written ones plus 95 FLORES devtest sentences that pass the same 3-way vote as
expand_dataset.py (none were dropped), prompts_extended.csv. Generation and detection as above. LM loss is still
measured on the European dev sentences, so dNLL is the same as in the European table.

Baseline accuracy: zh 0.16 (GPT-2 mostly continues Chinese prompts in English), ru 0.87; 0.80 on the 10
hand-written prompts.

Paper hook on the 10 hand-written prompts: L0H0 0.40 (paper 0.40), L4H5 0.30 (0.30), L4H2 0.20 (0.30),
L0H1 / L0H7 / L1H10 0.20 (0.20), L6H1 / L0H4 / L3H1 / L9H9 0.0 (0.0). 9 of 10 match.

True head ablation on 200 prompts: mean SR 0.146 (sd 0.095).
- The European heads from the paper are not at zero: L6H1 0.090, L0H4 0.220, L3H1 0.130, L9H9 0.050. The paper's
  zh/ru heads: L0H0 0.300 (rank 12), L4H5 0.065 (rank 124), L1H10 0.200 (rank 33).
- The highest SR is at layer 0 (L0H10 0.545, L0H9 0.480, L0H5 0.475), almost all wrong to correct on zh. With these
  heads ablated GPT-2 repeats Chinese characters from the prompt (e.g. 阶阶阶...) and langdetect calls it zh.
  L0H10 also has dNLL +1.30.
- Correct to wrong comes mostly from ru: L11H0 (ru c->w 0.47, dNLL +0.041), L9H8 (0.41, +0.072), L0H7 (0.35,
  +0.78). The new outputs are mostly English.
- The heads overlap with the European ones: Spearman over the 144 heads is 0.79 for SR and 0.64 for c->w, and 4 of
  the top 10 c->w heads are shared (L0H7, L6H10, L4H8, L9H8). L11H0, L11H2 and L1H5 rank higher on zh/ru.

So "the European heads have no effect on zh/ru and other heads in layers 0-4 take over" doesn't hold with the
fixed hook.

# BLOOM-1b7 (running)

The first attempt had two problems.
- fp16 with left padding gives NaN logits on some rows, and 2,257 of the 2,500 baseline generations came out empty.
  Single-prompt fp16 and batched fp32 both give normal text, so BLOOM now runs in fp32. The Qwen runs don't have
  this: the few empty Qwen base outputs are a real end-of-text on one prompt.
- bloom_experiment.py (and section 3 of the paper) uses hidden 1024 / head dim 64, but bloom-1b7 has hidden 2048
  with 16 heads of 128. So the original hook zeroed 64 dims of the self_attention output, which includes the
  residual, at h*64, and its 16 "heads" only covered the first half of the hidden size. Paper mode keeps the 64-wide
  slice to reproduce what was run; head mode uses 128.

Paper hook, 25 hand-written prompts, every third layer, fp32 batched: baseline accuracy 0.88, mean SR 0.060
(sd 0.039), max 0.24 at L15H5 (4.6 sd), 12 heads above 0.1. The paper has max 0.16, 2.60 sd, 4 heads above 0.1.
This doesn't match yet; the fp16 single-prompt run, as in bloom_experiment.py, is queued after the 2,500-prompt
head-mode run.

Mean vs zero ablation and the matched-null redistribution test are in Chaewon's PR #6 (results/gpt2-mean-ablation,
results/gpt2-redistribution). TABLES.md has the paper's tables recomputed from results/ (python tables.py).
