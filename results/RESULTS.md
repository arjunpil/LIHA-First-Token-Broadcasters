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

Overall accuracy has a floor near 0.20, though: English prompts are a fifth of the set and stay English under every
ablation here (0.99), so the paper's chance line at 0.20 is really that floor. On the non-English prompts alone,
L6H10 takes accuracy from 0.296 to 0.038, i.e. it already removes almost every correct non-English continuation and
there's little left for the other heads to remove. The rise after k=4 (0.09-0.14) is GPT-2 copying the prompt back,
which langdetect counts as the right language. fig3_accuracy_curve plots the non-English accuracy.

The multi-head run also recomputes base and the k=1 condition of every order (L6H10, L0H7, L0H10, L8H2, L2H10, L1H2)
in a separate process. The detected labels match results/gpt2 on all 2,500 prompts for each, and dNLL is within 0.001.

figures/ has fig1_ablation_heatmap (SR per head, same layout as the paper's fig 1a), fig1_c2w_heatmap (correct->wrong,
x marks dNLL > 0.1) and fig3_accuracy_curve (fig 1b, accuracy and dNLL). File names match the paper's figures/.

# GPT-2 amplification, 2026-10-07

Each head's slice at the c_proj input scaled by 2, 3 or 5 (amplify.py), one head at a time, 2,500 prompts. Heads: the
top five by c->w with dNLL <= 0.1, the top five by SR (the paper's selection rule), and L6H1. Full table in
results/gpt2-amp/summary.md.

| head | x2 | x3 | x5 | dNLL at x5 |
|---|---|---|---|---|
| L6H10 | 0.660 | 0.775 | 0.870 | +0.041 |
| L8H6 | 0.601 | 0.703 | 0.747 | +0.125 |
| L2H5 | 0.502 | 0.385 | 0.226 | +0.686 |
| L0H10 | 0.358 | 0.120 | 0.007 | +3.728 |
| L6H1 | 0.426 | 0.421 | 0.413 | +0.016 |

Baseline accuracy 0.435. Scaling L6H10 by 3 raises it to 0.775 with dNLL +0.006, and by 5 to 0.870: French goes from
0.07 to 0.81, German 0.45 to 0.91, Spanish 0.47 to 0.89, Italian 0.20 to 0.74, English stays at 1.00. Each output
follows its own prompt's language. The new correct outputs aren't prompt copies: the share of their 4-grams that
appear in the prompt goes down (0.31 at baseline, 0.14 at x5), though they're somewhat more repetitive (0.48 to
0.60), as GPT-2's non-English text already is. The paper's heads (L6H1, and the top-SR heads like L0H10) don't help,
and the zero-ablation candidates that failed mean ablation (L2H5, L2H3) hurt at x5 with a large dNLL.

This is the opposite of the paper's "no observed accuracy improvement" for 2-5x amplification.
Added 10/8: most of what scaling adds is repetition. At 5x, 76% of the outputs in the prompt language repeat
themselves and 15% copy the prompt (quality.py's split); outputs that are neither go from 2.3% of the non-English
prompts at baseline to 5.5% at 3x and 7.5% at 5x. So scaling keeps GPT-2 in the prompt's language, mostly without
making it fluent.

# L6H10 versions of the section 5 / appendix numbers, 2026-10-07

attention.py, 500 prompts (100 per language), greedy, 40 steps. results/gpt2-attention has the attention and entropy
figures (same file names as the paper's) and summary.md with a Table 6 replacement.
- During generation L6H10 puts 0.73 of its attention on the first token, with entropy 1.39. That sits between the
  paper's heads (L6H1 0.75 / 0.95, L9H9 0.78 / 0.94) and random heads (entropy 1.64-2.31). The other c->w heads go
  from sink-like (L4H8, L8H6, about 0.61) to hardly looking at the first token (L2H5, L2H3, about 0.07), so
  first-token attention doesn't pick them out. The old entropy figure used L2H5 and L4H8 as its random heads.
- On non-English prompts L6H10 attends to the first token less when the output stays in the prompt language (0.60,
  122 prompts) than when it goes to English (0.76, 278 prompts), the same direction as the paper's L6H1 numbers
  (0.847 vs 0.923).
- On the prompt itself its attention to the first token is 0.44-0.96 across query positions (5th-95th percentile);
  the paper has 0.62-1.00 for L6H1.
- Probing on all 2,500 prompts from the last prompt token: 0.37 from the embeddings, then 0.96-0.98 after every layer
  from layer 0 on. The prompt language is linearly readable everywhere past the first layer, so probing can't point
  to particular layers and the paper's 85% vs 58% comparison doesn't carry over.

# L6H10 language identity pilot, 2026-10-07

identity.py. L6H10's mean output per language over 200 FLORES dev sentences each. set:X replaces the head's output
with language X's mean at every position; add:X adds 3 x (X's mean minus the mean over the five languages). Table in
results/gpt2-identity/summary.md.

- Replacing with any language's mean acts like ablation: non-English outputs go to English (0.78-0.96), even with the
  prompt's own language (German prompts with the German mean: 0.17 German, 0.80 English). The patched-in language
  almost never shows up (0.01 or less).
- Adding the English direction sends everything to English (0.93-0.97). Adding the German, Spanish or Italian
  direction makes non-English prompts keep their own language more (German prompts 0.45 to 0.55-0.71, Spanish 0.47 to
  0.55-0.65), but never switches them to the added language. The French direction does nothing.
- The means line up the same way. The English offset (norm 0.68) points against German, Spanish and Italian (cosine
  -0.76 to -0.93), which point roughly together (0.33 to 0.71). French's offset is small (0.12) and leans toward
  English, in line with GPT-2 continuing French prompts in English 90% of the time.

At the level of its average output, L6H10 separates English from German, Spanish and Italian but doesn't steer
between those three. That supports describing it as keeping the prompt language instead of falling back to English.
It doesn't rule out language identity in the input-specific part of the output (the own-language mean already fails,
so that part is what matters); swapping outputs position by position between aligned sentences would test that.

# Setting checks, 2026-10-07

checks.py, each with the main heads plus three control heads drawn at random from the dNLL <= 0.1 heads outside the
top ten by c->w. Tables in results/gpt2-sampling, results/gpt2-truncated and results/qwen-format.

Sampling (GPT-2, temperature 0.7, three seeds, each condition compared with the same-seed baseline): L6H10's c->w is
0.256-0.272 across seeds, higher than with greedy decoding (0.210); L4H8 0.17-0.19, L2H5 0.14-0.15, the controls
0.01-0.09, L6H1 about 0.02.

Truncated prompts (GPT-2, FLORES sentences cut to their first half, at least four words): the model now continues a
sentence instead of starting a new one, and baseline non-English accuracy is 0.716 instead of 0.296. L6H10 only takes
it to 0.644 (c->w 0.095), close to the control heads (0.02-0.08). L6H10 seems to matter mostly when GPT-2 starts a
new sentence after a complete one, as in the FLORES setup. Mid-sentence, the preceding words already fix the language.

Qwen prompt format (125 prompts): instruct without the chat template and base with it.

| L22H6 c->w | raw text | chat template |
|---|---|---|
| base | 0.160 | 0.144 |
| instruct | 0.168 | 0.480 |

With the same input format base and instruct are close; the instruct model leans on L22H6 much more only in the chat
format it was tuned on. L17H7 follows the same pattern (0.264 for instruct with the template, 0.016-0.064 otherwise).
The tuning effect on these heads only shows up with chat-formatted input.

# Detector and prompt-split checks (GPT-2), 2026-10-07

The head-hook generations relabeled with langid, fastText (lid.176) and a 2-of-3 vote give the same picture: c->w over
the 144 heads correlates with the langdetect version at Spearman 0.994 or higher, L6H10 is the top c->w head among the
dNLL <= 0.1 heads under every detector, and the top five are the same heads. Across 200 random half splits of the
prompts (stratified by language), c->w per head correlates at 0.976 between halves (5th percentile 0.969) and L6H10 is
first in both halves every time. results/gpt2/robustness.md, from robustness.py.

Of the five top c->w heads, only L6H10 holds up under the mean ablation of PR #6: c->w 0.210 with zero ablation
and 0.177 with mean ablation, while L2H5 and L4H8 drop to 0.057, L8H6 to 0.083 and L2H3 to 0.010. L6H10's LM loss
change is also specific to non-English text: +0.001 on English and +0.011 to +0.029 on fr/de/es/it.

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

# BLOOM-1b7, 2026-10-07

The first attempt had two problems.
- fp16 with left padding gives NaN logits on some rows, and 2,257 of the 2,500 baseline generations came out empty.
  Single-prompt fp16 and batched fp32 both give normal text, so BLOOM now runs in fp32. The Qwen runs don't have
  this: the few empty Qwen base outputs are a real end-of-text on one prompt.
- bloom_experiment.py (and section 3 of the paper) uses hidden 1024 / head dim 64, but bloom-1b7 has hidden 2048
  with 16 heads of 128. So the original hook zeroed 64 dims of the self_attention output, which includes the
  residual, at h*64, and its 16 "heads" only covered the first half of the hidden size. Paper mode keeps the 64-wide
  slice to reproduce what was run; head mode uses 128.

Paper hook on the 25 hand-written prompts, every third layer, run exactly as bloom_experiment.py does (fp16, one
prompt at a time, 64-wide slice): baseline accuracy 0.88, max SR 0.20 at L9H14 (3.77 sd), 12 heads above 0.1. fp32
batched gives max 0.24. The paper has max 0.16, 2.60 sd and 4 heads above 0.1, so unlike GPT-2 and Qwen the submitted
BLOOM numbers don't reproduce even with the original settings.

True head ablation, all 384 heads, 2,500 prompts, fp32: baseline accuracy 0.772 (en 0.87, fr 0.76, de 0.83, es 0.52,
it 0.88). 482 of the 2,500 baseline outputs are empty, because BLOOM often ends the document right after a complete
FLORES sentence, and those count as wrong. Mean SR 0.011 (sd 0.009), a tenth of GPT-2's.
- The top c->w head, L23H12 (0.090, dNLL +0.000), doesn't switch language. With it removed BLOOM stops right away on
  218 more prompts.
- The heads that do switch language sit in layers 18-21 and act on single languages. L21H15 takes German from 0.834 to
  0.634, mostly to English; L18H14 and L19H15 do the same on a smaller scale. Their LM loss change is on German
  (+0.015 to +0.032) and Italian (+0.04 to +0.08) and near zero on the other languages. Italian outputs go to French
  or Spanish more often than to English.
- There is nothing like GPT-2's L6H10. The largest real switch rate from one head is 0.046 (L21H15).

results/bloom and results/bloom-paper25-fp16.

# GPT-2 medium, OLMo-2 1B and Pythia-1B, 2026-10-08

Same head sweep as for GPT-2 (fixed hook, 2,500 prompts, greedy 40 tokens), then followup.py on the top three c->w
heads with dNLL <= 0.1: mean ablation and scaling by 2, 3 and 5. results/gpt2-medium, results/olmo2-1b and their
-followup folders. For the GPT-2 models the mean condition also replaces the head at the BOS position of the loss
sentences, so its c->w is comparable but its dNLL isn't.

GPT-2 medium (24 layers x 16 heads), baseline non-English accuracy 0.300, close to GPT-2 small's 0.296.
- L13H6, at about the same relative depth as L6H10, takes non-English accuracy to 0.066, the lowest of the 384 heads,
  with dNLL +0.006. All four languages drop to 0.05-0.10, and 476 of the 479 flipped outputs are English. Mean
  ablation keeps most of it (c->w 0.151 vs 0.192). Scaling it by 3 raises accuracy from 0.440 to 0.740 and by 5 to
  0.817 (non-English 0.300 to 0.772, German 0.35 to 0.93) without moving LM loss.
- Two weaker heads do the same, L19H14 (c->w 0.158) and L16H6 (0.133). Both hold under mean ablation and help when
  scaled.

OLMo-2 1B (16 x 16) ends the document right after most complete FLORES sentences (p(end-of-text) 0.28 on average), so
it runs with the end-of-text token blocked. Baseline non-English accuracy 0.921.
- The heads with the highest switch rates (L2H0, L13H10, L12H0) also raise LM loss by 0.3-0.55. Among heads with
  dNLL <= 0.1, L15H5 in the last layer stands out: c->w 0.134 (next 0.072), dNLL +0.035, flips going to English. It
  acts on the Romance languages: French 0.90 to 0.68, Spanish 0.98 to 0.84, Italian 0.97 to 0.81, German 0.83 to
  0.81. Mean ablation keeps most of it (0.115). Scaling helps little because accuracy is near the ceiling
  (non-English 0.921 to 0.938 at 5x).

Pythia-1B (16 x 8), baseline non-English accuracy 0.990 and no early end-of-text, so no blocking. Mean SR 0.015.
- L1H7 has c->w 0.635 but breaks generation: the model emits only newlines after the prompt (1,456 of 2,500 outputs
  empty), English prompts included. It isn't a language effect.
- Apart from it no head moves more than 4.4% of outputs (L9H3), and that drops to 0.5% under mean ablation.

Across the seven models, the two GPT-2 models, which mostly fail to continue non-English text, each have one mid-depth
head that keeps the prompt language, with a large effect on all four languages. In the models that handle these
languages well there is either no such head (Pythia, BLOOM) or a weaker one later in the network (OLMo L15H5 for the
Romance languages, Qwen L22H6).

# What the outputs that stay in the prompt language look like, 2026-10-08

quality.py, results/quality.md. Baseline outputs of non-English prompts that langdetect puts in the prompt language:

| model | outputs | repetition | prompt copy | other |
|---|---|---|---|---|
| GPT-2 small | 592 | 57% | 35% | 8% |
| GPT-2 medium | 600 | 35% | 48% | 17% |
| OLMo-2 1B | 1,842 | 11% | 7% | 82% |
| Pythia-1B | 1,980 | 8% | 9% | 83% |

Repetition means half or more of the words are repeats, prompt copy means half or more of the 4-grams come from the
prompt. So in GPT-2 almost all of what counts as staying in the prompt language is repeating or copying; the GPT-2
models can't write fluent French, German, Spanish or Italian. L6H10 sends 82-89% of each kind to English, so its
effect isn't limited to the degenerate outputs. The metric is prompt-language retention, not output quality, and the
GPT-2 results should be described that way: the head keeps generation tied to the non-English prompt, and without it
the model starts unrelated English text.

# Content of the continuations, 2026-10-08

content.py, results/gpt2-content. Cosine similarity between each non-English prompt and its continuation with a
multilingual encoder, Qwen3-Embedding-0.6B (LaBSE as a second check). FLORES reference pairs give the scale:
Qwen3-Embedding scores a sentence and the next sentence of the same article 0.38 in the same language and 0.37 in
English, and an unrelated sentence 0.18 and 0.17, so it barely favours same-language pairs. LaBSE does (0.43 vs 0.33
for next sentences), which is why it's only the second check.
- GPT-2 continuations in the prompt language: 0.471. Baseline drifts to English: 0.219.
- Prompts that L6H10 sends to English: 0.453 before, 0.195 after, i.e. at the level of an unrelated sentence. 9% of
  them stay on topic (0.37 or more). The control heads that flip a few prompts show the same drop.
- LaBSE gives the same pattern (0.395 before, 0.112 after).
The English that appears without L6H10 isn't the same content in another language; GPT-2 drops the prompt and starts
generic English text.


# Qwen2.5-1.5B-Instruct on 2,500 prompts, 2026-10-08

results/qwen-instruct-2500. Same 2,500 prompts as GPT-2, chat template, fp16, greedy 40 tokens, true head ablation
of every head in layers 17 and 22 (the full sweep of all layers is running). Baseline retention 0.922 overall and
0.902 on the 2,000 non-English prompts (en 1.000, fr 0.978, de 0.822, es 0.840, it 0.968).

| head | c->w | w->c | non-English retention | dNLL | same-layer dNLL |
|---|---|---|---|---|---|
| L22H6 | 0.500 | 0.001 | 0.278 | +0.228 | +0.004 |
| L17H7 | 0.325 | 0.001 | 0.496 | +0.002 | +0.002 |
| L17H8 | 0.235 | 0.000 | 0.609 | +0.007 | +0.002 |
| L17H6 | 0.109 | 0.000 | 0.766 | -0.000 | +0.002 |

Without L22H6 retention is fr 0.544, de 0.210, es 0.350, it 0.008. Of the prompts kept in the prompt language at
baseline, French, German and Spanish flip almost only to English (216, 302 and 235, plus 11 Spanish to Portuguese).
Italian goes mostly to Spanish (208), then English (165), Portuguese (79) and French (26).

The outputs are fluent: of the 1,804 baseline outputs in the prompt language, 0% are repetition and 1% prompt copy.
When L22H6 sends them to English they stay on the prompt's content (Qwen3-Embedding similarity to the prompt 0.475
before, 0.558 after, 918 prompts, against 0.384 for the next FLORES sentence and 0.179 for a random one). So in Qwen
the head behaves like language confusion, not like the GPT-2 case where the English is unrelated.

dNLL caveat: L22H6 raises LM loss far more than the rest of its layer. It does the same in the base model (+0.216 on
125 prompts), where it flips only 0.16, so the loss increase alone doesn't produce the switch.

Follow-up (results/qwen-instruct-followup, mean over prompt tokens and scaling, chat template):
| condition | non-English retention | c->w | w->c | dNLL |
|---|---|---|---|---|
| baseline | 0.902 | | | |
| L22H6 mean | 0.341 | 0.450 | 0.001 | +0.165 |
| L17H7 mean | 0.905 | 0.016 | 0.018 | -0.004 |
| L17H8 mean | 0.807 | 0.076 | 0.001 | +0.003 |
| L17H7 x2 | 0.987 | 0.000 | 0.068 | +0.009 |
| L17H8 x3 | 0.996 | 0.001 | 0.076 | +0.002 |
| L22H6 x2 | 0.811 | 0.112 | 0.039 | +0.037 |

L22H6 holds under mean ablation, the L17 heads mostly don't, so their zero-ablation effect may come from the
out-of-distribution input. Scaling L17H7 or L17H8 up keeps almost every non-English reply in the prompt language
(L17H8 x3 fixes 191 of the baseline's wrong replies) without the repetition seen in GPT-2 (0% repetition, about 1%
copy). Whether the replies are still good answers isn't checked yet; one German reply is grammatical but off.

# Qwen2.5-1.5B full sweeps, 2026-10-08

results/qwen-instruct-full, results/qwen-base-full. Every head (336) on all 2,500 prompts, chat template for
instruct. L22H6 is the top correct->wrong head in both: 0.500 in instruct (then L17H7 0.325, L17H8 0.235, L0H6
0.193 with dNLL +0.211), 0.150 in base, where the next head is at 0.023. Mean correct->wrong over heads is 0.019 in
instruct and 0.003 in base. Baseline non-English retention is 0.902 in instruct and 0.984 in base.

Prompt format on 2,500 prompts (results/qwen-format-2500): L22H6 correct->wrong is 0.176 for instruct on raw text
and 0.089 for base with the chat template.

figures/fig1_qwen_c2w.png (python experiments/figures.py) shows correct->wrong for every head of both models on one
scale, with the top head boxed.

# System prompt check, 2026-10-08

results/qwen-system-2500, checks.py qwen-system. Qwen2.5's chat template adds an English system prompt ("You are
Qwen, created by Alibaba Cloud. You are a helpful assistant.") when none is given. Same 2,500 prompts with that
default, with no system turn, and with the same system prompt translated into the prompt's language:
| setting | baseline non-English retention | L22H6 c->w | L17H7 | L17H8 | controls (max) |
|---|---|---|---|---|---|
| English system prompt (default) | 0.902 | 0.500 | 0.325 | 0.235 | 0.002 |
| system prompt in the prompt's language | 0.996 | 0.379 | 0.085 | 0.020 | 0.001 |
| no system prompt | 0.983 | 0.258 | 0.155 | 0.072 | 0.015 |

L22H6 matters in all three, most with the English system prompt. The L17 heads matter mostly with it. The English
default system prompt also lowers baseline retention by itself.

# Language Confusion Benchmark, Qwen2.5-1.5B-Instruct, 2026-10-08

results/qwen-instruct-lcb, lcb.py. LCB (Marchisio et al., 2024): 800 monolingual prompts in fr/de/es/it (the reply
should stay in that language) and 1,196 crosslingual ones (English prompt asking for a reply in fr/de/es/it), plus
200 English. Default chat template, greedy 100 tokens, the benchmark's line-level pass rate (LPR). Mean ablation uses
the head's mean over the 2,500 FLORES prompts. Δ is the paired change on the non-English prompts with a bootstrap
95% CI. samples.jsonl.gz has every reply.
| condition | monolingual LPR | Δ | crosslingual LPR | Δ |
|---|---|---|---|---|
| baseline | 0.982 | | 0.704 | |
| L22H6 zero | 0.747 | -0.273 [-0.305, -0.240] | 0.427 | -0.276 [-0.303, -0.250] |
| L22H6 mean | 0.775 | -0.246 [-0.276, -0.215] | 0.455 | -0.247 [-0.273, -0.220] |
| L17H7 zero | 0.921 | -0.065 | 0.667 | -0.036 |
| L17H8 zero | 0.976 | -0.004 | 0.697 | -0.008 |
| L17H8 x3 | 0.990 | +0.008 [-0.003, +0.018] | 0.711 | +0.005 |
| L22H6 x2 | 0.982 | +0.001 | 0.520 | -0.186 |
| 6 control heads (L17, L22) | | -0.004 to +0.005 | | -0.022 to +0.000 |

English prompts stay at 0.995. Italian collapses in both tasks (monolingual 1.00 -> 0.00, crosslingual 0.69 ->
0.00): replies start in Italian and slide into Spanish and Portuguese, and "write a poem in Italian" gets an English
poem. Because the crosslingual set breaks as much as the monolingual one, L22H6 keeps the reply in the requested
language whether the language comes from the prompt or from an instruction. The L17 heads and the L17H8 scaling
that looked useful on FLORES do not carry over.

# More base/instruct pairs, 2026-10-08

Every head is screened on 125 prompts (25 per language, results/<model>-screen). If the strongest head flips at
least 10% of the correct prompts, the layers of the top heads are rerun on all 2,500 and the top heads get the
follow-up. Instruct models use their chat template (one BOS, thinking off, a fixed date). Models after Qwen2.5 run
in fp32.
| model | heads per layer | baseline non-English retention | top instruct head (c->w) | dNLL (layer mean) | mean ablation | same head in base |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 12 | 0.902 | L22H6 0.500 | +0.228 (+0.004) | 0.450 | 0.150 |
| Qwen2.5-3B | 16 | 0.992 | L27H13 0.515 | +0.234 (+0.006) | 0.338 | 0.202 |
| Qwen3-1.7B | 16 | 0.994 | L18H12 0.324 | +0.128 (+0.003) | 0.028 | 0.067 |
| Gemma-3-1B | 4 | 0.997 | L11H3 0.413 | +0.720 (-0.012) | 0.089 | 0.001 |
| OLMo-2-1B | 16 | 0.997 | L12H8 0.080 | +0.136 (+0.022) | 0.056 | 0.007 |
| Llama-3.2-1B | 32 | 1.000 (screen) | none, strongest 0.048 (screen) | | | |
| Llama-3.2-3B | 24 | 0.990 (screen) | none, strongest 0.008 (screen) | | | |
| OLMo-3-7B (bf16) | 32 | 0.960 | L14H25 0.134 | +0.202 (+0.000) | 0.007 | 0.104 (screen) |

Gemma-3's L11H3 sends 1,032 non-English replies elsewhere, 873 of them to English; 1,031 are fluent (not
repetition or prompt copy). Its base model has no head above 0.088 on the screen, and under mean ablation the head
keeps only German down (0.60). OLMo-2's L12H8 is small but holds under mean ablation, mostly on German and Italian.
Neither Llama has such a head on FLORES at 1B or 3B, but Llama-3.2-1B has one on LCB (next sections). Qwen2.5-3B
and Qwen3-1.7B are in the next section.

Mean vs zero ablation and the matched-null redistribution test are in PR #6 (results/gpt2-mean-ablation,
results/gpt2-redistribution). TABLES.md has the paper's tables recomputed from results/ (python tables.py).

# Qwen2.5-3B and Qwen3-1.7B on 2,500 prompts, 2026-10-09

results/<model>, results/<model>-followup, screens in results/<model>-screen. fp32, Qwen3 with thinking off.
| model | head | c->w | dNLL (layer mean) | mean ablation c->w | x2 c->w | same head in base (mean ablation) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | L27H13 | 0.515 | +0.234 (+0.006) | 0.338 | 0.006 | 0.202 (0.114) |
| Qwen3-1.7B | L18H12 | 0.324 | +0.128 (+0.003) | 0.028 | 0.001 | 0.067 (0.022) |

Qwen2.5-3B repeats the 1.5B picture: one head in a late layer (27 of 36, against 22 of 28 at 1.5B), the next head
at 0.020, the same head weaker in base (0.202, against 0.150 at 1.5B), and most of the effect holds under mean
ablation, where Italian (0.06) and German (0.45) drop most. Qwen3-1.7B's L18H12 holds much less under mean ablation
on FLORES. Qwen3 also has L0H3 (0.466 instruct, 0.434 base) and in base L1H5 (0.196), but those raise dNLL by more
than 1 and break the model, so the follow-up skips them.

# LCB across models, 2026-10-09

Same setup as the Qwen2.5-1.5B LCB section, for each model's top FLORES head. The Llamas have no FLORES head, so
they get the top two heads of their screen; in Llama-3.2-3B these tie with many others at one prompt in 125.
Controls are random heads from the same layer, zero-ablated (three per layer, one for the Llamas).
results/<model>-lcb, samples.jsonl.gz has every reply.
| model | head | mono LPR | Δ mono, zero | Δ mono, mean | cross LPR | Δ cross, zero | Δ cross, mean | same-layer controls, Δ mono / Δ cross |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | L22H6 | 0.982 | -0.273 [-0.305, -0.240] | -0.246 | 0.704 | -0.276 [-0.303, -0.250] | -0.247 | -0.004 to +0.005 / -0.022 to +0.000 (L17 and L22) |
| Qwen2.5-3B | L27H13 | 0.982 | -0.537 [-0.575, -0.503] | -0.409 | 0.888 | -0.449 [-0.478, -0.421] | -0.331 | +0.000 to +0.003 / -0.025 to +0.004 |
| Qwen3-1.7B | L18H12 | 0.985 | -0.149 [-0.174, -0.123] | -0.067 | 0.823 | -0.508 [-0.535, -0.478] | -0.230 | -0.004 to -0.003 / +0.000 to +0.021 |
| Gemma-3-1B | L11H3 | 0.984 | -0.628 [-0.662, -0.593] | -0.244 | 0.118 | -0.091 [-0.110, -0.073] | -0.092 | -0.005 to +0.003 / -0.023 to +0.038 |
| Gemma-3-4B | L24H0 | 0.990 | -0.259 [-0.292, -0.227] | -0.120 | 0.133 | -0.107 [-0.126, -0.090] | -0.053 | -0.001 to +0.000 / +0.000 to +0.012 |
| OLMo-2-1B | L12H8 | 0.986 | -0.272 [-0.302, -0.239] | -0.128 | 0.931 | -0.335 [-0.362, -0.308] | -0.143 | +0.000 to +0.004 / -0.003 to +0.004 |
| OLMo-3-7B (bf16) | L14H25 | 0.972 | -0.041 [-0.060, -0.024] | +0.001 | 0.874 | -0.018 [-0.033, -0.003] | +0.015 | -0.011 to +0.004 / +0.001 to +0.007 |
| Llama-3.2-1B | L8H25 | 0.997 | -0.016 [-0.026, -0.008] | -0.015 | 0.874 | -0.711 [-0.737, -0.684] | -0.357 | -0.009 / +0.001 |
| Llama-3.2-3B | L0H2 | 0.993 | +0.001 [-0.006, +0.009] | +0.000 | 0.911 | -0.003 [-0.013, +0.008] | +0.000 | +0.003 / +0.001 |
| Llama-3.2-3B | L2H17 | 0.993 | +0.004 [-0.001, +0.010] | -0.004 | 0.911 | -0.006 [-0.014, +0.002] | -0.003 | +0.004 / -0.001 |

Every model except Llama-3.2-3B and OLMo-3-7B has a head whose removal moves replies out of the requested language
far beyond its same-layer controls. The crosslingual replies it breaks are mostly English; the monolingual ones are
not always (Qwen2.5-3B has 7% English lines there). How the drop splits between the two tasks differs. Qwen2.5 and
OLMo-2 lose both. Gemma-3-1B loses mostly monolingual, but it already answers most crosslingual prompts in English
at baseline (0.118, 55% English lines), so there is little left to lose there; Gemma-3-4B is the same (0.133,
44% English lines). Llama-3.2-1B's L8H25 leaves
monolingual replies alone and takes crosslingual from 0.874 to 0.164 (81% English lines), the largest drop here. It
stays under the FLORES threshold (0.048 on the screen) because FLORES only tests keeping the prompt's language.
Qwen3-1.7B leans the same way (-0.149 mono, -0.508 cross). So the head keeps the prompt's language in some models,
follows a requested language in others, and does both in Qwen2.5 and OLMo-2. Mean ablation keeps about 90% of the
zero-ablation drop in Qwen2.5-1.5B, 75% in Qwen2.5-3B, 43-50% in Qwen3, OLMo-2 and Llama-3.2-1B crosslingual, and
39% and 46% in Gemma-3-1B and 4B monolingual.

Since Llama-3.2-1B's head only shows up on LCB, every head of both Llamas is being screened on 100 crosslingual
prompts (lcb.py --screen).

# Comparing models

The ablation is the same in every model (the head's slice of the attention output projection's input set to zero),
but the size of the effect isn't comparable across models. Heads per layer go from 4 (Gemma-3-1B) to 32
(Llama-3.2-1B), so one head is 25% to 3% of a layer. Gemma-3, OLMo-2, OLMo-3 and Qwen3 use QK-norm, and Gemma-3,
OLMo-2 and OLMo-3 also normalize the attention output before adding it to the residual stream, so zeroing a head
rescales what the other heads write. Gemma-3's L5 and L11 are both global-attention layers. Compare each head with
the same-layer controls of its own model, as in the tables above.

Generation uses each model's generation_config apart from sampling. Qwen2.5-1.5B-Instruct sets repetition_penalty
1.1 and Qwen2.5-3B-Instruct 1.05, so their greedy runs use it; no other model sets one.

# Qwen2.5-1.5B LCB in 14 languages, 2026-10-09

results/qwen-instruct-lcb-all. Same setup as the five-language run, now with all 14 non-English LCB languages (2,200
monolingual and 4,186 crosslingual prompts), L22H6 against three random heads of layer 22. Chinese and Japanese are
segmented with jieba and MeCab (fugashi) before the 5-word filter, as in LCB's compute_metrics.py. The first version
of this section split on whitespace, which skipped 198 of 200 zh and 96 of 100 ja monolingual baseline replies and
left the zh and ja crosslingual baselines at 0.22 and 0.07 (commit dfce47e); the saved replies were rescored without
regenerating them. The replies were generated with --bs 250 --token-budget 8000; the rescoring wrote lcb.py's
defaults (125, 24000) into summary.json's args, which lcb.py --report-only now keeps from the earlier run.
| condition | mono LPR | Δ mono | cross LPR | Δ cross |
|---|---|---|---|---|
| baseline | 0.973 | | 0.666 | |
| L22H6 zero | 0.777 | -0.213 [-0.231, -0.195] | 0.386 | -0.277 [-0.292, -0.263] |
| L22H6 mean | 0.738 | -0.257 [-0.276, -0.237] | 0.376 | -0.286 [-0.302, -0.273] |
| 3 control heads (L22) | | +0.002 to +0.007 | | -0.001 to +0.002 |

Monolingual, baseline to L22H6 zero: it 1.00 to 0.00, hi 0.99 to 0.39, tr 0.95 to 0.39, pt 0.95 to 0.54, fr 0.99 to
0.73, vi 1.00 to 0.82, es 0.97 to 0.83, id 0.90 to 0.79, ko 0.95 to 0.87, ar and de 0.99 to 0.96, while ru stays at
1.00, zh at 0.98 and ja goes from 0.96 to 0.95. Crosslingual drops are largest for it (0.69 to 0.00), pt (0.65 to
0.08), vi (0.61 to 0.10), hi (0.74 to 0.13) and id (0.65 to 0.19), while ar, de, ru and zh (0.81 to 0.78) lose 0.09
or less and ja goes from 0.54 to 0.51. So zh, ja and ru barely depend on the head, while hi and ko, also written in
their own scripts, do. Where the lost lines go: monolingual it, pt and fr replies go mostly to Spanish, es to
Portuguese, hi to Korean, Russian and English, and tr to English and Korean; crosslingual replies go mostly to
English, except it and pt, which go to Spanish. The Korean is real: Turkish replies switch to fluent Korean
(sometimes Japanese) mid-reply. In the pooled numbers mean ablation does at least as much as zero ablation, but per
language it differs (es 0.83 with zero, 0.97 with mean).

Without the head, 16 of 100 monolingual and 51 of 299 crosslingual ko replies are skipped (3 and 8 at baseline), 11
and 50 of them because the reply turned into Chinese or Japanese, which the benchmark splits on whitespace when ko
is expected. The ko drops above (0.95 to 0.87, 0.50 to 0.34) therefore understate the change.

# OLMo-2-1B post-training stages, 2026-10-09

results/olmo2-1b-{sft,dpo}-instruct and their -lcb runs. OLMo-2-0425-1B releases its checkpoints after SFT and after
DPO; Instruct is the released final model. Layer 12 on all 2,500 FLORES prompts and LCB for L12H8, same chat
template in all three.
| checkpoint | FLORES L12H8 c->w | rest of layer 12 (max) | LCB mono LPR | Δ mono, zero / mean | LCB cross LPR | Δ cross, zero / mean |
|---|---|---|---|---|---|---|
| base | 0.007 | | | | | |
| SFT | 0.028 | 0.002 | 0.997 | -0.076 / -0.038 | 0.895 | -0.247 / -0.091 |
| DPO | 0.071 | 0.008 | 0.989 | -0.199 / -0.120 | 0.933 | -0.244 / -0.123 |
| Instruct | 0.080 | 0.005 | 0.986 | -0.272 / -0.128 | 0.931 | -0.335 / -0.143 |

The same head carries the effect at every stage and nothing else in layer 12 does. Its crosslingual role is there
after SFT; keeping the prompt's language (FLORES, LCB monolingual) grows mainly with DPO. The base model can't take
LCB prompts, so its row is FLORES only.

# Sampling, 2026-10-09

results/qwen-instruct-lcb-t07-s{0,1}. Qwen2.5-1.5B-Instruct LCB (five languages) with the model's own sampling
settings (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1), two seeds. Each batch uses the same seed in
every condition, so the comparison stays paired.
| run | mono LPR | Δ mono, L22H6 zero / mean | cross LPR | Δ cross, zero / mean |
|---|---|---|---|---|
| greedy | 0.982 | -0.273 / -0.246 | 0.704 | -0.276 / -0.247 |
| seed 0 | 0.985 | -0.273 / -0.238 | 0.690 | -0.245 / -0.232 |
| seed 1 | 0.982 | -0.274 / -0.223 | 0.698 | -0.276 / -0.242 |

The three control heads of layer 22 stay between -0.013 and +0.006. The effect is not specific to greedy decoding.

# Content across models, 2026-10-09

results/<model>-content (content.py, three random heads from the same layer as controls). FLORES prompts whose
continuation flips to English when the head is removed: similarity between the prompt and the continuation before
and after, with Qwen3-Embedding-0.6B. For scale, the next FLORES sentence scores 0.38 and a random one 0.18.
| model | head | flips to English | before | after |
|---|---|---|---|---|
| Qwen2.5-1.5B (2026-10-08) | L22H6 | 918 | 0.475 | 0.558 |
| Qwen2.5-3B | L27H13 | 514 | 0.590 | 0.657 |
| Qwen3-1.7B | L18H12 | 739 | 0.635 | 0.683 |
| Gemma-3-1B | L11H3 | 873 | 0.639 | 0.559 |
| OLMo-2-1B | L12H8 | 101 | 0.702 | 0.668 |

The control heads flip at most two prompts each. In every model the English continuation stays on the prompt's
content, far above the next FLORES sentence: Qwen gains a little, Gemma and OLMo-2 lose a little.

# Llama-3.2-1B-Instruct LCB head screen, 2026-10-09

results/llama3.2-1b-instruct-lcbscreen (lcb.py --screen). Every head zero-ablated in turn on 100 crosslingual LCB
prompts (25 per language). Baseline pass rate 0.900; L8H25 takes it to 0.190, and the next head is L13H4 at 0.800.
So L8H25 is the one crosslingual head, and the full LCB run above already covers it. The same screen for
Llama-3.2-3B is queued.

# OLMo-3-7B, 2026-10-09

results/olmo3-7b-instruct (with -screen, -followup and -lcb) and results/olmo3-7b (with -screen and -followup).
bf16: fp32 would take 17 h for the screen, and bf16 batches match single-prompt runs on only 10 of 20 prompts, so
differences of a few prompts are noise. Instruct screen: L14H25 at 0.120, every other head 0.04 or less. On 2,500
prompts L14H25 is at 0.134 with dNLL +0.202 against +0.0003 for the rest of layer 14, but at 0.007 under mean
ablation, and scaling it by 2 to 5 only moves accuracy from 0.968 to 0.977-0.980, about as much as mean ablation
does. LCB for L14H25: zero ablation moves monolingual by -0.041 and crosslingual by -0.018, mean ablation by +0.001
and +0.015, so what is left comes from the out-of-distribution input.

The base model (baseline accuracy 0.83, many prompts flipping both ways) has the same head at 0.104 on the screen.
On 2,500 prompts its strongest heads are L15H20 at 0.151 and L20H18 at 0.133 with zero ablation (dNLL +0.234 and
+0.061 against +0.0003 for the rest of their layers), but 0.036 and 0.028 under mean ablation, with w->c at 0.026
and 0.025 and the control head and scaling conditions at 0.02 to 0.05, which is the bf16 noise floor. So OLMo-3-7B
has no head that holds up under mean ablation in either model, and none that is specific to the instruct model.

# Gemma-3-4B and SmolLM3-3B, 2026-10-09

results/gemma3-4b-instruct (with -screen, -followup and -lcb) and results/gemma3-4b (with -screen and -followup),
fp32. Instruct screen: L24H0 at 0.240, every other head 0.016 or less. On 2,500 prompts (layers 0 and 24) the
baseline is 0.997 and L24H0 has c->w 0.214 and w->c 0.000, with dNLL +0.202 against -0.005 for the rest of layer 24;
the next head is at 0.002. Mean ablation leaves 0.012, and scaling it by 2 to 5 changes nothing (accuracy 0.998).
The base model has the same head: 0.112 on its screen and 0.148 on 2,500 prompts (baseline 0.990, dNLL +0.142
against +0.003), 0.056 under mean ablation. LCB for L24H0 (table above): monolingual -0.259, crosslingual -0.107,
mean ablation -0.120 and -0.053, controls within 0.012.

SmolLM3-3B instruct (thinking off), screen only (results/smollm3-instruct-screen): baseline 0.984 on 125 prompts.
L1H12 has c->w 0.880 with dNLL +2.58; without it 75 of the 125 replies (12 to 18 of 25 in each language, English
included) fill with repeated `</think>` tokens, so it breaks generation rather than changing the language. Every other
head is at 0.016 or less, so no head changes the language on FLORES without breaking generation. The 2,500-prompt run
was stopped and the base model was not run.

# Why zero and mean ablation differ, 2026-10-09

results/<model>-diag (diagnose.py), the six heads on the 2,500 FLORES prompts in the precision of their sweeps.
Statistics of the head's contribution after the output projection over the user's text and the baseline
continuation (the template tokens before the text left out), then generation with the head replaced. Language means
are over the user's text and the continuation; other-language = German for en/fr/es/it prompts, French for de prompts.
c->w over all 2,500 prompts; in brackets, the share of non-English replies in the swapped-in language.

| model | head | norm rank in layer | mean's share of the energy | language's share of the rest | zero | follow-up mean | continuation mean | minus continuation mean | own-language mean | English mean | other-language mean | random, same norm | x0.5 | zero, norm held |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | L22H6 | 1 of 12 | 0.18 | 0.62 | 0.501 | 0.451 | 0.431 | 0.027 | 0.052 | 0.486 | 0.530 (0.182) | 0.544 | 0.071 | |
| Qwen2.5-3B | L27H13 | 2 of 16 | 0.38 | 0.71 | 0.515 | 0.338 | 0.372 | 0.059 | 0.001 | 0.595 | 0.794 (0.996) | 0.636 | 0.047 | |
| Qwen3-1.7B | L18H12 | 1 of 16 | 0.34 | 0.56 | 0.324 | 0.028 | 0.155 | 0.138 | 0.000 | 0.667 | 0.780 (0.904) | 0.377 | 0.009 | |
| Gemma-3-1B | L11H3 | 1 of 4 | 0.76 | 0.03 | 0.413 | 0.089 | 0.196 | 0.070 | 0.000 | 0.782 | 0.977 (0.997) | 0.666 | 0.008 | 0.160 |
| Gemma-3-4B | L24H0 | 2 of 8 | 0.56 | 0.73 | 0.214 | 0.012 | 0.011 | 0.003 | 0.000 | 0.222 | 0.796 (0.989) | 0.291 | 0.000 | 0.234 |
| OLMo-2-1B | L12H8 | 2 of 16 | 0.31 | 0.67 | 0.080 | 0.056 | 0.054 | 0.005 | 0.000 | 0.100 | 0.750 (0.910) | 0.130 | 0.005 | 0.074 |

In five of the six models another language's mean moves 0.904 to 0.997 of the non-English continuations into that
language, and the prompt's own language mean keeps them; Qwen2.5-1.5B is the exception (0.182). Holding the
post-attention norm fixed accounts for part of the zero-ablation effect in Gemma-3-1B only. In Qwen3, Gemma-3-1B and
Qwen2.5-3B the means over all five languages move replies mostly to another European language (Qwen3's continuation
mean sends 347 of 388 flips to Italian, Gemma-3-1B's 418 of 491 to French and Italian), so there mean ablation is not
a neutral removal; in Qwen2.5-1.5B they send replies mostly to English. With the head zeroed, Gemma-3-4B's replies go
mostly to Portuguese and Spanish (453 of 535), not to English. EXPERIMENTS.md section 10 has the reading against the
explanations it tests.

# Steering on LCB, Gemma-3-1B, 2026-10-09

results/gemma3-1b-instruct-steer (steer.py, steer_quality.py; plan in experiments/steer_plan.md, written before the
runs). L11H3's output replaced by its FLORES mean for a language (steer: the language the reply should be in; swap:
de for en/fr/es/it, fr for de), or shifted by that mean minus the mean over all languages (add), at every position;
the same four conditions on the three layer-11 control heads of the LCB run.

| condition | mono LPR (change [95% CI]) | cross LPR (change [95% CI]) | replies in the swap language, mono / cross | English lines, cross | skipped, cross | repetition |
|---|---|---|---|---|---|---|
| baseline | 0.984 | 0.118 | 0.000 / 0.000 | 0.55 | 0.019 | 0.004 |
| L11H3 steer | 0.996 (+0.013 [+0.004, +0.021]) | 0.560 (+0.440 [+0.410, +0.469]) | 0.000 / 0.000 | 0.17 | 0.007 | 0.006 |
| L11H3 swap | 0.000 | 0.000 | 0.994 [0.988, 0.999] / 0.594 [0.565, 0.622] | 0.17 | 0.008 | 0.010 |
| L11H3 add steer | 0.966 (-0.015 [-0.028, -0.004]) | 0.165 (+0.047 [+0.031, +0.064]) | 0.000 / 0.000 | 0.48 | 0.021 | 0.004 |
| L11H3 add swap | 0.080 | 0.045 | 0.828 [0.802, 0.853] / 0.123 [0.104, 0.142] | 0.47 | 0.019 | 0.012 |
| three controls, all four conditions | change -0.003 to +0.006 | change -0.034 to +0.034 | 0.000 / 0.000 | | | |

Crosslingual LPR under steer: German 0.08 to 0.82, French 0.16 to 0.47, Spanish 0.11 to 0.46, Italian 0.14 to 0.49.
The 530 crosslingual replies that steer turns from fail to pass have cosine 0.763 with the baseline reply to the same
prompt and 0.199 with the baseline reply to another prompt (Qwen3-Embedding-0.6B); their median perplexity under the
unmodified model is 4.9, against 5.8 for baseline crosslingual replies in the requested language.

# Detector check and CIs for the instruct heads, 2026-10-09

results/detectors (detectors.py). The 2,500-prompt continuations relabeled with langid, fastText and a 2-of-3 vote:
every head's c->w moves by 0.02 or less and keeps its rank (first; Qwen3's L18H12 second behind L0H3, which breaks
the model). langdetect c->w with bootstrap 95% CIs: Qwen2.5-1.5B 0.500 [0.481, 0.520], Qwen2.5-3B 0.515 [0.495,
0.536], Qwen3-1.7B 0.324 [0.306, 0.343], Gemma-3-1B 0.413 [0.394, 0.431], Gemma-3-4B 0.214 [0.198, 0.230], OLMo-2-1B
0.080 [0.070, 0.091].

# LCB word-level pass rate (WPR), Qwen2.5-1.5B 14 languages, 2026-10-09

results/qwen-instruct-lcb-all/wpr.md (lcb_wpr.py, as in LCB's compute_metrics.py, for ar/hi/ja/ko/ru/zh). Baseline
0.96 to 0.99 monolingual and 0.93 to 0.99 crosslingual. With L22H6 zeroed or mean-ablated, Hindi drops (monolingual
0.98 to 0.89 and 0.82, crosslingual 0.99 to 0.83 and 0.83); the other five languages move by 0.03 or less, and the
three layer-22 controls keep Hindi at 0.99 to 1.00.

# Steering on LCB, all six heads, 2026-10-10

results/<model>-steer. Same design as the Gemma-3-1B section above, read against experiments/steer_plan.md.

| model | head | cross baseline | steer: mono change [95% CI] | steer: cross change [95% CI] | swap: replies in the swapped-in language, mono / cross | add steer: cross change | add swap: mono / cross | controls: cross change / swap share |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-1B | L11H3 | 0.118 | +0.013 [+0.004, +0.021] | +0.440 [+0.410, +0.469] | 0.994 [0.988, 0.999] / 0.594 [0.565, 0.622] | +0.047 [+0.031, +0.064] | 0.828 / 0.123 | -0.034..+0.034 / 0.000 |
| Gemma-3-4B | L24H0 | 0.133 | -0.007 [-0.016, +0.001] | +0.079 [+0.064, +0.095] | 0.985 [0.976, 0.992] / 0.137 [0.118, 0.157] | +0.040 [+0.029, +0.053] | 0.045 / 0.023 | -0.023..+0.008 / 0.000 |
| Qwen2.5-3B | L27H13 | 0.888 | +0.009 [-0.001, +0.019] | +0.007 [-0.005, +0.019] | 0.982 [0.974, 0.991] / 0.829 [0.810, 0.849] | +0.018 [+0.006, +0.030] | 0.165 / 0.081 | -0.002..+0.022 / 0.000 |
| Qwen3-1.7B | L18H12 | 0.823 | -0.004 [-0.013, +0.004] | -0.013 [-0.029, +0.004] | 0.926 [0.907, 0.942] / 0.701 [0.676, 0.727] | +0.041 [+0.028, +0.055] | 0.015 / 0.048 | +0.000..+0.041 / 0.000 |
| OLMo-2-1B | L12H8 | 0.931 | +0.004 [-0.005, +0.013] | +0.000 [-0.010, +0.010] | 0.844 [0.818, 0.869] / 0.828 [0.806, 0.849] | +0.003 [-0.006, +0.012] | 0.009 / 0.018 | -0.002..+0.003 / 0.000 |
| Qwen2.5-1.5B | L22H6 | 0.704 | -0.001 [-0.010, +0.008] | -0.020 [-0.033, -0.008] | 0.043 [0.029, 0.057] / 0.019 [0.012, 0.027] | -0.042 [-0.057, -0.026] | 0.000 / 0.000 | -0.008..+0.007 / 0.000 |

Every baseline matches the model's earlier LCB run. By the rule fixed in advance the swap share is above the
controls' in all six, at 0.84 to 0.99 of the monolingual replies in five models and 0.043 in Qwen2.5-1.5B (whose
monolingual LPR still drops by 0.335 under swap). steer raises crosslingual LPR in the two Gemma models only. The
replies steer fixes have cosine 0.69 to 0.84 with the baseline reply to the same prompt, against 0.15 to 0.21 for
another prompt's, and under steer no model's skipped share or repetition rises by more than 0.002.

# Llama-3.2-3B crosslingual head, 2026-10-10

results/llama3.2-3b-instruct-lcbscreen and -lcb-top. Every one of the 672 heads zero-ablated on 100 crosslingual
prompts: baseline 0.939, L13H19 0.616, next L8H13 0.879. Full five-language LCB for L13H19 with three layer-13
controls: monolingual 0.993, +0.000 [-0.006, +0.006] (mean -0.005); crosslingual 0.911, -0.335 [-0.363, -0.308] (mean
-0.250); controls +0.000 to +0.006 monolingual and -0.003 to +0.000 crosslingual; English crosslingual lines 0.05 to
0.36. Like Llama-3.2-1B's L8H25, it acts on requested languages only.

# Qwen3-1.7B LCB in 14 languages, 2026-10-10

results/qwen3-1.7b-instruct-lcb-all, read with lcb14_judge.py as fixed in experiments/lcb14_plan.md (judge.md). The
same reading of the Qwen2.5-1.5B run is in results/qwen-instruct-lcb-all/judge.md.

| condition | mono LPR | Δ mono | cross LPR | Δ cross |
|---|---|---|---|---|
| base | 0.971 | | 0.789 | |
| L18H12 zero | 0.658 | -0.293 [-0.313, -0.273] | 0.198 | -0.592 [-0.608, -0.577] |
| L18H12 mean | 0.661 | -0.304 [-0.324, -0.284] | 0.294 | -0.497 [-0.514, -0.481] |
| 3 controls (L18H6, L18H13, L18H14) | 0.966 to 0.975 | -0.004 to +0.003 | 0.791 to 0.797 | +0.002 to +0.009 |

Affected languages (CI below zero and below every control): 12 of 14 monolingual, all but vi (-0.020 [-0.060,
+0.020]) and zh (-0.015 [-0.065, +0.035]), with ja -0.570, ru -0.300 and hi -0.220; 14 of 14 crosslingual (-0.456 to
-0.760). Qwen2.5-1.5B under the same reading: 10 of 14 monolingual (not de, ja, ru, zh) and 12 of 14 crosslingual (not
ja, ru). The planned check (hi affected, zh/ja/ru not) holds for Qwen2.5-1.5B and not for Qwen3-1.7B.

WPR (wpr.md, now with the number of replies per cell): baseline 0.97 to 1.00 monolingual and 0.89 to 0.96
crosslingual; under ablation some cells rest on few replies (crosslingual ko 1, zh 12, ja 16) and are not read.
Skipped replies under zero ablation: 28 of 2,200 monolingual (3 at baseline), 68 of 4,186 crosslingual (66).

# Language heads under MITra's method, 2026-10-10

results/mitra-check/summary.txt (mitra_rank.py). The per-head language scores published with Translation Heads
(arXiv 2602.04613; github.com/Blyzi/mitra, commit daa721b), base models, 20 directions (English to and from 10
languages), 50 to 142 FLORES examples each, ranked with their top-head rule (number of directions a head tops).

| model | their top heads (directions topped) | our head | our head, rank on the mean over 20 directions |
|---|---|---|---|
| Qwen3-1.7B-Base | L18H12 (20) | L18H12 | 1 of 448 |
| Llama-3.2-1B | L8H25 (7: 6 from English), L9H8 (7: all into English), L13H4 (4) | L8H25 | 1 of 512 |
| Gemma-3-1B-pt | L15H2 (20) | L11H3 | 3 of 104 |

On en->fr, L8H25 tops 90 of Llama's 142 examples and L12H7 47; the one-example check in ctli PR #14 had L12H7 first.

# Gemma-3-1B LCB in 14 languages, 2026-10-10

results/gemma3-1b-instruct-lcb-all, read as for Qwen3-1.7B above (judge.md, wpr.md).

| condition | mono LPR | Δ mono | cross LPR | Δ cross |
|---|---|---|---|---|
| base | 0.978 | | 0.129 | |
| L11H3 zero | 0.266 | -0.730 [-0.748, -0.710] | 0.013 | -0.113 [-0.123, -0.102] |
| L11H3 mean | 0.515 | -0.488 [-0.510, -0.466] | 0.011 | -0.117 [-0.127, -0.107] |
| 3 controls (L11H0, L11H1, L11H2) | 0.977 to 0.981 | +0.000 to +0.001 | 0.114 to 0.177 | -0.014 to +0.050 |

All 14 languages are affected on monolingual prompts (pt -0.520 to vi -0.980; zh -0.840, ja -0.788, ru -0.550, hi
-0.919) and on crosslingual ones (ar -0.048 to tr -0.276), where the baseline is low. The planned check (hi affected,
zh/ja/ru not) does not hold. Skipped replies under zero ablation: 6 of 2,200 monolingual (2 at baseline), 108 of 4,186
crosslingual (77). WPR: monolingual baseline 0.98 to 1.00; crosslingual cells rest on 19 to 46 replies at baseline and
0 to 7 without the head and are not read.

# OLMo-2-1B and Llama-3.2-1B LCB in 14 languages, 2026-10-10

results/olmo2-1b-instruct-lcb-all and llama3.2-1b-instruct-lcb-all, read as for Qwen3-1.7B above.

| model | condition | mono LPR | Δ mono | cross LPR | Δ cross |
|---|---|---|---|---|---|
| OLMo-2-1B | base | 0.976 | | 0.873 | |
| OLMo-2-1B | L12H8 zero | 0.662 | -0.346 [-0.367, -0.327] | 0.491 | -0.379 [-0.394, -0.364] |
| OLMo-2-1B | L12H8 mean | 0.537 | -0.445 [-0.467, -0.424] | 0.430 | -0.442 [-0.458, -0.427] |
| OLMo-2-1B | 3 controls | 0.973 to 0.977 | -0.002 to +0.002 | 0.872 to 0.883 | -0.002 to +0.010 |
| Llama-3.2-1B | base | 0.985 | | 0.778 | |
| Llama-3.2-1B | L8H25 zero | 0.902 | -0.069 [-0.081, -0.058] | 0.084 | -0.695 [-0.709, -0.681] |
| Llama-3.2-1B | L8H25 mean | 0.872 | -0.101 [-0.115, -0.087] | 0.292 | -0.485 [-0.500, -0.470] |
| Llama-3.2-1B | 3 controls | 0.984 to 0.986 | -0.003 to +0.000 | 0.764 to 0.781 | -0.015 to +0.003 |

OLMo-2: 13 of 14 languages affected on monolingual prompts, all but hi (-0.010 [-0.061, +0.040]); 14 of 14 on
crosslingual ones (-0.140 to -0.692). Crosslingual WPR without the head, on 75 to 216 replies per language: 0.65 to
0.85, against 0.86 to 0.97 at baseline and 0.83 to 0.97 under the controls. 42 of the 6,586 prompts generate past
OLMo-2's 4,096-token context (three long crosslingual complex_prompts per language).

Llama-3.2-1B: 14 of 14 affected on crosslingual prompts (-0.579 to -0.764); on monolingual prompts 6 of 14: ko
-0.388, tr -0.303, ja -0.180, ar -0.100, zh -0.095, it -0.090. Crosslingual WPR without the head rests on 0 to 45
replies and is not read.

Summary of the four families with Qwen2.5-1.5B: the planned check (hi affected, zh/ja/ru not, monolingual) holds
only for Qwen2.5-1.5B. Languages not affected on monolingual prompts: Qwen2.5-1.5B de/ja/ru/zh, Qwen3-1.7B vi/zh,
Gemma-3-1B none, OLMo-2-1B hi, Llama-3.2-1B de/es/fr/hi/id/pt/ru/vi. On crosslingual prompts every language is
affected in the four, and all but ja/ru in Qwen2.5-1.5B.

# Qwen2.5-7B rerun with the head rule, 2026-10-10

results/qwen2.5-7b-instruct (layers 0 and 19 on 2,500 prompts), -followup, -lcb, -diag, -steer; results/qwen2.5-7b
and qwen2.5-7b-followup (base, layer 19). bf16.

- Head choice (experiments/head_rule.md): L0H25 (c->w 0.629) and L0H22 (0.466) keep 0.512 and 0.518 of the English
  continuations in English and are set aside; L19H1 (0.200, dNLL +0.026, rest of layer +0.001) keeps 1.000. The other
  layer-19 heads reach at most 0.006. Mean ablation 0.018; x2 0.002, x3 0.426, x5 0.762 (dNLL +3.66). Base model:
  L19H1 0.025 zero, 0.011 mean.
- LCB, five languages: monolingual 0.984, zero -0.009 [-0.020, +0.003], mean +0.003; crosslingual 0.950, zero -0.074
  [-0.089, -0.059], mean -0.035; three layer-19 controls +0.001 to +0.003 and -0.008 to +0.001. Crosslingual lines in
  English 0.04 to 0.10.
- Diagnosis (c->w): zero 0.202, follow-up mean 0.021, continuation mean 0.008, minus continuation mean 0.070,
  own-language mean 0.000, English mean 0.188, other-language mean 0.352 with 0.417 of non-English continuations in
  the swapped-in language, random same norm 0.228, x0.5 0.017. With the head zeroed, 439 of 505 flips go to Chinese
  (33 to English); the random vector (485 of 570) and the English mean (399 of 469) also send them to Chinese, and
  English prompts stay English. For comparison, zero sends Qwen2.5-1.5B's flips mostly to English (919 of 1,252) and
  Qwen2.5-3B's to English (514), Chinese (299) and Spanish (221) of 1,288.
- Steering on LCB: steer +0.003 [-0.004, +0.010] mono and -0.003 [-0.011, +0.006] cross; swap -0.280 mono and -0.269
  cross, with 0.122 [0.099, 0.144] and 0.128 [0.110, 0.147] of the replies in the swapped-in language (controls
  0.000); add steer +0.004 cross; add swap 0.000 / 0.001.
- First pipeline on L0H25 (results/*-L0H25run): LCB zero -0.495 mono and -0.485 cross with 0.39 and 0.44 English
  lines; the prompt language's own mean still gives c->w 0.653; steer lowers LPR by 0.529 mono and 0.514 cross and
  leaves 0.260 and 0.168 of the replies unscored.

# Qwen3-4B screen, 2026-10-10

results/qwen3-4b-instruct-screen: all 1,152 heads on the 125 FLORES prompts (fp32, bs 125). Baseline non-English
retention 0.980; the largest c->w is L8H3's 0.024, so no layer goes on to 2,500 prompts and the base model is not
run. A crosslingual LCB screen of every head, as for the two Llamas, started at 16:18 KST.

# Qwen2.5-3B LCB in 14 languages, 2026-10-10

results/qwen2.5-3b-instruct-lcb-all, added before the run (experiments/lcb14_plan.md, 16:48 KST) and read as for
Qwen3-1.7B above (judge.md, wpr.md). 16:49 to 20:19 KST, sharing the GPU with the two crosslingual screens.

| condition | mono LPR | Δ mono | cross LPR | Δ cross |
|---|---|---|---|---|
| base | 0.987 | | 0.869 | |
| L27H13 zero | 0.478 | -0.520 [-0.543, -0.498] | 0.406 | -0.457 [-0.473, -0.441] |
| L27H13 mean | 0.472 | -0.555 [-0.577, -0.533] | 0.401 | -0.463 [-0.478, -0.446] |
| 3 controls (L27H6, L27H12, L27H14) | 0.986 to 0.990 | -0.001 to +0.002 | 0.830 to 0.880 | -0.037 to +0.009 |

13 of 14 languages affected on monolingual prompts, all but zh (+0.005 [-0.020, +0.030]), with ja -0.700, ru -0.372
and hi -0.368; 14 of 14 on crosslingual ones (zh -0.166 to it -0.826). The planned check (hi affected, zh/ja/ru not)
does not hold, so by the plan the Qwen2.5-1.5B pattern is read as a property of the 1.5B model, not of Qwen2.5.
Chinese is unaffected on monolingual prompts in the three Qwen models and affected in the other three families.

Skipped replies under zero ablation: 279 of 2,200 monolingual (5 at baseline) and 251 of 4,186 crosslingual (62),
273 and 204 of them written mainly in Han characters or kana. LCB's scorer (compute_metrics.py, followed by lcb.py)
counts words by spaces unless the expected language is zh or ja and skips replies without a line of five words, so
these replies are left out rather than failed. Skipped by cell: monolingual ko 91 of 100, ar 80 of 300, vi 41 of
100; crosslingual ko 162 of 299. WPR: baseline 0.95 to 1.00 monolingual and 0.88 to 0.98 crosslingual; without the
head some cells rest on few replies (monolingual ko none, crosslingual ko 14) and are not read.

Outside the plan, lcb14_judge.py --skipped-fail counts the replies an intervention leaves unscorable as failures
(judge_skipped_fail.md in each of the six 14-language runs). No affected language and no pattern reading changes.
Changes of 0.05 or more: Qwen2.5-1.5B ko -0.084 to -0.206 (mono) and -0.093 to -0.220 (cross), Qwen3-1.7B ko -0.605
to -0.700, OLMo-2 ko -0.440 to -0.490, Qwen2.5-3B ar -0.668 to -0.750, ru -0.372 to -0.460, tr -0.655 to -0.710
(mono) and ko -0.647 to -0.831 (cross).
