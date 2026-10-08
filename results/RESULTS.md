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

Of the five top c->w heads, only L6H10 holds up under Chaewon's mean ablation (PR #6): c->w 0.210 with zero ablation
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


Mean vs zero ablation and the matched-null redistribution test are in Chaewon's PR #6 (results/gpt2-mean-ablation,
results/gpt2-redistribution). TABLES.md has the paper's tables recomputed from results/ (python tables.py).
