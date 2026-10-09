# Experiments

An index of every experiment in this repo, for writing the paper. Each entry says why it was run, what and how,
when, where the results are and which code produced them, then what the result shows and what it doesn't.
results/RESULTS.md is the running log with the full tables; this file is the short version. Status as of 2026-10-09.

## Shared setup

- Data: 2,500 FLORES prompts, 500 per language in en/fr/de/es/it (495 FLORES-200 devtest sentences plus 5
  hand-written starts, built with the rules of the original expand_dataset.py). Head screens use the first 25 per
  language (125 prompts). The Language Confusion Benchmark (LCB, Marchisio et al. 2024) adds monolingual prompts
  written in the language and crosslingual prompts in English asking for a reply in another language.
- Intervention: one attention head at a time. Zero ablation sets the head's slice of the input to the attention output
  projection to zero. Mean ablation (followup.py; PR #6 for GPT-2 small) replaces it with its mean over the prompt
  tokens of the 2,500 FLORES prompts; for instruct models those tokens include the chat template. Scaling multiplies
  it.
- Metrics on FLORES: the language of the greedy 40-token continuation (langdetect, seed 0). c->w is the share of all
  prompts that are in the prompt's language at baseline and in another language with the head removed, w->c the
  reverse, and the switch rate (SR) their sum. dNLL is the head's change in LM loss on 100 FLORES dev sentences per
  language, shown next to the mean change for the other heads in the same layer.
- Metrics on LCB: the benchmark's line-level pass rate (LPR), averaged over its sources, with paired bootstrap 95%
  CIs on the change.
- Instruct models get each prompt as a single user turn with their own chat template (one BOS, thinking off, a fixed
  date where the template inserts one). Qwen2.5's template adds its default English system prompt.
- Hardware: one H100 80GB. fp16 for Qwen2.5-1.5B, bf16 for OLMo-2-1B base and OLMo-3-7B, fp32 for everything else.

## 1. Hook check and reproduction

- Why: to reproduce the submitted numbers before building on them.
- What: GPT-2 small (the paper's 25 hand-written prompts), Qwen2.5-1.5B base and instruct (125 prompts, as in
  qwen_experiment.py) and BLOOM-1b7 (25 prompts), with the paper's settings.
- How: the original hook and a corrected one, side by side. The original hook zeroes a slice of the attention output
  after the output projection. After the projection every output dimension combines all heads, so the zeroed slice
  does not correspond to one head; in BLOOM the zeroed tensor also contains the residual stream. The corrected hook
  zeroes the head's slice of the input to the output projection.
- When / where: 2026-10-06 to 10-07; results/gpt2, results/qwen-instruct, results/qwen-base,
  results/bloom-paper25-fp16; RESULTS.md, first sections.
- Code: experiments/sweep.py (`--modes paper` for the original hook, `--modes head` for the corrected one).
- Result: with the original hook, GPT-2 reproduces within one prompt (L6H1 0.28, paper 0.32) and Qwen instruct L0H5
  gives 0.224, as in the paper. BLOOM does not reproduce with its own settings (largest switch rate 0.20, paper 0.16);
  its script uses head size 64, while bloom-1b7 has 128. With the corrected hook, GPT-2 L6H1 has a switch rate of
  0.038 on 2,500 prompts (rank 136 of 144) and Qwen instruct L0H5 0.056 on the 125 prompts (rank 38 of 336). The GPT-2
  layer-0 heads with high switch rates raise LM loss by +0.78 to +1.30, and some of them (L0H0) mostly flip outputs
  from wrong to correct.
- Reading: the head-ablation numbers in the submitted version come from the original hook. Under the corrected hook
  the first-token broadcaster heads it reported (GPT-2 L6H1, Qwen L0H5) change the output language little, and the
  large layer-0 effects in GPT-2 come with large LM loss increases.

## 2. GPT-2 small with the corrected hook

- Why: to find which GPT-2 heads change the output language under the corrected hook.
- What: all 144 heads on the 2,500 FLORES prompts. Follow-ups on the top heads: several heads ablated together,
  scaling by 2 to 5, first-token attention, 100 zh and 100 ru prompts, sampling (temperature 0.7, three seeds),
  prompts cut to their first half, two more language detectors and random half splits, replacement by per-language
  means, output quality and content, and mean ablation and downstream attention redistribution (PR #6).
- How: zero ablation per head. Follow-ups use the heads with the largest c->w among those with dNLL at most 0.1, and
  control heads drawn at random from the dNLL <= 0.1 heads outside the top ten.
- When / where: 2026-10-06 to 10-08; results/gpt2, gpt2-multi*, gpt2-amp, gpt2-attention, gpt2-zhru, gpt2-sampling,
  gpt2-truncated, gpt2-identity, gpt2-content, gpt2-mean-ablation, gpt2-redistribution, results/quality.md;
  tables/README.md.
- Code: sweep.py, multi.py, amplify.py, attention.py, checks.py, robustness.py, identity.py, quality.py, content.py;
  PR #6 for mean ablation and redistribution.
- Result:
  - L6H10 has the largest c->w among the heads with dNLL <= 0.1 (0.210, dNLL +0.016) and the lowest accuracy of the
    144 single-head ablations (0.229; non-English 0.296 to 0.038). 511 of the 524 flipped outputs are English. Mean
    ablation gives c->w 0.177; over all 144 heads, c->w under mean and zero ablation correlates at Spearman 0.69.
  - Ablating up to ten heads in order of c->w keeps accuracy at 0.23 to 0.31 while dNLL rises to +0.49.
  - Scaling L6H10 by 5 raises accuracy from 0.435 to 0.870 (dNLL +0.041). At 5x, 76% of the outputs in the prompt's
    language repeat themselves and 15% copy the prompt (quality.py's split applied to the gpt2-amp outputs).
  - With langid, fastText and a 2-of-3 vote, c->w per head correlates with langdetect at Spearman 0.994 or higher and
    L6H10 stays first among the dNLL <= 0.1 heads. Across 200 random half splits it is first in both halves every
    time.
  - Sampling: L6H10 c->w 0.256 to 0.272 across seeds (greedy 0.210). Prompts cut mid-sentence: baseline non-English
    accuracy 0.716, L6H10 c->w 0.095, control heads 0.02 to 0.08.
  - zh/ru: c->w per head correlates with the European set at Spearman 0.64 (switch rate 0.79), and 4 of the top ten
    c->w heads are shared.
  - First-token attention correlates negatively with the ablation effect (Spearman -0.52 with the switch rate).
  - At baseline, 57% of the outputs that stay in the prompt's language are repetition and 35% copy the prompt. For the
    prompts L6H10 sends to English, similarity to the prompt goes from 0.453 to 0.195 (unrelated FLORES sentence
    0.18).
  - Replacing L6H10 with any language's mean, including the prompt's own, sends 0.78 to 0.96 of the non-English
    outputs to English.
  - Redistribution (PR #6): ablating L6H10 raises first-token attention in downstream heads (no matched permutation
    out of 100,000 reaches the observed value), and across the 12 layer-6 heads the increase correlates with c->w
    (rho 0.70 with zero ablation, 0.81 with mean ablation).
- Reading: GPT-2 has one mid-layer head whose removal sends non-English continuations to English, and the result is
  the same across detectors, prompt splits and sampling. The continuations it keeps in the prompt's language are
  mostly repetition or copies, and without it GPT-2 writes English unrelated to the prompt, unlike the instruct models
  of sections 4 and 9a, where the content is kept. The effect is much smaller when the prompt stops mid-sentence. The
  head's per-language average output does not reproduce its effect. The redistribution result shows that attention
  changes downstream; PR #6 notes that it does not by itself show compensation.

## 3. Other models without chat tuning

- Why: to check whether base models other than GPT-2 small have a single head that changes the output language.
- What: GPT-2 medium (24 x 16 heads), BLOOM-1b7 (24 x 16), Pythia-1B (16 x 8) and OLMo-2-1B base (16 x 16), every
  head on the 2,500 FLORES prompts.
- How: the same sweep, then mean ablation and scaling by 2, 3 and 5 on the top three heads by c->w with dNLL <= 0.1.
  BLOOM, GPT-2 medium and Pythia run in fp32, OLMo-2 base in bf16. OLMo-2 base ends the document right after most
  FLORES sentences, so it runs with the end-of-text token blocked.
- When / where: 2026-10-07 to 10-08; results/gpt2-medium, results/bloom, results/pythia-1b, results/olmo2-1b and
  their -followup folders.
- Code: sweep.py, followup.py.
- Result:
  - GPT-2 medium: L13H6 (layer 13 of 24; L6H10 is in layer 6 of 12) takes non-English accuracy from 0.300 to 0.066
    with dNLL +0.006, and 476 of the 479 flipped outputs are English. Mean ablation gives c->w 0.151 (zero 0.192).
  - BLOOM-1b7: 482 of the 2,500 baseline outputs are empty because the model ends the document. Mean switch rate
    0.011. The top c->w head (L23H12) stops generation instead of changing the language. The heads that do change
    it are in layers 18 to 21 and act on one language each; the largest, L21H15, takes German from 0.834 to 0.634.
  - Pythia-1B: baseline non-English accuracy 0.990. L1H7 has c->w 0.635 because the model then emits only newlines
    (1,456 empty outputs, English prompts included). No other head has c->w above 0.044 (L9H3).
  - OLMo-2-1B base: baseline non-English accuracy 0.921. L15H5 in the last layer has c->w 0.134 (next head 0.072)
    with dNLL +0.035, on French, Spanish and Italian but not German; mean ablation gives 0.115.
- Reading: among these base models, GPT-2 medium has one head with a large effect on all four languages, as GPT-2
  small does; BLOOM and Pythia have none; OLMo-2 base has one that acts on the Romance languages. Section 7 shows
  that the Qwen2.5 and Gemma-3-4B base models also have one, at c->w 0.15 to 0.20.

## 4. Qwen2.5-1.5B, base and instruct

- Why: Qwen2.5-1.5B is the instruct model of the submitted version and the model studied in the most detail here.
- What: all 336 heads (28 x 12) of the base and instruct models on the 2,500 FLORES prompts; mean ablation and
  scaling on the top heads of the instruct model; L22H6 and the L17 heads with two other input formats (instruct on
  raw text, base with the chat template) and three system prompt settings (Qwen's default English system prompt, no
  system prompt, the default translated into the prompt's language); the form and content of the replies.
- How: zero ablation sweep; followup.py; checks.py `qwen-format` and `qwen-system`; quality.py; content.py
  (Qwen3-Embedding-0.6B similarity between the prompt and the continuation).
- When / where: 2026-10-07 to 10-08; results/qwen-instruct-full, qwen-base-full, qwen-instruct-2500 (with content/
  and quality.md), qwen-instruct-followup, qwen-format-2500, qwen-system-2500; figures/fig1_qwen_c2w.
- Code: sweep.py, followup.py, checks.py, quality.py, content.py, figures.py.
- Result:
  - L22H6 has the largest c->w in both models: 0.500 in instruct (next L17H7 0.325 and L17H8 0.235; mean over all
    heads 0.019) and 0.150 in base (next 0.023; mean 0.003).
  - In instruct, non-English retention goes from 0.902 to 0.278 (fr 0.978 to 0.544, de 0.822 to 0.210, es 0.840 to
    0.350, it 0.968 to 0.008). French, German and Spanish replies go to English; Italian ones to Spanish (208),
    English (165), Portuguese (79) and French (26).
  - dNLL +0.228, against +0.004 for the other heads of layer 22. In base, +0.216.
  - Mean ablation: L22H6 c->w 0.450; L17H7 0.016 and L17H8 0.076.
  - System prompt: c->w 0.500 with the default English one (baseline retention 0.902), 0.258 with none (0.983) and
    0.379 with the translated one (0.996).
  - Input format: on raw text, instruct 0.176 and base 0.150; with the chat template, instruct 0.500 and base 0.089.
  - Of the 1,804 baseline replies in the prompt's language, 0% are repetition and 1% copy the prompt. For the 918
    prompts that L22H6 sends to English, similarity to the prompt is 0.475 before and 0.558 after (next FLORES
    sentence 0.384, random sentence 0.179).
- Reading: in this model, removing one head takes most non-English replies out of their language, and the effect
  stays under mean ablation. The same head exists in the base model with less than a third of the effect. The two
  models are close on raw text, and only the instruct model with its own chat template depends strongly on the head.
  The default English system prompt lowers baseline retention and enlarges the effect, but L22H6 matters in all three
  settings. The L17 heads lose most of their zero-ablation effect under mean ablation. In the base model L22H6 raises
  LM loss by a similar amount (+0.216) but changes far fewer replies (0.150), so the loss increase does not by itself
  produce the switch. The replies that switch to English keep the prompt's content.

## 5. Qwen2.5-1.5B on the Language Confusion Benchmark

- Why: FLORES measures how a model continues a sentence. LCB (Marchisio et al. 2024) measures whether a chat model
  replies in the language the user writes in (monolingual) or asks for (crosslingual).
- What: Qwen2.5-1.5B-Instruct.
  - Five languages: fr/de/es/it monolingual (800 prompts) and crosslingual (1,196), plus 200 English prompts. L22H6,
    L17H7 and L17H8 with zero and mean ablation, scaling (L22H6 x2, L17H7 x2, L17H8 x3), and three random control
    heads from each of layers 17 and 22.
  - All 14 non-English LCB languages (2,200 monolingual, 4,186 crosslingual): L22H6 and three controls from layer 22.
  - Sampling with the model's own settings (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1), two
    seeds, five languages.
- How: 100 new tokens, greedy except for the sampling runs, where each batch uses the same seed in every condition.
  LPR as in LCB's compute_metrics.py, averaged over sources; paired bootstrap 95% CIs on the change, pooled over the
  non-English prompts. Chinese and Japanese are segmented with jieba and MeCab before the 5-word line filter. Mean
  ablation uses the head's mean over the 2,500 FLORES prompts.
- When / where: 2026-10-08 to 10-09; results/qwen-instruct-lcb, qwen-instruct-lcb-all, qwen-instruct-lcb-t07-s0
  and -s1. Every reply is in samples.jsonl.gz.
- Code: lcb.py.
- Result:
  - Five languages, greedy: L22H6 zero takes monolingual LPR from 0.982 to 0.747 (paired change -0.273 [-0.305,
    -0.240]) and crosslingual from 0.704 to 0.427 (-0.276 [-0.303, -0.250]). Mean ablation: -0.246 and -0.247. The
    six control heads: -0.004 to +0.005 monolingual, -0.022 to 0.000 crosslingual.
  - L17H7 zero -0.065 monolingual and -0.036 crosslingual, L17H8 zero -0.004 and -0.008. Their mean ablation raises
    crosslingual LPR (+0.068 and +0.039). Scaling: L17H8 x3 +0.008 and +0.005, L22H6 x2 +0.001 and -0.186.
  - Sampling: L22H6 zero -0.273 and -0.274 monolingual, -0.245 and -0.276 crosslingual for the two seeds; controls
    -0.013 to +0.006.
  - 14 languages: monolingual 0.973 to 0.777 (-0.213 [-0.231, -0.195]) and crosslingual 0.666 to 0.386 (-0.277
    [-0.292, -0.263]); mean ablation -0.257 and -0.286; controls -0.001 to +0.007. Monolingual by language: it 1.00
    to 0.00, hi 0.99 to 0.39, tr 0.95 to 0.39, pt 0.95 to 0.54, fr 0.99 to 0.73, while ru stays at 1.00 and zh at
    0.98, and ja goes from 0.96 to 0.95. In the line labels of the samples, the lost lines go mostly to Spanish for
    Italian and Portuguese (and for French in the monolingual set), otherwise mostly to English; some Turkish
    replies switch to Korean mid-reply.
  - Without the head, 11 of 100 monolingual and 50 of 299 crosslingual Korean replies turn into Chinese or Japanese
    and are skipped by the line filter, against 3 and 8 skipped replies of any kind at baseline.
- Reading: L22H6 matters on LCB, in both tasks, under sampling and in most of the 14 languages; Chinese, Japanese and
  Russian change by 0.03 or less. The monolingual and crosslingual drops are similar in size, which fits a head that
  tracks the language the reply should be in, whether the prompt is written in it or asks for it. The L17 heads and
  the scaling that raised FLORES retention do not carry over to LCB, and doubling L22H6 lowers crosslingual LPR. The
  Korean drop is understated because of the filter. An earlier version of the 14-language results split Chinese and
  Japanese on whitespace and skipped almost all of them (commit dfce47e); the numbers here are rescored.

## 6. Mechanism of L22H6 (PR #12)

- Why: to find when the head acts and what it reads.
- What: Qwen2.5-1.5B-Instruct in fp32 with eager attention (transformers 4.57.6). 96 crosslingual LCB prompts, 24
  per language, all answered in the requested language at baseline. The 24 Italian prompts are ones whose reply had
  switched language under full removal of L22H6 in an earlier run; of the 72 French, German and Spanish prompts, 64
  had kept their language and 8 had switched. Also a pilot of 32 new prompts, 8 per language, that name the
  requested language and a second, unrelated one.
- How: L22H6 zeroed only while the prompt is read, only while the reply is generated, or both, with L22H8 as a
  control head; L22H6's attention weights to the requested language word; during generation, masking only L22H6's
  attention to that word, against masking a nearby token or the same word for L22H8; in the pilot, masking the
  requested or the unrelated language word. Pass or fail is LCB's line check on 100 new tokens.
- When / where: 2026-10-08; PR #12, results/qwen-l22h6-mechanism (not merged yet).
- Code: experiments/liha_l22h6_reproduce.py and tools/analyze_l22h6_results.py (PR #12).
- Result:
  - Replies passing, out of 96: clean 95, L22H8 removed 94, L22H6 removed only on the prompt 88, only during
    generation 69, both 64. Italian: 24, 24, 21, 0 and 0 of 24. French, German and Spanish: 21 to 24 of 24 under
    every condition.
  - At the last prompt position L22H6 puts 0.705 (German) to 0.867 (Spanish) of its attention on the requested
    language word, against 0.002 to 0.003 on a nearby token and 0.005 to 0.009 for the other heads of layer 22.
    During the reply the share is 0.33 to 0.48 early on and 0.18 to 0.26 later.
  - Of the 95 replies that pass at baseline, masking the edge to the requested word during generation fails 7 (4
    Italian, 2 German, 1 French), all among the 27 that fail when the whole head is removed during generation (24
    Italian). Masking a nearby token, or the word for L22H8, fails none.
  - Pilot, 31 replies that pass at baseline: masking the requested word fails 4 (all Italian, 4 of 8), masking the
    unrelated word fails none, removing the head during generation fails 9 (all 8 Italian). Mean attention is 0.79
    on the requested word and 0.01 on the unrelated one.
- Reading: on these prompts the head acts mostly while the reply is generated: removing it only during generation
  reproduces all 24 Italian switches, removing it only on the prompt 3. It attends strongly to the requested language
  word, and cutting just that access reproduces 7 of the 27 switches. The large effect is on Italian, whose prompts
  were selected among those that switch; on French, German and Spanish, where most prompts had not switched before,
  every condition passes 21 or more of 24. In the pilot the two masked words get very different attention (0.79 and
  0.01), so it does not separate what the word says from how much it is attended. A run on 96 monolingual prompts
  (no language name in the prompt) has been reported, but its files are not in PR #12 yet.

## 7. Base and instruct pairs across families

- Why: to check whether instruct models of other families have a head like L22H6, and whether their base models have
  the same head.
- What: the instruct and base models of Qwen2.5-3B, Qwen3-1.7B (thinking off), Gemma-3-1B and 4B, OLMo-2-1B,
  Llama-3.2-1B and 3B, OLMo-3-7B and SmolLM3-3B (thinking off), on the FLORES prompts. fp32, except OLMo-3-7B in
  bf16.
- How: every head of the instruct model is screened on 125 prompts. If the strongest head reaches c->w 0.1 on the
  screen, the layers holding the top two heads by c->w and the top head with dNLL <= 0.1 are rerun on all 2,500
  prompts, and the top three heads with dNLL <= 1 get mean ablation and scaling. Base models: Qwen2.5-3B, Qwen3-1.7B,
  Gemma-3-4B and OLMo-3-7B go through the same steps; Gemma-3-1B's base stayed under the threshold on the screen and
  its layers 5 and 11 were run on 2,500 prompts anyway; OLMo-2-1B's base has the full sweep of section 3. The Llama
  base models were not run, since neither instruct model reached the threshold. SmolLM3-3B instruct reached it only
  through a head that breaks generation (below); its run was stopped after the screen and its base was not run.
- When / where: 2026-10-08 to 10-09; results/<model>-screen, results/<model>, results/<model>-followup.
- Code: sweep.py, followup.py.
- Result (2,500 prompts unless marked):

| model | heads per layer | baseline non-English retention | head | c->w, zero | c->w, mean | dNLL (rest of layer) | other heads of the layer, max c->w | same head in base, zero (mean) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 12 | 0.902 | L22H6 | 0.500 | 0.450 | +0.228 (+0.004) | 0.007 | 0.150 |
| Qwen2.5-3B | 16 | 0.992 | L27H13 | 0.515 | 0.338 | +0.234 (+0.006) | 0.014 | 0.202 (0.114) |
| Qwen3-1.7B | 16 | 0.994 | L18H12 | 0.324 | 0.028 | +0.128 (+0.003) | 0.002 | 0.067 (0.022) |
| Gemma-3-1B | 4 | 0.997 | L11H3 | 0.413 | 0.089 | +0.720 (-0.012) | 0.002 | 0.001 |
| Gemma-3-4B | 8 | 0.997 | L24H0 | 0.214 | 0.012 | +0.202 (-0.005) | 0.000 | 0.148 (0.056) |
| OLMo-2-1B | 16 | 0.997 | L12H8 | 0.080 | 0.056 | +0.136 (+0.022) | 0.005 | 0.007 |
| OLMo-3-7B (bf16) | 32 | 0.960 | L14H25 | 0.134 | 0.007 | +0.202 (+0.000) | 0.010 | 0.104 on the screen |
| Llama-3.2-1B | 32 | 1.000 on the screen | none; strongest L8H25, 0.048 on the screen | | | | | not run |
| Llama-3.2-3B | 24 | 0.992 on the screen | none; strongest 0.008 on the screen | | | | | not run |
| SmolLM3-3B | 16 | 0.984 on the screen | none; L1H12 0.880 on the screen breaks generation, the rest 0.016 or less | | | | | not run |

- Other base-model heads: OLMo-2-1B base has its own head in the last layer, L15H5 (0.134, section 3). OLMo-3-7B base
  has L15H20 at 0.151 and L20H18 at 0.133, which drop to 0.036 and 0.028 under mean ablation. OLMo-2-1B instruct's
  L12H8 is at 0.136 on the screen and 0.080 on 2,500 prompts. With SmolLM3-3B's L1H12 removed, replies in every
  language, English included, turn into repeated `</think>` tokens (12 of 25 English prompts still pass on the
  screen), and dNLL is +2.58.
- Reading: in six of the ten instruct models one head is far above every other head of its layer (0.080 to 0.515,
  against at most 0.014). In each of the six the same head has a smaller effect in the base model: 0.001 and 0.007 in
  Gemma-3-1B and OLMo-2, 0.067 in Qwen3 (about a fifth of the instruct value), 0.150 and 0.202 in Qwen2.5 (30% and
  39%), 0.148 in Gemma-3-4B (69%). Mean ablation keeps 66% to 90% of the zero-ablation effect in Qwen2.5 and 70% in
  OLMo-2, but 6% to 22% in Gemma-3 and 9% in Qwen3; section 10 tests why. OLMo-3-7B's top heads lose most of their
  effect under mean ablation (0.134 to 0.007 in instruct, 0.151 and 0.133 to 0.036 and 0.028 in base). The two Llamas
  and SmolLM3 have no head that changes the language on FLORES without breaking generation. Effect sizes are not
  compared across models: heads per layer range from 4 to 32, and Gemma-3, OLMo-2 and Qwen3 normalize differently from
  Qwen2.5 and Llama, so each head is compared with the other heads of its own layer.

## 8. The heads on LCB, across models

- Why: to see what the heads from section 7 do on chat prompts, in both LCB tasks.
- What: five languages, as in section 5. Each instruct model's top FLORES head with zero and mean ablation, plus
  random control heads from the same layer (three per layer, one for the Llamas). The Llamas, which have no FLORES
  head, get the top two heads of their FLORES screen. A second head was also tested in Gemma-3-1B (L5H0), OLMo-2-1B
  (L0H10) and Llama-3.2-1B (L0H1). For Llama-3.2-1B every head (512) was also screened on 100 crosslingual prompts,
  25 per language, with the pass rate pooled over prompts.
- How: as in section 5. fp32, except OLMo-3-7B in bf16.
- When / where: 2026-10-08 to 10-09; results/<model>-lcb, results/llama3.2-1b-instruct-lcbscreen.
- Code: lcb.py (`--screen` for the head screen).
- Result (LPR at baseline and paired change with the head removed; Qwen2.5-1.5B in section 5):

| model | head | mono, baseline | mono, zero [95% CI] | mono, mean | cross, baseline | cross, zero [95% CI] | cross, mean | controls, mono / cross |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | L27H13 | 0.982 | -0.537 [-0.575, -0.503] | -0.409 | 0.888 | -0.449 [-0.478, -0.421] | -0.331 | +0.000..+0.003 / -0.025..+0.004 |
| Qwen3-1.7B | L18H12 | 0.985 | -0.149 [-0.174, -0.123] | -0.067 | 0.823 | -0.508 [-0.535, -0.478] | -0.230 | -0.004..-0.003 / +0.000..+0.021 |
| Gemma-3-1B | L11H3 | 0.984 | -0.628 [-0.662, -0.593] | -0.244 | 0.118 | -0.091 [-0.110, -0.073] | -0.092 | -0.005..+0.003 / -0.023..+0.038 |
| Gemma-3-4B | L24H0 | 0.990 | -0.259 [-0.292, -0.227] | -0.120 | 0.133 | -0.107 [-0.126, -0.090] | -0.053 | -0.001..+0.000 / +0.000..+0.012 |
| OLMo-2-1B | L12H8 | 0.986 | -0.272 [-0.302, -0.239] | -0.128 | 0.931 | -0.335 [-0.362, -0.308] | -0.143 | +0.000..+0.004 / -0.003..+0.004 |
| Llama-3.2-1B | L8H25 | 0.997 | -0.016 [-0.026, -0.008] | -0.015 | 0.874 | -0.711 [-0.737, -0.684] | -0.357 | -0.009..-0.004 / +0.001..+0.004 |
| Llama-3.2-3B | L0H2 | 0.993 | +0.001 [-0.006, +0.009] | +0.000 | 0.911 | -0.003 [-0.013, +0.008] | +0.000 | +0.003..+0.004 / -0.001..+0.001 |
| Llama-3.2-3B | L2H17 | 0.993 | +0.004 [-0.001, +0.010] | -0.004 | 0.911 | -0.006 [-0.014, +0.002] | -0.003 | same |
| OLMo-3-7B | L14H25 | 0.972 | -0.041 [-0.060, -0.024] | +0.001 | 0.874 | -0.018 [-0.033, -0.003] | +0.015 | -0.011..+0.004 / +0.001..+0.007 |

  - Second heads: Gemma-3-1B L5H0 +0.000 / +0.012, OLMo-2-1B L0H10 -0.021 / -0.054, Llama-3.2-1B L0H1 -0.004 / -0.027.
  - Share of crosslingual lines in English, baseline to zero ablation: Qwen2.5-3B 0.09 to 0.35, Qwen3 0.14 to 0.64,
    OLMo-2 0.05 to 0.28, Llama-3.2-1B 0.09 to 0.81, Gemma-3-1B 0.55 to 0.89, Gemma-3-4B 0.44 to 0.55.
  - Llama-3.2-1B screen: baseline 0.90; with L8H25 removed 0.19; the next two heads 0.80 (L13H4) and 0.83 (L6H31).
- Reading: removing the head lowers the two tasks by different amounts in different models: by similar amounts in
  Qwen2.5 (-0.537 and -0.449 at 3B, -0.273 and -0.276 at 1.5B) and OLMo-2 (-0.272 and -0.335), mostly crosslingual in
  Qwen3 (-0.149 and -0.508) and Llama-3.2-1B (-0.016 and -0.711), mostly monolingual in Gemma-3-1B (-0.628 and
  -0.091) and Gemma-3-4B (-0.259 and -0.107), whose crosslingual baselines are already low (0.118 and 0.133; 55% and
  44% English lines). The crosslingual lines that are lost mostly become English. Llama-3.2-1B's L8H25 barely matters
  on FLORES (0.048 on the screen) and on monolingual LCB, but it is the only one of the 512 heads whose removal takes
  the crosslingual screen below 0.80. On LCB, mean ablation keeps more of the zero-ablation drop than on FLORES for
  Gemma-3-1B (39% of the monolingual drop, against 22% on FLORES), Gemma-3-4B (46%, against 6%) and Qwen3 (45% of the
  crosslingual drop, against 9%). The two Llama-3.2-3B heads come from a FLORES screen where they
  tie with many others at one prompt; the screen of all its heads on crosslingual prompts is running. OLMo-3-7B's head
  changes LPR by less than 0.05, and under mean ablation LPR does not drop (+0.001, +0.015).

## 9. Smaller checks across models

### 9a. Content of the replies that switch

- Why: to check whether the replies that switch language without the head keep the prompt's content in the other
  models, as in Qwen2.5-1.5B (section 4).
- What: Qwen2.5-3B (L27H13), Qwen3-1.7B (L18H12), Gemma-3-1B (L11H3) and OLMo-2-1B (L12H8), on their 2,500-prompt
  FLORES generations; three random control heads from the same layer.
- How: cosine similarity between the prompt and its 40-token continuation with Qwen3-Embedding-0.6B, for the
  non-English prompts that stay in their language at baseline and switch to English with the head removed. FLORES
  sentence pairs give the scale: the next sentence of the same article 0.384, a random sentence 0.179.
- When / where: 2026-10-09; results/<model>-content.
- Code: content.py (`--same-layer-controls 3`).
- Result:

| model | prompts switching to English | similarity to the prompt, before | after | before vs after continuation |
|---|---|---|---|---|
| Qwen2.5-3B | 514 | 0.590 | 0.657 | 0.582 |
| Qwen3-1.7B | 739 | 0.635 | 0.683 | 0.600 |
| Gemma-3-1B | 873 | 0.639 | 0.559 | 0.586 |
| OLMo-2-1B | 101 | 0.702 | 0.668 | 0.607 |

  The control heads switch 0 to 2 prompts each.
- Reading: in all four models the English continuation is closer to the prompt than the next FLORES sentence is
  (0.559 to 0.683, against 0.384). It is closer than before the switch in the two Qwen models and less close in
  Gemma-3-1B and OLMo-2.

### 9b. OLMo-2 post-training checkpoints

- Why: OLMo-2-0425-1B releases a checkpoint after each post-training step, so L12H8 can be followed from base to
  instruct.
- What: the base model, the SFT and DPO checkpoints, and the released instruct model, which adds RLVR on math data
  after DPO (model card). FLORES layer 12 on 2,500 prompts, and LCB for L12H8 with three layer-12 control heads.
- How: as in sections 7 and 8. SFT, DPO and instruct use the same chat template and run in fp32. The base model runs
  without a template, in bf16 and with the end-of-text token blocked (section 3).
- When / where: 2026-10-08 (base) and 10-09; results/olmo2-1b, olmo2-1b-sft-instruct, olmo2-1b-dpo-instruct,
  olmo2-1b-instruct and their -lcb runs.
- Code: sweep.py, lcb.py.
- Result:

| checkpoint | FLORES non-English retention | L12H8 c->w | dNLL | rest of layer 12, max c->w | LCB mono: baseline / zero / mean | LCB cross: baseline / zero / mean |
|---|---|---|---|---|---|---|
| base | 0.921 | 0.007 | -0.000 | 0.149 (L12H0, dNLL +0.421) | | |
| SFT | 0.991 | 0.028 | +0.109 | 0.002 | 0.997 / -0.076 / -0.038 | 0.895 / -0.247 / -0.091 |
| DPO | 0.998 | 0.071 | +0.134 | 0.008 | 0.989 / -0.199 / -0.120 | 0.933 / -0.244 / -0.123 |
| instruct | 0.997 | 0.080 | +0.136 | 0.005 | 0.986 / -0.272 / -0.128 | 0.931 / -0.335 / -0.143 |

  The 95% CIs of the zero-ablation changes: monolingual [-0.095, -0.059], [-0.230, -0.172], [-0.302, -0.239];
  crosslingual [-0.273, -0.221], [-0.268, -0.219], [-0.362, -0.308]. Layer-12 controls: -0.008 to +0.005.
- Reading: L12H8's effect on FLORES goes from 0.007 in base to 0.028 after SFT, 0.071 after DPO and 0.080 in the
  released model. On LCB the crosslingual drop is the same after SFT and after DPO (overlapping CIs) and larger in the
  released model, while the monolingual drop grows at each step. This is one 1B model; the base run differs in
  precision and in blocking the end-of-text token; and the last step is RLVR on math data, so the comparison does
  not separate the effect of a single training step.

## 10. Why zero and mean ablation differ (running)

- Why: in section 7, mean ablation keeps most of the zero-ablation effect in Qwen2.5 and OLMo-2 and little of it in
  Gemma-3, Qwen3 and OLMo-3. This run tests four explanations for the gap: the zeroed input is out of distribution;
  the norm applied to the attention output (Gemma-3, OLMo-2) rescales the other heads when one is zeroed; the model
  relies on a constant part of the head's output, which mean ablation keeps; or the head carries information that
  differs with the prompt's language.
- What: L22H6 (Qwen2.5-1.5B), L27H13 (Qwen2.5-3B), L18H12 (Qwen3-1.7B), L11H3 (Gemma-3-1B), L24H0 (Gemma-3-4B) and
  L12H8 (OLMo-2-1B), each on the 2,500 FLORES prompts, in the precision of section 7.
- How:
  - Statistics of the head's contribution after the output projection over the user's text and the baseline
    continuation. The template tokens before the user's text are left out, since the head's output there is the
    same for every prompt. Reported: the mean norm and its rank in the layer, the share of the energy in the mean
    vector, the share of the remaining variance explained by the prompt's language, and the cosine between the
    follow-up's mean (all prompt tokens, template included) and the mean over continuation tokens.
  - Generation with the head replaced by: zero; the follow-up's mean; the continuation mean; the head minus the
    continuation mean; the mean of the prompt's own language; the English mean; the mean of another language
    (German for en/fr/es/it prompts, French for de prompts); a random vector with the same norm at each position;
    half its value; and, for Gemma-3 and OLMo-2, zero with the post-attention norm held at its clean value. Every
    condition is generated in per-language batches.
- When / where: started 2026-10-09 18:37 KST; out/<model>-diag, to be added to results/.
- Code: diagnose.py. Its parts were checked on small random models; the run above is its first on these models.
- Planned reading, set before the results: flips with the random vector point to an out-of-distribution input; an
  effect that disappears when the norm is held fixed points to the norm; an effect from subtracting the mean points
  to a constant signal; a difference between the follow-up's mean and the continuation mean shows that the result of
  mean ablation depends on which tokens the mean is taken over; replies that move to the swapped-in language mean the
  head carries language identity. Section 2 has a version of the language-mean test for GPT-2's L6H10, where every
  language mean acted like ablation.

## 11. Steering the heads on LCB (running)

- Why: in section 10, replacing the head's output with its mean output for another language moves FLORES replies into
  that language in some of the models. This run asks whether the same vectors set the reply language on chat
  prompts, and whether they raise the crosslingual pass rate of the models that often answer in English (Gemma-3-1B
  0.118 and Gemma-3-4B 0.133, section 8).
- What: the six heads of section 7 (Qwen2.5-1.5B L22H6, Qwen2.5-3B L27H13, Qwen3-1.7B L18H12, Gemma-3-1B L11H3,
  Gemma-3-4B L24H0, OLMo-2-1B L12H8) and the three same-layer control heads of each model's LCB run (sections 5 and
  8), on the same five-language LCB prompts: fr/de/es/it monolingual (800) and crosslingual (1,196), English
  monolingual (200).
- How:
  - Vectors: the head's per-language mean output from section 10 (2,500 FLORES prompts, user's text and baseline
    continuation). Nothing is computed on LCB.
  - steer replaces the head's output with the mean of the language the reply should be in, swap with the mean of
    another language (German for en/fr/es/it, French for de). add steer and add swap add that mean minus the mean
    over all languages instead, with the coefficient fixed at 1. All four run on the head and on the three controls.
    The head is changed at every position (template, prompt and reply), as in section 10 and in mean ablation.
  - Precision, batching and greedy decoding as in sections 5 and 8, so the baseline should reproduce those runs.
  - Reported: LPR and its paired change with a bootstrap 95% CI, the share of replies entirely in the swap language
    with a bootstrap CI, the share of English lines, repetition, and the share of replies that the 5-word line filter
    skips.
- When / where: queued 2026-10-09 evening; out/<model>-steer, to be added to results/.
- Code: steer.py. The replacement for each prompt inside a mixed-language batch was checked against single-prompt
  runs on a small random model.
- Planned reading, set before the results (experiments/steer_plan.md):
  - The head sets the reply language on LCB if its share of replies in the swap language has a CI above every
    control's share.
  - steer fixes crosslingual replies if the change in crosslingual LPR has a CI above zero and above every control's
    change, while the skipped share and repetition rise by no more than 0.05.
  - Replace and add are reported side by side, and all six models are reported.
  - If Gemma-3-1B's crosslingual change under steer does not have a CI above zero, a second run takes the vectors from
    the LCB monolingual baseline replies and applies them to the crosslingual prompts only, to tell a FLORES vs chat
    domain gap from a head that does not set the language in chat.
  - LPR checks only the language. Whether steered replies keep the content of the baseline replies is checked
    afterwards with embedding similarity.

## 12. Still running (2026-10-09, 19:30 KST)

- Section 10: four of the six models are done (out/<model>-diag), Gemma-3-4B and Qwen2.5-3B are running. The
  results go into section 10 when all six are in. Section 11 starts after it.
- Llama-3.2-3B: every head on crosslingual LCB prompts, as for Llama-3.2-1B.
- Qwen3-4B, instruct and base: a second Qwen3 size (queued after section 11).

## What the results support and what they do not

Supported so far:
- In six of the ten instruct models, removing one attention head takes non-English replies out of their language far
  more than removing any other head of the same layer (section 7: 0.080 to 0.515, against at most 0.014). In
  Llama-3.2-1B one head does this for crosslingual requests only (section 8).
- The replies that switch keep the prompt's content (sections 4 and 9a).
- In Qwen2.5-1.5B the effect holds on LCB chat prompts, under sampling, and with or without the default system prompt
  (sections 4 and 5); the heads of Qwen2.5-3B, Qwen3, Gemma-3-1B, Gemma-3-4B and OLMo-2 also lower LCB scores, and
  Llama-3.2-1B's L8H25 lowers crosslingual ones (section 8).
- In each of the six models the same head has a smaller effect in the base model (section 7), and in Qwen2.5-1.5B
  the dependence is large only for the instruct model with its own chat template (section 4).
- In Qwen2.5-1.5B, on crosslingual prompts, the head acts mostly while the reply is generated and attends to the
  requested language word (section 6, with the Italian prompts selected among ones that switch).

Not supported, or not tested:
- That instruction tuning creates the head: in Qwen2.5 and Gemma-3-4B the same head is already there in the base model
  with a smaller effect; in Gemma-3-1B and OLMo-2 it is absent from the base model, and these runs do not show what
  produces it (section 7).
- That every instruct model has such a head: Llama-3.2-3B, OLMo-3-7B and SmolLM3-3B do not (section 7).
- That the head encodes the language: the experiments show that the models depend on it, not what it represents.
  Section 10 tests whether its output differs by language in a way that matters.
- That the effect is the same under mean ablation: it holds in Qwen2.5 and OLMo-2 but not in Gemma-3, Qwen3 or
  OLMo-3 (section 7).
- A ranking of effect sizes across models: heads per layer and normalization differ (section 7).
- Which post-training step produces the dependence (section 9b).
- That access to the language word explains the head's effect: masking it reproduces 7 of 27 switches (section 6).
- The first-token broadcaster claims of the submitted version (section 1).

## Known limitations

- Ten instruct models from seven families, 1B to 7B parameters. FLORES covers five European languages; LCB covers
  the same four non-English ones for every model and 14 languages for Qwen2.5-1.5B only.
- Single-turn prompts. Instruct models use their default chat template, including Qwen2.5's English system prompt.
- Interventions act on one head at a time: zero, mean and scaling, plus phase-limited ablation and attention masking
  in section 6. Several heads at once only for GPT-2. The mean used for mean ablation includes the chat template
  tokens of instruct models; section 10 also uses means taken without them.
- FLORES labels come from langdetect on a 40-token continuation. LCB labels come from fastText per line and skip
  lines under five words, so Korean replies that turn into Chinese or Japanese are skipped and the Korean drop is
  understated (section 5).
- In section 7, head selection uses c->w on 125 prompts and does not look at dNLL, so a head that breaks generation
  can pass the threshold (SmolLM3-3B L1H12, dNLL +2.58). The follow-ups of section 7 skip heads with dNLL above 1.
- Mean and zero ablation disagree in Gemma-3, Qwen3 and OLMo-3; section 10 is running.
- Precision: OLMo-3-7B in bf16, where batched and single-prompt generations agree on only 10 of 20 prompts; OLMo-2-1B
  base in bf16 with the end-of-text token blocked. Section 6 runs Qwen2.5-1.5B in fp32 (transformers 4.57.6),
  sections 4 and 5 in fp16.
- Base models were not run for Llama-3.2 and SmolLM3-3B. OLMo-3-7B base has L14H25 only on the 125-prompt screen.
- Gemma-3-1B's crosslingual LCB baseline is 0.118, too low to read a change.
- Qwen2.5-Instruct's generation config adds a repetition penalty (1.1 for 1.5B, 1.05 for 3B) to every greedy run.
- The section 6 results come from one model and 96 prompts, and the Italian prompts were selected among ones that
  switch.
