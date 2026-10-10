# LCB in 14 languages for more models: selection and reading, set before the runs

Written 2026-10-09, 22:00 KST, before any of these runs.

## Question

In Qwen2.5-1.5B, removing L22H6 changes Chinese, Japanese and Russian LPR by 0.03 or less, while Hindi, Korean and
Arabic drop (EXPERIMENTS.md section 5). Is this pattern specific to Qwen2.5-1.5B or shared across model families?

## Selection

The smallest instruct model of every family in which a head was found (sections 7 and 8): Qwen3-1.7B (L18H12),
Gemma-3-1B (L11H3), OLMo-2-1B (L12H8) and Llama-3.2-1B (L8H25, a crosslingual-only head). Qwen2.5 is covered by
Qwen2.5-1.5B. One model per family, because the question is whether the pattern holds across families; the smallest
one, because it matches Qwen2.5-1.5B's size and keeps each run near an hour. The larger models of the same families
(Qwen2.5-3B, Gemma-3-4B) would test size rather than family and are left out for time. No 14-language result of
these models had been seen when this rule was set.

## Design

As in the Qwen2.5-1.5B run: the 14 non-English LCB languages and English, monolingual and crosslingual; zero and mean
ablation of the head; three random same-layer control heads (seed 0); each model's precision and batch settings of
its five-language run; zh/ja segmented as in the benchmark; WPR for ar, hi, ja, ko, ru and zh.

## Reading, fixed now

- Per model and language: the head's paired change in LPR under zero ablation with a bootstrap 95% CI, against the
  three controls' changes, monolingual and crosslingual separately.
- A language counts as affected if the head's CI lies below zero and its change is below every control's change.
- The Qwen2.5-1.5B pattern is read as shared by a model if, on monolingual prompts, Hindi is affected and Chinese,
  Japanese and Russian are not.
- All four models are reported whatever the outcome. Llama-3.2-1B's head acts on crosslingual prompts only, so its
  monolingual results are expected to be flat and are reported as they come.

## Note added 2026-10-10, 09:01 KST

The selection above says the smallest models keep each run near an hour; the first one, Qwen3-1.7B, took 1 h 19 min
for three of its six conditions. The selection and the reading are unchanged.

## Qwen2.5-3B, added 2026-10-10, 16:48 KST, before its run

With all four families run, only Qwen2.5-1.5B leaves Chinese, Japanese and Russian alone. Whether that belongs to the
Qwen2.5 family or to the 1.5B model is open, so Qwen2.5-3B (L27H13) gets the same run: the 14 non-English languages
and English, zero and mean ablation, three random controls from layer 27 (seed 0), its settings from section 8 (fp32,
batch 250, token budget 8000). The reading is the one above, unchanged. If 3B shares the pattern, it is read as a
Qwen2.5 trait; if not, as a property of the 1.5B model. Both outcomes are reported. This is the last run added before
submission.
