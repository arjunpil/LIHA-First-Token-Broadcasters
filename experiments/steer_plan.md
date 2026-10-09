# Steering on LCB: plan and reading, set before the results

Written 2026-10-09, 19:20 KST, before the smoke test (20 prompts, Qwen2.5-1.5B) and before any full run.

## Question

The diagnosis (diagnose.py) found that replacing a head's output with its mean output for one language moves FLORES
continuations into that language (Gemma-3-1B L11H3: 0.997 of non-English prompts end in the swapped-in language;
Qwen2.5-1.5B L22H6: 0.182). Does the same vector set the reply language on chat prompts (LCB), and does it raise the
crosslingual pass rate of models that often answer in English?

## Design

- Models and heads: the six instruct models with a FLORES head (EXPERIMENTS.md section 7): Qwen2.5-1.5B L22H6,
  Qwen2.5-3B L27H13, Qwen3-1.7B L18H12, Gemma-3-1B L11H3, Gemma-3-4B L24H0, OLMo-2-1B L12H8. All six run whatever
  the earlier ones show.
- Vectors: the head's mean output (before the output projection) over the user's text and the baseline continuation
  of the 2,500 FLORES prompts, per language, from the diagnosis run. Nothing is computed on LCB.
- Data: LCB, fr/de/es/it monolingual (800) and crosslingual (1,196), English monolingual (200); the same prompts,
  precision and batching as the earlier LCB run of each model, so the baseline should match it.
- Conditions, each on every prompt:
  - steer: replace the head's output with the mean of the language the reply should be in.
  - swap: replace it with the mean of another language (de for en/fr/es/it, fr for de).
  - add steer / add swap: add (that language's mean minus the mean over all languages) instead of replacing,
    coefficient fixed at 1.
  - the same four on the three same-layer control heads of the earlier LCB run.
  - The head is changed at every position (template, prompt and reply), as in the diagnosis and in mean ablation.
- Metrics: LPR (benchmark definition, source average); paired change against the baseline with a bootstrap 95% CI;
  share of replies entirely in the swap language, with a bootstrap CI; English line share; repetition; share of
  replies skipped by the 5-word line filter.

## Reading, fixed now

- Swap (does the vector set the language): for each model, the target head's share of non-English replies in the swap
  language, monolingual and crosslingual, against the control heads' shares. Read as setting the language on LCB when
  the head's CI lies above every control's share.
- Fix (crosslingual): the change in crosslingual LPR under steer. Read as a fix when its CI lies above zero and above
  every control's change, and the skipped share and repetition rise by no more than 0.05 over the baseline.
- No harm (monolingual): the change in monolingual LPR under steer, read with its CI.
- Replace and add are reported side by side; neither is chosen after the results.
- No change of vectors, coefficient, positions or prompts after seeing results. Any later variant is reported as a
  separate run with its reason.
- If Gemma-3-1B's crosslingual change under steer has a CI that includes zero or lies below it, a second run uses
  vectors from the LCB monolingual baseline replies, applied to the crosslingual prompts only, to tell a FLORES vs chat
  domain gap from the head not setting the language in chat.
- LPR checks the language only. Whether steered replies keep the content of the baseline replies is checked after the
  run with embedding similarity, reported without a pass threshold.

## Added 2026-10-09, 20:28 KST, after seeing Gemma-3-1B's baseline and steer conditions

The steer condition of Gemma-3-1B had finished (crosslingual LPR up, the controls not yet run). Two quality checks are
added to every model, as checks rather than criteria (steer_quality.py):
- Content: cosine similarity (Qwen3-Embedding-0.6B) between each reply and the baseline reply to the same prompt,
  against the baseline reply to another prompt of the same task and language, overall and on the prompts whose
  pass/fail changed. This fixes the design of the content check listed above.
- Fluency: perplexity of the reply text alone under the unmodified model, median over the scored replies and over
  those entirely in the expected language, read against the baseline replies in the same language.
