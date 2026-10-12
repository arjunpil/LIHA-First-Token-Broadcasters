# Controls for the value patch

Written on 2026-10-12 before the runs; the commit time is the record. These controls were added after the results of
mechanism_value_patch.md were in; what is fixed here in advance is how they are read, not the choice to run them.
Follows experiments/mechanism_value_patch.md, whose plan and smoke-run note were committed in b3de0d3 at 11:35:13
UTC, seconds before those runs started; the KST times written inside that file are off and the commit time is the
record.

Three objections remain to the value patch. The other-language patch changes the head's output more than the
same-language patch (mean norm 2.0 against 0.4 in Qwen2.5-1.5B, 0.36 against 0.10 in Gemma-3-1B), so its extra
failures could come from the size of the change. Each prompt is tested with one other language (German, or French
for German prompts), so the replies that move into that language could reflect a bias toward German or French.
And over the first eight reply tokens Gemma-3-1B L11H2 attends to the name slightly more than L11H3 (0.079 against
0.064), while the control head was L11H1, chosen at the last prompt token.

## Interventions

All act in the decoding steps after the first generated token, as before, on the same prompts and with donors
computed the same way, run by run (sample 1: the 96 prompts of results/gemma-l11h3-mechanism/prompt_manifest.csv;
sample 2: the 384 prompts of results/valuepatch-<model>-s2/prompt_manifest.csv).

- value_patch_random1, 2, 3: each name value v_j is moved by its distance to the other-language donor in a random
  direction, v_j + |v_other - v_j| u, with one unit vector u per prompt (shared by its name positions, so that the
  moves add up as the other-language ones do) drawn from a normal distribution seeded by the run seed, the prompt id
  and the seed index. The change to the head's output is then at least as large as under value_patch_other at every
  step (triangle inequality); only the direction of the move differs. Three seeds.
- value_patch_other2: the donor is the mean for a second language, Spanish for German prompts, Italian for Spanish
  and French prompts, French for Italian prompts (each differs from the first donor language and from the prompt's).
- Gemma-3-1B only: the control head's name mask with L11H2 as the control head.

The clean replies must reproduce those of the earlier runs exactly; otherwise the runs are not used.

## Reading

Counts are over the baseline-passing replies of each model, samples 1 and 2 pooled; McNemar tests are exact and
two-sided, uncorrected. The earlier value_patch_same, value_patch_other and name-mask outcomes are taken from the
earlier runs.

- Size: if value_patch_other fails more replies than value_patch_random with p < 0.05 for each of the three
  seeds, the extra failures of the other-language patch are read as depending on the direction of the move, not on
  its size alone. Otherwise the paper says that the failure counts do not separate the language from the size of
  the change.
- Direction under the random patch: for each seed, the number of scored replies entirely in the first donor language
  is reported; the move into that language is read as specific to its values if value_patch_other puts more replies
  there than value_patch_random with p < 0.05 for each seed.
- Second language: value_patch_other2 puts more scored replies entirely in the second donor language than
  value_patch_same, with p < 0.05. If this fails in a model, the paper says that the move into the donor language was
  shown for German and French only in that model.
- L11H2: in Gemma-3-1B, the L11H3 name mask fails more replies than the L11H2 name mask, with p < 0.05. Otherwise the
  paper says that masking L11H2's attention to the name also makes replies fail.
- Reported in every case: pass-to-fail, fail-to-pass and newly unscorable replies, replies entirely in each donor
  language, and the mean output-change norm of each patch.
- If the runs take much longer than the smoke run predicts, the second language comes first, then the random patch,
  then the L11H2 mask.

## Smoke run (15:28 UTC, before the main runs)

Gemma-3-1B, 8 prompts outside both samples (seed 7), all new conditions and the identity patch; the runner printed
its pass counts, which are not used. Patching each name value with itself left every reply unchanged, and every
patch acted in decoding only. Mean change norm per step: value_patch_other 0.350, value_patch_other2 0.313,
value_patch_random1 to 3 0.365 to 0.367 (at least the other-language value in 7 of 8 prompts; replies diverge, so
the steps compared are not the same). About 2.5 s per prompt and condition, so the Gemma runs take about 1.5 h.
