# Value patching and a larger crosslingual sample for the mechanism checks

Written 2026-10-12, 20:45 KST, before the runs. Follows experiments/mechanism_checks.md.

Two questions are left open by the mechanism checks. Masking the head's attention to the language name removes
far more attention than either control (over the first eight reply tokens, L22H6 0.387 against 0.021 for L22H7 and
0.001 for the nearby tokens), so its failures could come from the amount of attention removed rather than from what
the name carries. And Gemma-3-1B passes on only 12 of its 95 scorable replies, so its result rests on 12 replies.

## Interventions

Both act in the decoding steps after the first generated token, in the selected head only, as the earlier masks do.

- value_patch_same: the value vectors at the language-name positions are replaced by the mean value at the name
  positions of the other prompts of the same run that name the same language (leave one out). The attention weights
  are left as they are: the patch adds sum over name positions j of a_tj (v_donor - v_j) to the head's attention
  output before o_proj. The shared key-value cache and the other heads of the layer are not touched (the layer has
  2 key-value heads for 12 query heads in Qwen2.5-1.5B, 1 for 4 in Gemma-3-1B).
- value_patch_other: the same with the mean for another language, German for French, Spanish and Italian prompts
  and French for German prompts (the swap of section 6.2).
- The donor means are the mean, over the run's prompts that name a language, of the value at that prompt's name
  positions, in a clean prefill. The norm of the change to the head's output is recorded at every step.
- A smoke run also patches each name position with its own value, which must leave every reply unchanged.

## Samples

1. The 96 prompts of the earlier checks (results/gemma-l11h3-mechanism/prompt_manifest.csv), both models: clean,
   value_patch_same and value_patch_other. The clean replies are compared with the earlier runs.
2. 384 new crosslingual prompts, 96 per language, drawn by the same rule as the 96 (English language name in the
   prompt, round robin over LCB sources, no model output used), seed 20261012, leaving out the 96. Both models, all
   seven conditions: clean, the four earlier interventions (name mask, nearby mask, control head's name mask with
   L22H7 and L11H1 as before, head zeroed in decoding) and the two patches.

## Reading

Counts are over the baseline-passing replies of each model unless said otherwise; McNemar tests are exact and
two-sided, uncorrected.

- Content beyond amount, on samples 1 and 2 pooled: value_patch_other fails more replies than value_patch_same,
  with p < 0.05.
- Direction, on samples 1 and 2 pooled: value_patch_other puts more scorable replies (all lines in the donor
  language) into the donor language than value_patch_same does, with p < 0.05 on all scorable prompts.
- If value_patch_same fails more replies than the nearby mask with p < 0.05, the patch is read as a general
  perturbation and the two readings above are not made.
- Replication, on sample 2 alone: the name mask fails more replies than the nearby mask and than the control
  head's mask, each with p < 0.05, the reading of mechanism_checks.md.
- Otherwise a reading is reported as not shown at this sample size.
- Reported in every case: baseline-passing n of scorable n, pass-to-fail, fail-to-pass and newly unscorable
  replies, by language, and the mean output-change norm of the two patches.

## Added after the smoke run, before the main runs (2026-10-12, 21:10 KST)

The smoke run (8 prompts outside both samples; the runner printed its pass counts, which are not used) passed the
identity check in both models, and the patches acted at every decoding step and never in the prefill. It also showed that the
other-language patch changes the head's output more than the same-language patch (mean norm per step 0.33 against
0.14 in Gemma-3-1B, 2.75 against 0.61 in Qwen2.5-1.5B). A difference in failures between the two patches could
therefore come from the size of the change as well as from the language it carries. The content reading above is
made only together with the direction reading; on its own, more failures under value_patch_other are reported as
not separating content from size.
