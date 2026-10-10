# Two checks on the mechanism runs

Written 2026-10-10, 21:10 KST, before the runs. They follow up on PR #12 (EXPERIMENTS.md section 6).

Runner: experiments/gemma_l11h3_attention.py, now with --model, --layer, --head, --control-head and --lid; its
defaults reproduce the Gemma-3-1B run. Both models get the prompts of results/gemma-l11h3-mechanism/prompt_manifest.csv:
96 crosslingual LCB prompts that name the requested language, 24 per language, drawn without regard to any model's
replies. fp32, eager attention, greedy decoding, 100 new tokens; every intervention acts only after the first
generated token.

1. Gemma-3-1B, L11H3, with a control head that attends to the language name. The control head is the other head of
   layer 11 with the largest mean attention from the last prompt token to the name in PR #12's clean run: L11H1
   (0.096; L11H2 0.066, L11H0 0.025). All five conditions are rerun; the clean, name-mask, nearby-mask and zeroing
   conditions are compared with PR #12's outputs.
2. Qwen2.5-1.5B, L22H6, on the same prompts. A clean pass first records the attention of all 12 heads of layer 22.
   The control head is the other head with the largest mean attention from the last prompt token to the name, picked
   by this rule before the interventions run. Then all five conditions.

Reading, for each model, on the replies that pass the line check without intervention:
- Count the replies that fail with the head's attention to the name masked, with its attention to as many nearby
  tokens masked, and with the control head's attention to the name masked. The head's mask is compared with each of
  the other two by an exact two-sided McNemar test on the paired outcomes.
- The edge to the name is read as specific to the head if its mask fails more of these replies than the control
  head's mask, with p < 0.05.
- For Qwen2.5-1.5B, PR #12's result (7 of 95 replies, on prompts whose Italian ones were chosen among switches) is read
  as holding on prompts drawn without selection if the head's mask fails more replies than both the nearby mask and
  the control head's mask, each with p < 0.05.
- Otherwise the result is reported as not shown at this sample size. Gemma-3-1B had 12 passing replies in PR #12, so
  a real difference may not reach p < 0.05. Both outcomes are reported, and zeroing the head during generation is
  reported alongside, as in PR #12.
