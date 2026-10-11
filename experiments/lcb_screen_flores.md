# LCB head screen on three models with a FLORES head

Written 2026-10-11, 13:19 KST, before the runs; the tie margin revised at 13:27 KST, still before them. On 10-11
Christian asked to treat LCB as a test of its own rather than a later stage of FLORES. So far, in the models with a
FLORES head, LCB was run only on that head and three same-layer controls (section 8); only the models without one got
a screen of every head.

Runner: lcb.py `--screen`, as for Llama-3.2-1B and 3B, Qwen3-4B and SmolLM3-3B (section 8): every head zero-ablated in
turn on the same 100 crosslingual LCB prompts (25 each for fr, de, es, it; seed 0), greedy, 100 new tokens, pooled
pass rate over the scorable replies. Each screen uses the precision, batch size and token budget of the model's
existing LCB run (results/<model>-lcb): Qwen2.5-1.5B fp16, 125 and 24,000; Qwen3-1.7B fp32, 250 and 4,000;
OLMo-2-1B fp32, 250 and 8,000.

Models: Qwen2.5-1.5B (fp16, 336 heads, FLORES head L22H6), Qwen3-1.7B (fp32, thinking off, 448 heads, L18H12) and
OLMo-2-1B (fp32, 256 heads, L12H8). Their crosslingual LPR on the full LCB run is 0.704, 0.823 and 0.931 at baseline.
Left out: Gemma-3-1B and 4B, whose crosslingual baselines (0.118 and 0.133) leave little room for a drop, and
Qwen2.5-3B and 7B (1,360 heads), for time.

Reading, for each model:
- Heads that fail the head rule's checks on the 125-prompt FLORES screen (dNLL above 1, or fewer than 0.9 of the
  English continuations kept in English; head_rule.md) are listed but not ranked, as SmolLM3-3B's L1H12 was. The
  screens are results/qwen-instruct, results/qwen3-1.7b-instruct-screen and results/olmo2-1b-instruct-screen.
- Reported: the screen's baseline, the FLORES head's drop and rank, and the strongest head and its drop.
- The two tests agree on a model if the FLORES head has the largest drop. A head whose drop is within 0.025 of the
  largest (at most two of the roughly 100 scorable prompts) counts as tied with it, and ties are reported as ties,
  not ranked.
- If another head has a larger drop by more than 0.025 and that drop is at least 0.1, it gets the full five-language
  LCB run with three same-layer controls, as the strongest heads of the Qwen3-4B and Llama-3.2-3B screens did. The
  crosslingual changes of the two heads on the full runs are then compared through their 95% CIs; overlapping CIs
  are reported as not separated.
- If no head drops the pass rate by 0.1, the screen is reported as not singling out a head.

## The other models with a FLORES head, added 2026-10-11, 13:40 KST, before any of the runs

Christian then asked to run the screen on every model the GPU allows, with results that arrive after the submission
going into the rebuttal or the final version. The same screen and reading apply to these, after the three above:
- Qwen2.5-3B (fp32, 576 heads, L27H13), Qwen2.5-7B (bf16, 784 heads, L19H1) and OLMo-3-7B (bf16, 1,024 heads,
  L14H25), on the crosslingual prompts. OLMo-3-7B's head is not counted in the paper (its LCB changes are below 0.05),
  so its screen asks whether LCB finds another head there.
- Gemma-3-1B (fp32, 104 heads, L11H3) and Gemma-3-4B (fp32, 272 heads, L24H0) on 100 monolingual prompts instead (25
  each for fr, de, es, it; seed 0), since their crosslingual baselines are too low; their monolingual LPR is 0.984
  and 0.990. This screen differs from the others in task, so it is reported separately.
- Precision, batch size and token budget again come from the model's existing LCB run: Qwen2.5-3B fp32, 250 and
  8,000; Qwen2.5-7B bf16, 125 and 4,000; OLMo-3-7B bf16, 125 and 4,000; Gemma-3-1B fp32, 250 and 6,000; Gemma-3-4B
  fp32, 64 and 2,000. Their FLORES screens are results/<model>-screen.
- A full LCB run for a head picked by any of the screens uses the same settings, so that its baseline replies match
  the existing run's.
