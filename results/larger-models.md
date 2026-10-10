# Larger Qwen models (preliminary)

A smaller version of the pipeline on Qwen3-8B, Qwen3-14B and Qwen2.5-32B, plus screens of Qwen2.5-1.5B and 7B
and a same-setup rerun of Qwen3-1.7B for reference. Run on 4x RTX PRO 6000 (96GB) with the experiment code at
34ac74d (PR #13); `big/` only adds the larger model keys and multi-GPU loading, `experiments/` is unchanged.

## Setup

- Screen: zero-ablate every head of the Instruct model (corrected o_proj-input hook) on 500 FLORES prompts
  (5 languages x 100, instead of 2,500), greedy 40 tokens. Base model: the layers of the Instruct top 3 heads.
- Head selection for LCB: top 3 by correct->wrong among heads with ΔNLL <= 0.1 (`big/top_heads.py`).
- LCB: 15 languages, both tasks, greedy 100 tokens, zh/ja segmented as in 34ac74d. Zero and mean ablation of the
  three heads, plus three random same-layer controls per layer (zero).
- Precision: Qwen3 fp32, Qwen2.5-1.5B fp32, Qwen2.5-7B and 32B bf16.

## Results

| model | heads | strongest head on the screen: SR (ΔNLL) | LCB: head, Δ mono / Δ cross | controls, max abs Δ |
|---|---|---|---|---|
| Qwen3-1.7B (rerun) | 448 | L18H12 0.346 (+0.128) | L18H12 −0.293 / −0.592 | 0.009 |
| Qwen3-8B | 1,152 | L24H27 0.010 (+0.059) | L24H27 −0.008 / −0.190 [−0.202, −0.177] | 0.003 |
| Qwen3-14B | 1,600 | L0H14 0.014 (+0.009) | none of L0H11, L25H15, L14H9: abs Δ <= 0.002 | 0.003 |
| Qwen2.5-1.5B | 336 | L22H6 0.508 (+0.204) | not run | |
| Qwen2.5-7B | 784 | L0H25 0.632 (+0.143); L0H22 0.380 (−0.000); L19H1 0.174 (+0.024) | not run | |
| Qwen2.5-32B | 2,560 | L0H14 0.100 (+0.375) | L0H27 −0.078 / −0.071 | 0.004 |

- Qwen3-8B L24H27 (relative depth 0.67, vs 0.64 for L18H12 in 1.7B): mean ablation gives −0.015 / −0.173, and the
  share of English lines in crosslingual replies goes from 0.08 to 0.27.
- Qwen2.5-32B L0H27 also lowers English monolingual LPR (1.000 -> 0.968) and raises ΔNLL by 0.086, so it looks like
  general damage rather than language selection.
- Qwen2.5-1.5B with this pipeline gives L22H6 SR 0.508, vs 0.480 on 2,500 prompts in the paper.
- The Qwen3-1.7B LCB rerun gives the same summary as `results/qwen3-1.7b-instruct-lcb-all` (one per-language cell
  differs by 0.01; 39,426 of 39,438 replies are identical), so it is not included again here.
- Script switches (`experiments/script_switch.py` definition): Qwen3-8B L24H27 switches 8/100 monolingual and
  16/299 crosslingual Korean replies into Han/kana, so the skipped-reply issue is small here.
- `*-headcmp-*`: attention to the language name in crosslingual prompts and probes on the head output
  (`big/head_compare.py`); `head.npz` (100 MB) is left out.

## Files

- `results/<model>-instruct-500/`, `results/<model>-500/`: screen (Instruct, Base)
- `results/<model>-instruct-lcb-all/`: LCB
- `results/qwen3-1.7b-instruct-L18-500/`: layer 18 of Qwen3-1.7B on the same 500 prompts

## Run

```
GPUS=0,1,2,3 MODELS="qwen3-8b qwen3-14b qwen2.5-32b" bash big/run_all_4gpu.sh
python big/head_compare.py --model qwen3-8b-instruct --head L24H27 --lcb-samples out/qwen3-8b-instruct-lcb
```

Single greedy pass and a smaller screen, so these numbers are preliminary.
