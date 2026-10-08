# Head ablation

The original LIHA scripts ablate a head by zeroing `h*head_dim:(h+1)*head_dim` of the attention output after the
output projection (`attn` output in GPT-2, `o_proj` output in Qwen, `self_attention` output in BLOOM). After the
projection every head writes to all hidden dimensions, so that slice is a block of residual dimensions rather than one
head, and in BLOOM the `self_attention` output already includes the residual connection. These scripts zero the head's
slice at the input of the output projection instead (`c_proj`, `o_proj`, `dense`), and also run the original hook
(`paper` mode) on the same prompts for comparison.

## Run
From the repo root:
```
pip install torch transformers datasets langdetect matplotlib
pip install langid fasttext-wheel        # only for prompts.py --set extended
python prompts/prompts.py                               # 2,500 European prompts, same rules as expand_dataset.py
python experiments/sweep.py --model gpt2 --bs 500       # head and paper modes, all 144 heads
python experiments/detect.py gpt2
python experiments/analyze.py gpt2                      # out/gpt2/summary.md
python experiments/sweep.py --model qwen-instruct --per-lang 25 --bs 125
python experiments/sweep.py --model qwen-base --per-lang 25 --bs 125

python prompts/prompts.py --set extended                # zh/ru, 100 each, prompts_extended.csv
python experiments/sweep.py --model gpt2 --prompts prompts/prompts_extended.csv --per-lang 100 --bs 200 --out out/gpt2-zhru
python experiments/multi.py run                         # fig 1b: cumulative ablation by SR, c->w and 3 random orders
python experiments/multi.py run --orders c2w-lowloss --out out/gpt2-multi-lowloss/gens.jsonl
python experiments/multi.py report                      # after detect.py gpt2-multi
python experiments/figures.py                           # fig 1a / 1b from results/
python tables/tables.py                                 # tables/README.md, the paper tables from results/
python experiments/amplify.py run                       # scale single heads by 2/3/5 at the c_proj input
python experiments/checks.py run gpt2-sampling          # also gpt2-truncated, qwen-format; then detect.py and checks.py report
python experiments/identity.py run                      # L6H10 output replaced or shifted by language means
python experiments/attention.py                         # L6H10 attention / entropy figures, table 6, probing
python experiments/sweep.py --model olmo2-1b --modes head --bs 250  # also gpt2-medium; OLMo runs with end-of-text blocked
python experiments/followup.py run --model olmo2-1b     # mean ablation and scaling for the top c->w heads
python experiments/quality.py gpt2,out/gpt2/gens.jsonl,results/gpt2/labels.json,L6H10  # repetition / prompt copy / other
python experiments/content.py                           # prompt vs continuation similarity, results/gpt2-content
python experiments/robustness.py --lid lid.176.bin      # other detectors and split halves on out/gpt2
python experiments/checks.py run qwen-system --per-lang 500 --out out/qwen-system-2500  # default / no / translated system prompt
python experiments/lcb.py --model qwen-instruct --heads L22H6,L17H7,L17H8 --scale L22H6:2,L17H7:2,L17H8:3
python experiments/lcb.py --model qwen-instruct --heads L22H6 --report-only   # rescore saved LCB replies
```
The newer models are screened first: every head on 125 prompts (`--per-lang 25 --out out/<model>-screen`), and
only if the strongest head flips at least 10% of the correct prompts are the layers of the top heads rerun on all
2,500 (`--layers`), followed by `followup.py`. `lcb.py` runs the Language Confusion Benchmark (Marchisio et al.,
2024) in fr/de/es/it/en with the chat template; it downloads the test sets and fastText's lid.176.bin on first use
and scores with the benchmark's line-level pass rate.

`--per-lang 25` matches the 125 prompts of qwen_experiment.py. `--layers` limits the sweep, e.g. `--layers 0,3,6,9,12,15,18,21`
for the BLOOM layers sampled in bloom_experiment.py. `first_token_attn.py` computes each GPT-2 head's attention to the
first token for the sink comparison.

Generation is greedy with 40 new tokens and batched with left padding, checked against single-prompt generation.
Each condition also records the LM loss on 100 FLORES dev sentences per language. `analyze.py` reports the switch
rate with a bootstrap CI, switches split into correct to wrong and wrong to correct, accuracy, and the LM loss change
next to the mean of the other heads in the same layer.

Results so far are in results/RESULTS.md.

BLOOM runs in fp32: in fp16, left padding gives NaN logits on some rows. Its paper mode keeps the 64-wide slice
from bloom_experiment.py, which assumed hidden 1024; the model's head dim is 128.

`results/<model>/` holds the outputs behind RESULTS.md: `summary.json` (per head and mode: switch rate with CI,
correct to wrong, wrong to correct, accuracy on the full set and on the 25 hand-written prompts, LM loss change) and
`labels.json` (the detected language of every generation, in the order of `prompts_european.csv`; Qwen uses its first
25 prompts per language). Modes are `head` for the fixed hook and `paper` for the original one. `gpt2-zhru` follows
`prompts_extended.csv`, and `gpt2-multi*` has one label list per order and step (`sr:k3` etc.).
