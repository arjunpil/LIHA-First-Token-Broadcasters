# Head ablation rerun

The scripts in the repo root ablate a head by zeroing `h*head_dim:(h+1)*head_dim` of the attention output after the
output projection (`attn` output in GPT-2, `o_proj` output in Qwen, `self_attention` output in BLOOM). After the
projection every head writes to all hidden dimensions, so that slice is a block of residual dimensions rather than one
head, and in BLOOM the `self_attention` output already includes the residual connection. These scripts zero the head's
slice at the input of the output projection instead (`c_proj`, `o_proj`, `dense`), and also run the original hook
(`paper` mode) on the same prompts for comparison.

## Run
```
pip install torch transformers datasets langdetect
python prompts.py                      # 2,500 European prompts, same rules as expand_dataset.py
python sweep.py --model gpt2 --bs 500  # head and paper modes, all 144 heads
python detect.py gpt2
python analyze.py gpt2                 # out/gpt2/summary.md
python sweep.py --model qwen-instruct --per-lang 25 --bs 125
python sweep.py --model qwen-base --per-lang 25 --bs 125
```
`--per-lang 25` matches the 125 prompts of qwen_experiment.py. `--layers` limits the sweep, e.g. `--layers 0,3,6,9,12,15,18,21`
for the BLOOM layers sampled in bloom_experiment.py. `first_token_attn.py` computes each GPT-2 head's attention to the
first token for the sink comparison.

Generation is greedy with 40 new tokens and batched with left padding, checked against single-prompt generation.
Each condition also records the LM loss on 100 FLORES dev sentences per language. `analyze.py` reports the switch
rate with a bootstrap CI, switches split into correct to wrong and wrong to correct, accuracy, and the LM loss change
next to the mean of the other heads in the same layer.

Results so far are in RESULTS.md.
