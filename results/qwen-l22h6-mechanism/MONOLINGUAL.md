# Monolingual LCB phase-ablation follow-up: L22H6

## Purpose

Test whether decoding-stage ablation of Qwen2.5-1.5B-Instruct attention head L22H6 changes response-language consistency when the user prompt is itself written in the expected language, **without an explicit request naming that response language**. This complements the outcome-enriched crosslingual LCB phase experiment in this directory; the two samples must not be treated as identically selected.

## Frozen sample and protocol

- **Prompts:** 96 saved LCB monolingual baseline prompts, with 24 each in Italian (it), French (fr), German (de), and Spanish (es). Drawn using seed `20261008` from the baseline rows of `results/qwen-instruct-lcb/samples.jsonl.gz`, with approximately equal representation of available sources. No filter for baseline correctness or prior head-ablation outcome was applied **by the sampling script**. All 96 selected prompts were marked baseline-correct and scorable in the saved upstream data; all 96 fresh clean responses also passed the run's language check. The manifest records the exact prompts, IDs, original source, and original baseline scoring fields.
- **Source composition:** Italian: 24 Okapi; German: 24 Okapi; French and Spanish: 8 Dolly + 8 Native + 8 Okapi each. Source and language effects are therefore not fully separable.
- **Prompt-language check:** A regex check found zero mentions of the strings “Italian,” “French,” “German,” or “Spanish” in these 96 prompts. This is a check for those explicit English language names, **not** a proof that no prompt contains any other language cue.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, float32/eager attention on NVIDIA GPU; 100 new tokens, greedy generation, cached decoding. A head is removed by zeroing its slice of the input to layer 22's attention output projection (`o_proj`) with a forward pre-hook.
- **Conditions:** clean; L22H6 prefill-only; L22H6 cached-decode-only; L22H6 full (prefill plus decode); and L22H8 full-head control. Every condition uses the same 96 prompts. A decode-only intervention begins *after* the first generated token, and the first token was checked to match clean for all 96 prompts.
- **Scoring:** Line-level fastText `lid.176.bin` classifier; lines with fewer than five whitespace-separated words excluded, line prediction threshold 0.3, and a response passes if all scorable lines match the prompt language. No response in this run was unscorable. This is a language-consistency check, not a measure of content quality. It does not include the Chinese/Japanese-specific tokenization correction used for the larger 14-language benchmark, which is irrelevant to these four languages.

## Results (passes out of 24; same prompts across conditions)

| Language | Clean | L22H8 control | L22H6 prefill only | L22H6 decode only | L22H6 full |
|---|---:|---:|---:|---:|---:|
| Italian | 24 | 24 | 23 | 0 | 0 |
| French | 24 | 24 | 23 | 21 | 18 |
| German | 24 | 24 | 24 | 23 | 23 |
| Spanish | 24 | 24 | 24 | 21 | 20 |
| **Pooled** | **96** | **96** | **94** | **65** | **61** |

Among matched clean-correct, fully scorable prompts, a two-sided exact paired binomial comparison of prefill-only versus decode-only yields p = 2.3842e-7 in Italian (23 discordant in the prefill-pass/decode-fail direction; zero in the reverse), and p = 2.9802e-8 pooled (30 vs 1). These are **exploratory statistics** on a fixed, modest-sized sample; Italian accounts for most of the pooled gap. The German, French, and Spanish differences are much smaller.

Manual review of the Italian decode-only responses found predominantly shifts into other Romance languages or mixed-language text, plus one visibly degenerate repetition. Thus the 24/24 Italian *scoring failures* should not all be called fluent language switches.

## Interpretation and limitations

The phase intervention suggests that L22H6 contributes to maintaining Italian response language during autoregressive generation **even without an explicit named output-language instruction**. This does not establish what prompt information the head accesses in monolingual examples, or that the effect generalizes equally across languages or data sources. It also does not imply that the head encodes language on its own or that the single-head output is sufficient to steer language.

The original crosslingual follow-up selected all 24 Italian prompts from earlier full-head language-switch cases, whereas the monolingual sampling script did not condition on prior ablation outcome. **Do not use the two 24-prompt results to estimate an unbiased cross-task effect size.**

This run was in a single-prompt FP32/eager environment (PyTorch `2.11.0+cu130`, Transformers `5.18.0`) rather than the upstream FP16/batched benchmark. Original model commit was not captured (`model_commit: null`), and the original upstream sample came from an unpinned main-branch file. The exact sampled prompts have therefore been frozen in the manifest. The stored metadata records the repository-relative source path rather than a user-specific GitHub URL; this does not retroactively pin the original source. These are exploratory within-run results rather than an exact, independent replication of the full upstream benchmark.

## Files and reproducibility

- `monolingual_prompt_manifest.csv`: the 96 frozen prompts and original baseline scoring fields.
- `monolingual_phase_outputs.csv`: all 480 generation records, responses, line-level scoring, and intervention call counts.
- `monolingual_phase_summary.csv`: aggregate conditions by language.
- `monolingual_experiment_metadata.json`: recorded run configuration, versions, and provenance.
- `experiments/liha_l22h6_monolingual_colab.py`: GPU/Colab-oriented reproduction runner that reads the **frozen 96-prompt manifest** from this directory and runs all five conditions. It saves the per-prompt records and a results ZIP; inside Colab it also offers a download. No user-identifying repository URL is required.
- `tools/check_l22h6_monolingual.py`: standard-library-only check that recalculates counts and exact paired comparisons from the committed CSVs, and tests record integrity, pairing, source distribution and hook counts. Run `python tools/check_l22h6_monolingual.py` from the repository root.

To rerun from a GPU machine with a checkout of the repository, install a suitable CUDA-enabled PyTorch build separately, run `pip install -r requirements-l22h6-monolingual.txt`, then run `python experiments/liha_l22h6_monolingual_colab.py` from the repository root. This monolingual run used Transformers 5.18.0, whereas the separate `requirements-l22h6.txt` pins Transformers 4.57.6 for the earlier crosslingual analysis; do not assume the two environments are interchangeable. The script reads the committed manifest and writes to `out/qwen-l22h6-monolingual/` by default. In Colab, work from the repository checkout or set `LIHA_REPO_ROOT` to its location. Dependencies include PyTorch/CUDA, Transformers, NumPy, pandas, SciPy, and fastText (the script installs `fasttext-community` if needed). The fastText language model and the Hugging Face model weights require a network download if not cached. The original environment is recorded above; exact bytewise reproduction is not guaranteed without the original model revision and full environment lockfile.
