# Gemma-3-1B L11H3: requested-language token attention

Exploratory input-side analysis of **L11H3** in `google/gemma-3-1b-it`, using 96 crosslingual Language Confusion Benchmark (LCB) prompts with an explicit target-language name (24 each in German, Spanish, French and Italian). This check is complementary to the separately reported output-replacement and language-steering results; it does **not** test steering itself.

## Reproduction records

- Runner: `experiments/gemma_l11h3_attention.py`, the standalone Python implementation used in the completed Colab experiment. The notebook itself is not part of this package.
- Frozen sample: `prompt_manifest.csv` (96 rows, including prompt IDs, language and source). Sampling used seed `20261009` and selected prompts containing the English name of the requested language, without filtering by Gemma's success or failure. Source pools and original LCB archive SHA-256 are recorded in `run_metadata.json`.
- Model: `google/gemma-3-1b-it`; resolved commit `dcc83ea841ab6100d6b47a070329e1ba4cf78752`; float32, eager attention; PyTorch `2.11.0+cu130`, Transformers `5.18.0`. 100 greedy generation tokens with cached decoding. Layer 11, query head 3; L11H0 used as control.
- Five paired conditions: clean, mask L11H3's attention to the requested-language token, mask its attention to equally many nearby tokens, mask L11H0's attention to the requested-language token, and zero L11H3 during decoding. All interventions begin after the first generated token.
- Attention is measured on the clean continuation by teacher forcing the generated token IDs. The last-prompt-position statistic is distinct from the later generation positions in `attention_by_prompt.csv`.
- Language scoring: fastText line-level language identification, skipping lines with fewer than five whitespace-separated words and requiring every scored line to match the target. Report failures on **scorable** responses; this is not a semantic-content metric.

## Observations

| Condition | Passes / scorable |
|---|---:|
| Clean | 12/95 |
| Target-language token mask | 5/95 |
| Nearby-token mask | 12/95 |
| Control-head target mask | 12/95 |
| L11H3 decode-only zero | 4/95 |

Seven originally passing responses become failures under the target-token mask, and eight under decode-only zeroing, with no gains in either condition. Neither negative control changes passing status on any of the 96 matched prompts. One German prompt is unscorable across the conditions.

At the final prompt position L11H3 assigns mean attention mass **0.13557** to the language-name tokens, versus **0.00858** to the nearby tokens, over the 96 prompts. The nearby-token control is **not attention-matched**. Other heads can also attend to the language token.

## Interpretation and limitations

The result supports a causal contribution of this *attention edge* to language retention on a small subset of responses that Gemma initially answered in the target language. It does not establish that the head semantically represents the requested language, or that this edge explains the stronger output-steering effect. Only 12/95 clean responses pass: ceiling/floor effects and uncertain generalization are substantial. The four languages and explicit-language-name prompt format are a limited scope; the sample has not been shown to reflect overall LCB accuracy or other languages. One prompt is unscorable.

The saved metadata include the exact resolved model commit and original LCB archive checksum, but **this extracted standalone runner has not been independently rerun in a fresh GPU environment**. The fastText model is downloaded from the official public endpoint; its exact file digest was not recorded. The original sampling URL includes `/HEAD/`, which can change; reproducibility should use the frozen manifest and the resolved model commit below.

## Check the results (no GPU required)

From the repository root:

```bash
python tools/check_gemma_l11h3_results.py
```

For a GPU reproduction (with CUDA PyTorch installed, dependencies from `requirements-gemma-l11h3.txt`, and access to the Gemma weights):

```bash
python experiments/gemma_l11h3_attention.py \
  --manifest results/gemma-l11h3-mechanism/prompt_manifest.csv \
  --model-revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 \
  --per-language 24 --max-new-tokens 100 \
  --out gemma_l11h3_reproduction
```

The recorded original run did not explicitly request a revision, but resolved to the commit above. The manifest-based rerun avoids reselection from a mutable benchmark archive. Compare the new per-prompt outputs, not merely aggregate counts.

## File index

- `condition_outputs.csv`: generated replies and interventions (480 records)
- `paired_outcomes.csv`: prompt-matched pass/fail comparisons (384 records)
- `attention_by_prompt.csv`: clean attention measurements (96 records)
- `attention_summary.csv`: per-language and pooled attention summaries
- `summary.csv`: language-pass counts for all five conditions
- `prompt_manifest.csv`: 96 frozen prompts with sources
- `run_metadata.json`: run versions, selection metadata, and caveats
