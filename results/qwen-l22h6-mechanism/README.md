# Qwen2.5-1.5B-Instruct L22H6: decoding-stage and language-instruction analyses

**Research status:** Exploratory, with within-environment replay validation. The results below do not establish a complete or independently replicated mechanism of language control.

## Research question and design

This study investigates whether Qwen2.5-1.5B-Instruct attention head L22H6 contributes to response-language consistency, and whether its role differs between prompt prefill and autoregressive decoding. The experiments use float32 weights, eager attention, greedy decoding, and interventions applied to the attention output projection or the attention mask at layer 22.

The 96-prompt analysis set is a fixed sample of previously baseline-correct examples from the **same Language Confusion Benchmark (LCB)** distribution as the preceding LIHA evaluation. It is not an external holdout or an independently sampled benchmark. The 32-prompt semantic-contrast pilot uses newly authored prompts that mention an incidental language in addition to explicitly requesting a response language. All reported comparisons are exploratory; the source prompt sets were used during analysis development.

## Findings

### Phase-specific head ablation

On the 24 Italian prompts in the 96-prompt analysis set:

| Intervention | Responses passing the line-level language check |
| --- | ---: |
| No ablation | 24/24 |
| L22H8 whole-head control | 24/24 |
| L22H6, prefill only | 21/24 |
| L22H6, decoding only | 0/24 |
| L22H6, prefill and decoding | 0/24 |

The contrast is concentrated in Italian; the remaining three languages show smaller or mixed effects. Decode-only interventions leave prompt prefill and the first generated token unchanged, then act on subsequent cached decoding steps. These results are conditional on the baseline-correct-enriched selection of prompts.

### Attention to explicit language-name tokens

Clean-generation attention weights assign the following mean mass to the requested-language word at the **last prompt position**:

| Requested language | L22H6 attention mass |
| --- | ---: |
| German | 0.705 |
| Spanish | 0.867 |
| French | 0.791 |
| Italian | 0.865 |

The head also attends to those tokens during continuation, with lower average mass later in the response. Same-layer control heads show far lower attention on the designated tokens. These are attention weights, not a direct measure of transferred information. In particular, comparable prompt-boundary attention to Italian and Spanish does not explain their contrasting vulnerability to head ablation.

### Decode-time attention-edge interventions

For 95 clean-correct, scorable prompts, masking L22H6 attention to the explicit requested-language word produced **seven line-level language-check failures**. Masking a nearby non-language token at the same head, or masking the language word at control head L22H8, produced **none**. The paired nearby-versus-language comparison had seven discordant cases in one direction and zero in the other (two-sided exact binomial/McNemar p = 0.015625). This exploratory p-value is not a confirmatory significance result.

Manual review found six sustained switches away from the requested language and one mixed-language response with a non-target-language heading. Therefore, **seven scored failures should not be described as seven sustained switches**. Whole-head decode ablation had a larger effect than masking this single token connection; language-token access alone does not account for the full head-ablation behavior.

The nearby-token control matches local position and token count, **not baseline attention mass**. It serves as a perturbation control but cannot isolate semantic content from attention strength.

### Requested versus incidental language: 32-prompt pilot

The semantic-contrast prompts pair an explicit response-language instruction with a reference to a differently named language in an unrelated handbook label. L22H6 placed substantially more attention on requested-language tokens than incidental-language tokens. For Italian, requested-token masking reduced language-check passes from **8/8 to 4/8**; incidental-token masking and the control-head mask retained **8/8**. Whole-head decode ablation yielded **0/8**.

Among 31 clean-correct prompts across all four languages, the requested-versus-incidental comparison had four discordant failures versus zero (two-sided exact p = 0.125). Since the two token positions receive highly unequal attention mass and the language pairings are limited, this pilot **does not establish semantic specificity under an attention-matched control**.

### First-divergence logit diagnostics

The 20-prompt diagnostic tracks changes in the logit margin between the tokens selected at the first clean-versus-ablated generation divergence. Because those tokens are selected *after observing a divergence*, a negative change in their margin is partly expected by construction. Token-replay agreement was also imperfect, and the original logit-generation calls did not pass an explicit attention mask. These saved values are retained for transparency, **not used as independent logit-attribution evidence**. The consolidated runner now supplies explicit attention masks in new diagnostic calls; its historical CSVs have not been overwritten.

## Reproduction

The repository contains two scripts:

- `tools/analyze_l22h6_results.py`: checks the saved CSVs and recomputes descriptive results without model inference.
- `experiments/liha_l22h6_reproduce.py`: runs phase ablation, attention measurement, decode-time attention-edge masks, semantic contrast, and first-divergence diagnostics.

From the repository root, using Python with a **CUDA-enabled PyTorch installation**, install the additional dependencies and run:

```bash
pip install -r requirements-l22h6.txt
python tools/analyze_l22h6_results.py
python experiments/liha_l22h6_reproduce.py --mode phase --smoke
python experiments/liha_l22h6_reproduce.py --mode edge --smoke
python experiments/liha_l22h6_reproduce.py --mode semantic --smoke
python experiments/liha_l22h6_reproduce.py --mode all
```

The model must be available through Hugging Face. The fastText `lid.176.bin` language identification model is downloaded on first use to `out/qwen-l22h6-mechanism/lid.176.bin` (under the repository’s ignored `out/` directory), unless `--lid` specifies another path. The scripts write new outputs under `out/qwen-l22h6-mechanism/`; committed result files are not overwritten. The `--smoke` option uses two prompts per language and is intended only to check execution, not to reproduce the paper's full results. See `--help` for additional options, including `--revision` to set a model commit.

**Environment and model version:** The original experimental environment recorded PyTorch `2.11.0+cu130`, Transformers `4.57.6`, float32 weights, and eager attention. The Hugging Face model commit was not recorded, and the full dependency environment was not locked; exact future response reproduction is therefore not guaranteed. When rerunning the experiments, record a resolved model revision and software versions separately from the historical metadata.

**Scoring:** The output-language check follows the LIHA/LCB line-based fastText procedure: remove punctuation, split into lines containing at least five words, and classify with `lid.176.bin` at a 0.3 confidence threshold. An output passes only when every scorable line matches the requested language. This can mark a largely correct-language response as failing due to a single off-language heading. The operational scoring rule corresponds to the benchmark setup under review in [LIHA PR #11](https://github.com/arjunpil/LIHA-First-Token-Broadcasters/pull/11).

## Validation performed and remaining limitations

The consolidated phase and attention-edge routines were checked on an NVIDIA L4 GPU:

- Four-prompt execution smoke test: four of four clean replies exactly matched historical clean replies; first-token and hook assertions passed.
- Follow-up replay of the 96-prompt analysis set: **96/96 phase-clean** and **96/96 attention-edge-clean** responses exactly matched the saved clean replies. Italian pass counts also matched across the replayed intervention conditions: phase decode-only 0/24, prefill-only 21/24; edge whole-head zero 0/24, requested-language mask 20/24.

These are **within-environment replay checks**, not an independent-machine replication. Non-clean responses were not all compared byte-for-byte, and the standalone fresh-process command-line entry point has not been independently exercised across every mode. The semantic-contrast and logit-diagnostic CLI paths have not been separately GPU-validated. This README reports verification already completed; it does not imply those remaining checks passed.

Additional limitations:

- Attention concentration cannot on its own establish causal use or a language-specific representation.
- A decode-stage hook using cached generation is not equivalent to fresh full-forward evaluation at each token.
- Prompt selection, intervention choices, language group comparisons, and associated p-values are exploratory and subject to selection and multiple-comparison concerns.
- No claim is made that instruction tuning created this head, that it encodes only language identity, or that access to the named language word fully mediates the head's effects.

## Included data

All paths below are relative to `results/qwen-l22h6-mechanism/` unless otherwise noted.

| File | Content |
| --- | --- |
| `prompt_manifest_96.csv` | Frozen 96-prompt sample with language and source fields |
| `liha_l22h6_phase_validation.csv` | Phase interventions, 96 prompts × 5 conditions |
| `liha_l22h6_attention_analysis.csv` | 96 prompt-level attention measurements |
| `liha_l22h6_language_edge_validation.csv` | Language-token, nearby-token, control-head, and whole-head comparisons; 96 × 5 |
| `liha_l22h6_semantic_contrast.csv` | Semantic-contrast pilot, 32 × 5 |
| `liha_l22h6_logit_divergence.csv` | Exploratory first-divergence diagnostics, 20 prompts |
| `liha_l22h6_language_edge_metadata.json` | Intervention settings and environment summary |
| `liha_l22h6_semantic_contrast_metadata.json` | Semantic-contrast design metadata |

The original model revision and any unrecorded environment settings cannot be reconstructed from these files.
