# [LIHA-First-Token-Broadcasters] 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the experiment code and data visualization for the paper "First-Token Broadcasters: Mechanistic Origins of Language
Identity and Distributed Robustness in Transformers" published at BlackboxNLP 2026 Workshop

## Abstract
Why do multilingual language models sometimes generate in the wrong
language, and why is this so hard to fix? We introduce **Language
Identity Head Ablation (LIHA)**, a causal intervention that zeros each
attention head individually and measures the resulting language switch
rate across a parallel dataset of 2,700 prompt-language pairs spanning
seven languages. Applied to GPT-2, LIHA identifies a small set of
*first-token broadcaster* heads—led by L6H1 (`switch rate 0.32`,
`3.23σ` above the population mean)—that attend persistently to
the first prompt token, propagating its language signal throughout
generation. Compensatory redistribution when heads are ablated is
statistically significant (`$p < 10^{-5}$)` and follows a directional,
hierarchical pattern: compensation always recruits heads in layers above
the ablated head, suggesting a feedforward cascade rather than global
diffusion. To probe how training regime shapes these circuits, we apply
LIHA to a controlled pair—Qwen2.5-1.5B-Base and
Qwen2.5-1.5B-Instruct—identical in architecture and size, differing
only in training. The base model is nearly flat (max `SR = 0.016`, 200/336 heads at `SR = 0.0`); 
the instruct model concentrates causal influence sharply at layer 0, led by **L0H5** (`SR = 0.224`, `8.93σ` above mean),
with all other layers near zero. This controlled comparison provides direct causal evidence that instruction
tuning reorganizes language identity circuits toward early-layer
localization. Extended experiments with Chinese and Russian confirm
that first-token broadcasting is script-specific in GPT-2, with
non-Latin languages handled at layer 0—the same locus as the
instruction-tuned model.
Code and data will be released upon publication.

## Experiment & Set-up
For reproducibility, we recommend using Conda for Jupyter Notebook. Our experiments were run on __3060 RTX 12 GB__ using __Python 3.12__

```
bash
conda create -n myenv python=3.12
conda activate myenv
pip install -r requirements.txt
```
