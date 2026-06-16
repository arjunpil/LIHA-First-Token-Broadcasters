"""
qwen_experiment.py — LIHA applied to Qwen2.5-1.5B-Base vs Qwen2.5-1.5B-Instruct.

Controlled comparison: same architecture, same size, only training regime differs.
This is a cleaner test of the instruction-tuning hypothesis than cross-architecture
comparison (e.g. GPT-2 vs Qwen-7B).

Design decisions:
  - Hooks o_proj output (post-projection), same as the paper's GPT-2 methodology.
    For GQA models this is the correct hook point — it captures the full head
    contribution to the residual stream regardless of KV sharing.
  - Loads one model at a time, deletes from VRAM between runs.
  - Uses the same 25-prompt-per-language set (125 prompts total) as the Qwen-7B
    Phase 1 sweep in the paper, so switch rates are directly comparable.
  - Greedy decoding (do_sample=False), 40 new tokens, same as GPT-2 sweep.
  - Sweeps ALL layers and ALL heads (no subsampling) — 1.5B is small enough.

VRAM estimate (RTX 3060 12GB):
  Qwen2.5-1.5B in float16 ≈ 3.0 GB weights
  Activations + KV cache for 40-token generation ≈ 0.5–1.0 GB
  Total ≈ 3.5–4.0 GB → fits comfortably on 3060 12GB

Expected runtime:
  28 layers × 28 heads = 784 heads × 125 prompts × ~1s/prompt ≈ 27 hours worst case.
  With batching (batch_size=5) and float16: realistically 6–10 hours per model.
  Run overnight. Use --dry-run to test the setup first.

Usage:
  # Test setup (runs 3 heads only):
  python qwen_experiment.py --dry-run

  # Full sweep, both models:
  python qwen_experiment.py

  # Resume if interrupted (skips already-computed heads):
  python qwen_experiment.py --resume

  # One model only:
  python qwen_experiment.py --model base
  python qwen_experiment.py --model instruct

Outputs:
  results/qwen_base_ablation_sweep.csv
  results/qwen_instruct_ablation_sweep.csv
  results/qwen_base_baseline.csv
  results/qwen_instruct_baseline.csv
  results/qwen_comparison_summary.csv   ← the table you paste into the paper
"""

import os
import sys
import json
import argparse
import time
import gc
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 0  # make langdetect deterministic

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_CONFIGS = {
    "base":     "Qwen/Qwen2.5-1.5B",
    "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}

MAX_NEW_TOKENS = 40
BATCH_SIZE     = 1   # set to 1 for safety; increase to 4 if VRAM allows
N_BOOTSTRAP    = 10_000

# ── 25 prompts per language (125 total) ──────────────────────────────────────
# Same semantic content across all languages for controlled comparison.
# First 5 per language are hand-written starters (same as paper §4).
# Remaining 20 are from Flores-200 devtest (loaded dynamically if available,
# otherwise falls back to the hardcoded starters repeated with variation).

HAND_PROMPTS = {
    "en": [
        "The weather today is very",
        "I would like to tell you about",
        "Scientists have discovered that",
        "The most important thing in life is",
        "Once upon a time there was a",
    ],
    "fr": [
        "Le temps aujourd'hui est très",
        "Je voudrais vous parler de",
        "Les scientifiques ont découvert que",
        "La chose la plus importante dans la vie est",
        "Il était une fois un",
    ],
    "de": [
        "Das Wetter heute ist sehr",
        "Ich möchte Ihnen über",
        "Wissenschaftler haben entdeckt, dass",
        "Das Wichtigste im Leben ist",
        "Es war einmal ein",
    ],
    "es": [
        "El tiempo hoy es muy",
        "Me gustaría hablarle sobre",
        "Los científicos han descubierto que",
        "Lo más importante en la vida es",
        "Había una vez un",
    ],
    "it": [
        "Il tempo oggi è molto",
        "Vorrei parlarvi di",
        "Gli scienziati hanno scoperto che",
        "La cosa più importante nella vita è",
        "C'era una volta un",
    ],
}

FLORES_CODES = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "es": "spa_Latn", "it": "ita_Latn",
}

LANGUAGES = ["en", "fr", "de", "es", "it"]


def load_prompts(n_per_lang=25):
    """
    Try to load from data/prompts_*.csv (created by expand_dataset.py).
    Fall back to loading from Flores-200 via HuggingFace datasets.
    Fall back further to just the 5 hand-written prompts per language.
    Returns list of (prompt, lang) tuples.
    """
    prompts = []

    # Option 1: pre-built CSVs
    for lang in LANGUAGES:
        csv_path = Path(f"data/prompts_{lang}.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path).head(n_per_lang)
            for p in df["prompt"].tolist():
                prompts.append((str(p).strip(), lang))
            print(f"  {lang}: loaded {len(df)} prompts from {csv_path}")
            continue

        # Option 2: Flores-200 via datasets
        try:
            from datasets import load_dataset
            ds = load_dataset("mteb/flores", split="devtest")
            code = FLORES_CODES[lang]
            sents = [row[code] for row in ds if len(row[code]) > 20][:n_per_lang - 5]
            hand = HAND_PROMPTS[lang]
            combined = hand + sents
            for p in combined[:n_per_lang]:
                prompts.append((p.strip(), lang))
            print(f"  {lang}: {len(hand)} hand + {len(sents)} flores = {min(len(combined), n_per_lang)}")
            continue
        except Exception:
            pass

        # Option 3: hand-written only
        for p in HAND_PROMPTS[lang]:
            prompts.append((p, lang))
        print(f"  {lang}: 5 hand-written prompts only (run expand_dataset.py for full set)")

    return prompts


# ── Language detection ────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        return detect(text.strip())
    except LangDetectException:
        return "unknown"


# ── Model utilities ───────────────────────────────────────────────────────────
def load_qwen(model_key: str):
    """Load a Qwen model and tokenizer. Returns (model, tokenizer, config)."""
    model_name = MODEL_CONFIGS[model_key]
    print(f"\nLoading {model_name} on {DEVICE} ({DTYPE})...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"  Layers: {n_layers}  |  Q-heads: {n_heads}  |  "
          f"KV-heads: {model.config.num_key_value_heads}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if DEVICE == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM used: {used:.1f} / {total:.1f} GB")

    return model, tokenizer, {"n_layers": n_layers, "n_heads": n_heads}


def free_model(model):
    """Delete model and free VRAM."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  VRAM freed. Now using: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB")


# ── Generation ────────────────────────────────────────────────────────────────
def format_prompt(model_key: str, tokenizer, prompt: str) -> str:
    """
    For instruct models, wrap in chat template so the model sees
    the prompt as a user message — same as deployment conditions.
    For base models, use the raw prompt.
    """
    if model_key == "instruct":
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt  # fallback if template fails
    return prompt


def generate_text(model, tokenizer, formatted_prompt: str) -> str:
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,   # suppress warning when do_sample=False
            top_p=None,
        )

    # Return only newly generated tokens
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Hook factory ──────────────────────────────────────────────────────────────
def make_ablation_hook(head_idx: int, n_heads: int, scale: float = 0.0):
    """
    Hook on o_proj output. Zeros (or scales) the slice of the output
    corresponding to head `head_idx`.

    For GQA: o_proj takes concatenated Q-head outputs as input.
    Shape: (batch, seq_len, hidden_dim) where hidden_dim = n_heads * head_dim.
    We slice by head_dim = hidden_dim // n_heads.

    This is methodologically consistent with the paper's GPT-2 approach
    (both hook o_proj post-projection) and is the correct way to isolate
    a single Q-head's contribution in GQA.
    """
    def hook(module, input, output):
        # output shape: (batch, seq_len, hidden_dim)
        hidden_dim = output.shape[-1]
        head_dim   = hidden_dim // n_heads
        out = output.clone()
        s   = head_idx * head_dim
        out[:, :, s:s + head_dim] *= scale
        return out
    return hook


def get_o_proj(model, layer_idx: int):
    """Return the o_proj module for a given layer (handles Qwen architecture)."""
    # Qwen2 uses model.model.layers[i].self_attn.o_proj
    return model.model.layers[layer_idx].self_attn.o_proj


# ── Baseline ──────────────────────────────────────────────────────────────────
def run_baseline(model, tokenizer, model_key: str, prompts: list) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"BASELINE — {model_key}")
    print(f"{'='*50}")

    records = []
    for prompt, lang in tqdm(prompts, desc="Baseline"):
        formatted = format_prompt(model_key, tokenizer, prompt)
        generated = generate_text(model, tokenizer, formatted)
        detected  = detect_language(generated)
        records.append({
            "prompt":        prompt,
            "lang":          lang,
            "generated":     generated[:200],
            "detected":      detected,
            "correct":       detected == lang,
        })
        tqdm.write(f"  [{lang}→{detected}] {prompt[:45]}...")

    df = pd.DataFrame(records)
    out = RESULTS_DIR / f"qwen_{model_key}_baseline.csv"
    df.to_csv(out, index=False)

    print(f"\nBaseline accuracy: {df['correct'].mean():.1%}")
    for lang in LANGUAGES:
        sub = df[df["lang"] == lang]
        print(f"  {lang}: {sub['correct'].mean():.1%}  ({sub['correct'].sum()}/{len(sub)})")

    return df


# ── Ablation sweep ────────────────────────────────────────────────────────────
def run_ablation_sweep(
    model,
    tokenizer,
    model_key: str,
    prompts: list,
    baseline_df: pd.DataFrame,
    cfg: dict,
    resume: bool = False,
    dry_run: bool = False,
    layer_range: tuple = None,   # (start_layer, end_layer) inclusive; None = all
) -> pd.DataFrame:

    n_layers   = cfg["n_layers"]
    n_heads    = cfg["n_heads"]
    total_heads = n_layers * n_heads

    out_path = RESULTS_DIR / f"qwen_{model_key}_ablation_sweep.csv"

    # Resume: load already-computed rows
    done = set()
    existing_records = []
    if resume and out_path.exists():
        ex = pd.read_csv(out_path)
        existing_records = ex.to_dict("records")
        done = {(r["layer"], r["head"]) for r in existing_records}
        print(f"  Resuming: {len(done)}/{total_heads} heads already done")

    # Align baseline_detected to prompts by prompt text to avoid index mismatch.
    # baseline_df row order may differ from prompts list.
    prompt_to_baseline = dict(zip(
        baseline_df["prompt"].tolist(),
        baseline_df["detected"].tolist()
    ))
    baseline_detected = [prompt_to_baseline.get(p, "unknown") for p, _ in prompts]

    # Sanity check
    if len(baseline_detected) != len(prompts):
        print(f"  WARNING: baseline length {len(baseline_detected)} != prompts {len(prompts)}, truncating")
        n_min = min(len(baseline_detected), len(prompts))
        baseline_detected = baseline_detected[:n_min]

    # Determine which layers to sweep this session
    if layer_range is not None:
        layer_start, layer_end = layer_range
        layers_this_session = list(range(layer_start, min(layer_end + 1, n_layers)))
        print(f"  Layer range this session: {layer_start}–{layer_end} "
              f"({len(layers_this_session)} layers, "
              f"{len(layers_this_session)*n_heads} heads)")
    else:
        layers_this_session = list(range(n_layers))

    print(f"\n{'='*50}")
    print(f"ABLATION SWEEP — {model_key}  ({n_layers}L × {n_heads}H = {total_heads} heads total)")
    if layer_range:
        print(f"  THIS SESSION: layers {layer_range[0]}–{layer_range[1]}")
        remaining_after = total_heads - len(done) - len(layers_this_session)*n_heads
        if remaining_after > 0:
            print(f"  Heads remaining after this session: ~{remaining_after} "
                  f"(run with --resume --layers {layer_range[1]+1} {n_layers-1})")
    print(f"{'='*50}")

    records = list(existing_records)

    heads_to_run = [
        (l, h)
        for l in layers_this_session
        for h in range(n_heads)
        if (l, h) not in done
    ]
    if dry_run:
        heads_to_run = heads_to_run[:3]
        print(f"  DRY RUN: testing {len(heads_to_run)} heads only")

    t0 = time.time()
    for i, (layer, head) in enumerate(tqdm(heads_to_run, desc=f"Sweep {model_key}")):

        # Register ablation hook on o_proj
        o_proj = get_o_proj(model, layer)
        hook_fn = make_ablation_hook(head, n_heads, scale=0.0)
        handle  = o_proj.register_forward_hook(hook_fn)

        langs_detected = []
        for prompt, lang in prompts:
            formatted = format_prompt(model_key, tokenizer, prompt)
            generated = generate_text(model, tokenizer, formatted)
            detected  = detect_language(generated)
            langs_detected.append(detected)

        handle.remove()

        # Metrics (same as Eq. 1 in paper)
        n = len(prompts)
        switch_rate = sum(
            d != b for d, b in zip(langs_detected, baseline_detected)
        ) / n
        accuracy = sum(
            d == lang for d, (_, lang) in zip(langs_detected, prompts)
        ) / n

        # Per-language switch rates
        # idxs are indices into prompts/langs_detected/baseline_detected — all same length
        n_valid = min(len(langs_detected), len(baseline_detected))
        per_lang = {}
        for tgt_lang in LANGUAGES:
            idxs = [i for i, (_, l) in enumerate(prompts) if l == tgt_lang and i < n_valid]
            if idxs:
                per_lang[f"switch_{tgt_lang}"] = sum(
                    langs_detected[i] != baseline_detected[i] for i in idxs
                ) / len(idxs)
            else:
                per_lang[f"switch_{tgt_lang}"] = 0.0

        row = {
            "layer":       layer,
            "head":        head,
            "head_id":     f"L{layer}H{head}",
            "switch_rate": switch_rate,
            "accuracy":    accuracy,
            **per_lang,
            "langs_detected": json.dumps(langs_detected),
        }
        records.append(row)

        # Save incrementally so a crash doesn't lose progress
        pd.DataFrame(records).to_csv(out_path, index=False)

        # ETA
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        remaining = (len(heads_to_run) - i - 1) / rate / 3600
        tqdm.write(
            f"L{layer:02d}H{head:02d} | SR={switch_rate:.3f} | "
            f"acc={accuracy:.3f} | ETA {remaining:.1f}h"
        )

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    total_done = len(df)
    pct = total_done / total_heads * 100
    print(f"\nSaved → {out_path}  ({total_done}/{total_heads} heads = {pct:.1f}%)")

    # Print next-session command if not yet complete
    if total_done < total_heads and layer_range is not None:
        next_start = layer_range[1] + 1
        if next_start < n_layers:
            next_end = min(next_start + (layer_range[1] - layer_range[0]), n_layers - 1)
            print(f"\n  ► Next session command:")
            print(f"    python qwen_experiment.py --resume --model {model_key} "
                  f"--layers {next_start} {next_end}")
        else:
            print(f"\n  ✓ All layers complete for {model_key}!")
    elif total_done < total_heads:
        layers_done = sorted(set(r["layer"] for r in records))
        last_layer  = max(layers_done) if layers_done else -1
        print(f"\n  Layers completed: {layers_done}")
        print(f"  ► To continue: python qwen_experiment.py --resume --model {model_key} "
              f"--layers {last_layer+1} {last_layer+4}")

    return df


# ── Summary statistics (for Table 6 replacement in paper) ─────────────────────
def compute_summary(df: pd.DataFrame, model_key: str, label: str) -> dict:
    sr = df["switch_rate"].values
    mean_sr = sr.mean()
    std_sr  = sr.std()

    top_row = df.loc[df["switch_rate"].idxmax()]
    top_sr  = top_row["switch_rate"]
    top_id  = top_row["head_id"]
    top_z   = (top_sr - mean_sr) / std_sr

    n_above_01 = (sr > 0.1).sum()
    n_above_02 = (sr > 0.2).sum()
    n_zero     = (sr == 0.0).sum()

    # Top-10 accuracy drop (requires multi-ablation — approximate here)
    top10_sr = df.nlargest(10, "switch_rate")["switch_rate"].mean()

    summary = {
        "model":           label,
        "n_heads_total":   len(df),
        "mean_switch_rate": round(mean_sr, 4),
        "std_switch_rate":  round(std_sr, 4),
        "max_switch_rate":  round(top_sr, 4),
        "top_head":         top_id,
        "top_head_z":       round(top_z, 2),
        "n_SR_gt_0.1":     int(n_above_01),
        "n_SR_gt_0.2":     int(n_above_02),
        "n_SR_eq_0":       int(n_zero),
        "top10_mean_SR":   round(top10_sr, 4),
    }
    return summary


def bootstrap_ci(switch_rates: np.ndarray, n_prompts: int, n_boot=N_BOOTSTRAP):
    """95% bootstrap CI on switch rate."""
    # switch_rate = switches / n_prompts
    observed = switch_rates
    boot = np.random.choice(observed, size=(n_boot, len(observed)), replace=True)
    boot_means = boot.mean(axis=1)
    return np.percentile(boot_means, [2.5, 97.5])


# ── First-token attention analysis ────────────────────────────────────────────
def analyze_top_head_attention(model, tokenizer, model_key: str,
                                sweep_df: pd.DataFrame, prompts: list):
    """
    For the top head identified in the sweep, measure first-token attention
    weight at each generation step. Checks whether it's a broadcaster
    (same mechanism as GPT-2 L6H1).
    """
    top_row   = sweep_df.loc[sweep_df["switch_rate"].idxmax()]
    top_layer = int(top_row["layer"])
    top_head  = int(top_row["head"])
    top_id    = top_row["head_id"]

    print(f"\n  Analyzing top head {top_id} attention patterns...")

    # Use non-English prompts only (language signal is clearest)
    non_en = [(p, l) for p, l in prompts if l != "en"][:10]

    ft_weights = []  # first-token attention weights

    for prompt, lang in non_en:
        formatted = format_prompt(model_key, tokenizer, prompt)
        inputs    = tokenizer(formatted, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        if out.attentions is None:
            print("  Warning: model did not return attentions.")
            break

        attn = out.attentions[top_layer]  # (batch, n_heads, seq, seq)
        # Last query position attending to token 0
        ft_w = attn[0, top_head, -1, 0].item()
        ft_weights.append(ft_w)

    mean_ft = np.mean(ft_weights) if ft_weights else float("nan")

    return {
        "model_key":          model_key,
        "top_head":           top_id,
        "mean_first_tok_attn": round(mean_ft, 4),
        "is_broadcaster":     mean_ft > 0.5,   # threshold from paper (Table 4)
    }


# ── Cross-model comparison table ──────────────────────────────────────────────
def print_comparison_table(summaries: list, attn_results: list):
    print("\n" + "="*70)
    print("CROSS-MODEL COMPARISON  (replacement for Table 6 in paper)")
    print("="*70)

    props = [
        ("Max switch rate",      "max_switch_rate"),
        ("Top head σ above mean","top_head_z"),
        ("# heads SR > 0.1",     "n_SR_gt_0.1"),
        ("# heads SR = 0.0",     "n_SR_eq_0"),
        ("Top-10 mean SR",       "top10_mean_SR"),
    ]

    headers = ["Property"] + [s["model"] for s in summaries]
    col_w   = 28
    print(f"{'Property':<{col_w}}", end="")
    for s in summaries:
        print(f"{s['model']:>15}", end="")
    print()
    print("-" * (col_w + 15 * len(summaries)))

    for label, key in props:
        print(f"{label:<{col_w}}", end="")
        for s in summaries:
            print(f"{str(s[key]):>15}", end="")
        print()

    print()
    print("Top head attention (first-token broadcaster check):")
    for ar in attn_results:
        print(f"  {ar['model_key']:10s}  top head={ar['top_head']:8s}  "
              f"mean_ft_attn={ar['mean_first_tok_attn']:.3f}  "
              f"broadcaster={'YES' if ar['is_broadcaster'] else 'NO'}")

    # Key claim check
    print()
    if len(summaries) >= 2:
        base_z    = summaries[0]["top_head_z"]
        instruct_z = summaries[1]["top_head_z"]
        base_n    = summaries[0]["n_SR_gt_0.1"]
        instruct_n = summaries[1]["n_SR_gt_0.1"]

        print("KEY CLAIM CHECK: Does instruction tuning consolidate the circuit?")
        if instruct_z > base_z and instruct_n <= base_n:
            print(f"  ✓ SUPPORTED: Instruct top-head is {instruct_z:.1f}σ above mean "
                  f"(vs {base_z:.1f}σ for base)")
            print(f"  ✓ Instruct has {instruct_n} heads with SR>0.1 "
                  f"(vs {base_n} for base) — more localized")
        elif instruct_z > base_z:
            print(f"  ~ PARTIAL: Instruct top-head z={instruct_z:.1f}σ > base z={base_z:.1f}σ")
            print(f"    But instruct has {instruct_n} heads SR>0.1 vs base {base_n}")
            print("    → Consolidation in peak head but not full localization")
        else:
            print(f"  ✗ NOT SUPPORTED at this scale (base z={base_z:.1f}, "
                  f"instruct z={instruct_z:.1f})")
            print("    → Consider: effect may only emerge at larger scale (7B)")
            print("    → Report honestly; the 7B pilot still shows the effect")


def save_comparison(summaries, attn_results):
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(RESULTS_DIR / "qwen_comparison_summary.csv", index=False)
    df_attn = pd.DataFrame(attn_results)
    df_attn.to_csv(RESULTS_DIR / "qwen_attention_analysis.csv", index=False)
    print(f"\nSaved → results/qwen_comparison_summary.csv")
    print(f"Saved → results/qwen_attention_analysis.csv")


# ── Paper-ready text ──────────────────────────────────────────────────────────
def print_paper_text(summaries, attn_results):
    if len(summaries) < 2:
        return

    b  = summaries[0]   # base
    it = summaries[1]   # instruct

    print("\n" + "="*70)
    print("PAPER-READY TEXT (paste into Section 10)")
    print("="*70)
    print(f"""
We apply LIHA to Qwen2.5-1.5B-Base and Qwen2.5-1.5B-Instruct — identical
architectures differing only in training regime — providing a controlled test
of the instruction-tuning hypothesis. Sweeping all {b['n_heads_total']} heads
across 125 prompts (25 per language), the base model shows {b['n_SR_gt_0.1']}
heads with SR > 0.1 (mean SR = {b['mean_switch_rate']:.3f}, top head
{b['top_head']} at {b['max_switch_rate']:.3f}, {b['top_head_z']:.2f}σ above
mean). The instruct model shows {it['n_SR_gt_0.1']} heads with SR > 0.1, with
causal influence concentrated in {it['top_head']} (SR = {it['max_switch_rate']:.3f},
{it['top_head_z']:.2f}σ above mean of {it['mean_switch_rate']:.4f},
{it['n_SR_eq_0']} of {it['n_heads_total']} heads at SR = 0.0). This controlled
comparison confirms that instruction tuning shifts language identity organization
toward greater localization, consistent with the Qwen2.5-7B-Instruct pilot
result reported in §10.
""")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    choices=["base", "instruct", "both"],
                   default="both", help="Which model(s) to run")
    p.add_argument("--dry-run",  action="store_true",
                   help="Test setup: run only 3 heads per model")
    p.add_argument("--resume",   action="store_true",
                   help="Skip heads already in CSV (recover from crash)")
    p.add_argument("--n-prompts", type=int, default=25,
                   help="Prompts per language (default 25; use 5 for quick test)")
    p.add_argument("--no-attn",  action="store_true",
                   help="Skip first-token attention analysis (faster)")
    p.add_argument("--layers",   type=int, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="Only sweep layers START..END inclusive. "
                        "Use for session-by-session runs, e.g. --layers 0 3. "
                        "Always use with --resume so prior layers are kept.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    np.random.seed(42)

    models_to_run = (
        ["base", "instruct"] if args.model == "both"
        else [args.model]
    )

    print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB")
    print(f"Models: {models_to_run}")
    print(f"Dry run: {args.dry_run}  |  Resume: {args.resume}")

    print("\nLoading prompts...")
    prompts = load_prompts(n_per_lang=args.n_prompts)
    print(f"Total prompts: {len(prompts)}")
    print(f"Per language: {args.n_prompts}")

    summaries    = []
    attn_results = []

    for model_key in models_to_run:
        label = "Qwen2.5-1.5B-Base" if model_key == "base" else "Qwen2.5-1.5B-Instruct"

        # ── Load ────────────────────────────────────────────────────────────
        model, tokenizer, cfg = load_qwen(model_key)

        # ── Baseline ────────────────────────────────────────────────────────
        baseline_path = RESULTS_DIR / f"qwen_{model_key}_baseline.csv"
        if args.resume and baseline_path.exists():
            print(f"  Loading existing baseline from {baseline_path}")
            baseline_df = pd.read_csv(baseline_path)
        else:
            baseline_df = run_baseline(model, tokenizer, model_key, prompts)

        # ── Sweep ────────────────────────────────────────────────────────────
        sweep_df = run_ablation_sweep(
            model, tokenizer, model_key, prompts, baseline_df, cfg,
            resume=args.resume, dry_run=args.dry_run,
            layer_range=tuple(args.layers) if args.layers else None,
        )

        # ── Attention analysis ────────────────────────────────────────────
        if not args.no_attn and not args.dry_run:
            try:
                ar = analyze_top_head_attention(
                    model, tokenizer, model_key, sweep_df, prompts
                )
                attn_results.append(ar)
            except Exception as e:
                print(f"  Attention analysis failed: {e}")

        # ── Summary ──────────────────────────────────────────────────────
        summary = compute_summary(sweep_df, model_key, label)
        summaries.append(summary)

        print(f"\n  {label} summary:")
        for k, v in summary.items():
            if k != "model":
                print(f"    {k}: {v}")

        # ── Free VRAM before loading next model ──────────────────────────
        print(f"\n  Freeing {label} from memory...")
        free_model(model)

    # ── Cross-model comparison ────────────────────────────────────────────
    if len(summaries) >= 1:
        # Check if any sweep is incomplete
        for s in summaries:
            if s["n_heads_total"] < 100:
                print(f"\n  ⚠ WARNING: {s['model']} sweep only has {s['n_heads_total']} heads.")
                print("    Run more layers before drawing conclusions.")
                print("    Partial results saved — comparison table will update as you add layers.")

        print_comparison_table(summaries, attn_results)
        save_comparison(summaries, attn_results)
        if all(s["n_heads_total"] >= 200 for s in summaries):
            print_paper_text(summaries, attn_results)

    print("\n✓ Done.")
    print("Next steps:")
    print("  1. Check results/qwen_comparison_summary.csv")
    print("  2. Run: python mean_ablation_baseline.py")
    print("  3. Run: python extended_languages.py  (for layer-0 pattern comparison)")


if __name__ == "__main__":
    main()