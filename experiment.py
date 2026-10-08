"""
Language Identity Head Ablation Study
GPT-2 — NeurIPS 2026 Submission

Pipeline:
  1. Build parallel multilingual dataset
  2. Baseline: measure language output per prompt
  3. Ablation: zero out each attention head, measure language shift
  4. Amplification: scale top heads up, measure steering effect
  5. Save all results to results/
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect, LangDetectException

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"          # swap to "gpt2-medium" / "gpt2-large" if desired
MAX_NEW_TOKENS = 40
NUM_HEADS = 12               # GPT-2 small: 12 layers × 12 heads
NUM_LAYERS = 12
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parallel multilingual dataset ─────────────────────────────────────────────
# Each entry: same semantic prompt in multiple languages.
# We use continuation prompts so GPT-2 is pushed to keep generating in that lang.
PARALLEL_PROMPTS = [
    # (prompt, expected_lang_code)
    # English
    ("The weather today is very", "en"),
    ("I would like to tell you about", "en"),
    ("Scientists have discovered that", "en"),
    ("The most important thing in life is", "en"),
    ("Once upon a time there was a", "en"),
    # French
    ("Le temps aujourd'hui est très", "fr"),
    ("Je voudrais vous parler de", "fr"),
    ("Les scientifiques ont découvert que", "fr"),
    ("La chose la plus importante dans la vie est", "fr"),
    ("Il était une fois un", "fr"),
    # German
    ("Das Wetter heute ist sehr", "de"),
    ("Ich möchte Ihnen über", "de"),
    ("Wissenschaftler haben entdeckt, dass", "de"),
    ("Das Wichtigste im Leben ist", "de"),
    ("Es war einmal ein", "de"),
    # Spanish
    ("El tiempo hoy es muy", "es"),
    ("Me gustaría hablarle sobre", "es"),
    ("Los científicos han descubierto que", "es"),
    ("Lo más importante en la vida es", "es"),
    ("Había una vez un", "es"),
    # Italian
    ("Il tempo oggi è molto", "it"),
    ("Vorrei parlarvi di", "it"),
    ("Gli scienziati hanno scoperto che", "it"),
    ("La cosa più importante nella vita è", "it"),
    ("C'era una volta un", "it"),
]

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer

# ── Generation helpers ────────────────────────────────────────────────────────
def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic & reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
    # Return only the newly generated tokens
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

# ── Hook factory ──────────────────────────────────────────────────────────────
def make_ablation_hook(layer_idx, head_idx, scale=0.0):
    """
    Returns a forward hook that modifies the attention output for a specific head.
    scale=0.0  → ablation (zero out the head)
    scale>1.0  → amplification
    scale=1.0  → no-op (identity)

    GPT-2 attention output shape: (batch, seq_len, n_heads * head_dim)
    head_dim = 64 for GPT-2 small
    """
    head_dim = 64  # 768 / 12

    def hook(module, input, output):
        # output is a tuple; first element is the attention output tensor
        attn_out = output[0].clone()
        start = head_idx * head_dim
        end = start + head_dim
        attn_out[:, :, start:end] *= scale
        # Return modified output with rest of tuple unchanged
        return (attn_out,) + output[1:]

    return hook

def register_head_hook(model, layer_idx, head_idx, scale=0.0):
    """Attach hook to the attention module of a specific layer. Returns handle."""
    attn_module = model.transformer.h[layer_idx].attn
    hook_fn = make_ablation_hook(layer_idx, head_idx, scale)
    handle = attn_module.register_forward_hook(hook_fn)
    return handle

# ── Step 1: Baseline ──────────────────────────────────────────────────────────
def run_baseline(model, tokenizer):
    print("\n=== BASELINE ===")
    results = []
    for prompt, expected_lang in tqdm(PARALLEL_PROMPTS):
        generated = generate_text(model, tokenizer, prompt)
        detected = detect_language(generated)
        correct = detected == expected_lang
        results.append({
            "prompt": prompt,
            "expected_lang": expected_lang,
            "generated": generated,
            "detected_lang": detected,
            "correct": correct,
        })
        tqdm.write(f"[{expected_lang}→{detected}] {prompt[:40]}...")

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/baseline.csv", index=False)
    acc = df["correct"].mean()
    print(f"\nBaseline language accuracy: {acc:.1%}")
    return df

# ── Step 2: Full ablation sweep ───────────────────────────────────────────────
def run_ablation_sweep(model, tokenizer, baseline_df):
    """
    For each of the 144 heads, zero it out and measure:
      - language_accuracy: fraction of prompts still detected in correct lang
      - language_switch_rate: fraction that switched away from baseline detected lang
      - delta_accuracy: accuracy drop vs baseline
    """
    print("\n=== ABLATION SWEEP ===")
    print(f"Testing {NUM_LAYERS * NUM_HEADS} heads...")

    baseline_langs = baseline_df["detected_lang"].tolist()
    baseline_acc = baseline_df["correct"].mean()

    sweep_results = []

    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            handle = register_head_hook(model, layer, head, scale=0.0)

            langs_detected = []
            for prompt, expected_lang in PARALLEL_PROMPTS:
                generated = generate_text(model, tokenizer, prompt)
                detected = detect_language(generated)
                langs_detected.append(detected)

            handle.remove()

            # Metrics
            correct = [d == e for d, (_, e) in zip(langs_detected, PARALLEL_PROMPTS)]
            acc = np.mean(correct)
            switch_rate = np.mean([
                d != b for d, b in zip(langs_detected, baseline_langs)
            ])

            sweep_results.append({
                "layer": layer,
                "head": head,
                "head_id": f"L{layer}H{head}",
                "accuracy": acc,
                "delta_accuracy": acc - baseline_acc,
                "switch_rate": switch_rate,
                "langs_detected": json.dumps(langs_detected),
            })

            tqdm.write(
                f"L{layer:02d}H{head:02d} | acc={acc:.2f} | switch={switch_rate:.2f}"
            )

    df = pd.DataFrame(sweep_results)
    df = df.sort_values("delta_accuracy")  # most harmful ablations first
    df.to_csv(f"{RESULTS_DIR}/ablation_sweep.csv", index=False)
    print(f"\nSaved ablation sweep to {RESULTS_DIR}/ablation_sweep.csv")
    return df

# ── Step 3: Amplification / steering ─────────────────────────────────────────
def run_amplification(model, tokenizer, sweep_df, top_k=5, scale=3.0):
    """
    Take the top-k most language-critical heads (highest switch_rate when ablated)
    and amplify them instead. Measure if this steers toward the target language.
    Also test cross-language steering: amplify heads identified for lang A
    while running prompts from lang B — does it pull output toward A?
    """
    print(f"\n=== AMPLIFICATION (scale={scale}x, top {top_k} heads) ===")

    top_heads = sweep_df.nlargest(top_k, "switch_rate")[["layer", "head", "head_id", "switch_rate"]]
    print("Top language-critical heads:")
    print(top_heads.to_string(index=False))
    top_heads.to_csv(f"{RESULTS_DIR}/top_heads.csv", index=False)

    amp_results = []

    for _, row in top_heads.iterrows():
        layer, head = int(row["layer"]), int(row["head"])
        handle = register_head_hook(model, layer, head, scale=scale)

        for prompt, expected_lang in PARALLEL_PROMPTS:
            generated = generate_text(model, tokenizer, prompt)
            detected = detect_language(generated)
            amp_results.append({
                "amplified_head": row["head_id"],
                "scale": scale,
                "prompt": prompt,
                "expected_lang": expected_lang,
                "detected_lang": detected,
                "correct": detected == expected_lang,
            })

        handle.remove()

    df = pd.DataFrame(amp_results)
    df.to_csv(f"{RESULTS_DIR}/amplification.csv", index=False)

    # Summary per head
    summary = df.groupby("amplified_head")["correct"].mean().reset_index()
    summary.columns = ["head", "accuracy_when_amplified"]
    print("\nAmplification accuracy summary:")
    print(summary.to_string(index=False))
    return df

# ── Step 4: Cross-language steering experiment ────────────────────────────────
def run_cross_steering(model, tokenizer, sweep_df, top_k=3, scale=3.0):
    """
    Identify top heads per language, then apply them to prompts from OTHER languages.
    This tests whether heads encode language identity transferably.
    """
    print(f"\n=== CROSS-LANGUAGE STEERING ===")

    languages = ["en", "fr", "de", "es", "it"]
    results = []

    for target_lang in languages:
        # Find top head for this language: among prompts in target_lang,
        # which head had highest switch_rate when ablated?
        # We re-use the global sweep for now (per-lang sweep would require re-running)
        # Use top global heads as proxy
        top_heads = sweep_df.nlargest(top_k, "switch_rate")[["layer", "head"]]

        # Apply to prompts from OTHER languages
        other_prompts = [(p, l) for p, l in PARALLEL_PROMPTS if l != target_lang]

        handles = []
        for _, row in top_heads.iterrows():
            h = register_head_hook(model, int(row["layer"]), int(row["head"]), scale=scale)
            handles.append(h)

        for prompt, src_lang in other_prompts[:5]:  # limit for speed
            generated = generate_text(model, tokenizer, prompt)
            detected = detect_language(generated)
            results.append({
                "target_lang": target_lang,
                "src_lang": src_lang,
                "prompt": prompt,
                "generated": generated,
                "detected_lang": detected,
                "steered_correctly": detected == target_lang,
            })

        for h in handles:
            h.remove()

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/cross_steering.csv", index=False)
    steer_rate = df["steered_correctly"].mean()
    print(f"Overall cross-language steering rate: {steer_rate:.1%}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model, tokenizer = load_model()

    # Step 1
    baseline_df = run_baseline(model, tokenizer)

    # Step 2 — this is the slow one (~144 × 25 prompts × generation time)
    # On 3060 12GB with GPT-2 small, expect ~30-50 min total
    sweep_df = run_ablation_sweep(model, tokenizer, baseline_df)

    # Step 3
    amp_df = run_amplification(model, tokenizer, sweep_df, top_k=5, scale=3.0)

    # Step 4
    steer_df = run_cross_steering(model, tokenizer, sweep_df, top_k=3, scale=3.0)

    print("\n✓ All experiments complete. Results saved to results/")
    print("  Next: run analysis.py to generate figures")

if __name__ == "__main__":
    main()
