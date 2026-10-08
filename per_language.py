"""
per_language.py — Identify which heads matter for each language specifically.

For each language, we measure: when we ablate head X, how often does that
specific language's prompts switch away? This tells us whether heads are
general-purpose language controllers or language-specific specialists.

Key question: Does L6H1 matter equally for French, German, Spanish, Italian?
Or do different heads specialize for different languages?
"""

import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect, LangDetectException
from tqdm import tqdm
import json
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 40
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PARALLEL_PROMPTS = [
    ("The weather today is very", "en"),
    ("I would like to tell you about", "en"),
    ("Scientists have discovered that", "en"),
    ("The most important thing in life is", "en"),
    ("Once upon a time there was a", "en"),
    ("Le temps aujourd'hui est très", "fr"),
    ("Je voudrais vous parler de", "fr"),
    ("Les scientifiques ont découvert que", "fr"),
    ("La chose la plus importante dans la vie est", "fr"),
    ("Il était une fois un", "fr"),
    ("Das Wetter heute ist sehr", "de"),
    ("Ich möchte Ihnen über", "de"),
    ("Wissenschaftler haben entdeckt, dass", "de"),
    ("Das Wichtigste im Leben ist", "de"),
    ("Es war einmal ein", "de"),
    ("El tiempo hoy es muy", "es"),
    ("Me gustaría hablarle sobre", "es"),
    ("Los científicos han descubierto que", "es"),
    ("Lo más importante en la vida es", "es"),
    ("Había una vez un", "es"),
    ("Il tempo oggi è molto", "it"),
    ("Vorrei parlarvi di", "it"),
    ("Gli scienziati hanno scoperto che", "it"),
    ("La cosa più importante nella vita è", "it"),
    ("C'era una volta un", "it"),
]

LANGUAGES = ["en", "fr", "de", "es", "it"]
NUM_LAYERS = 12
NUM_HEADS = 12

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def make_ablation_hook(head_idx, scale=0.0):
    head_dim = 64
    def hook(module, input, output):
        attn_out = output[0].clone()
        start = head_idx * head_dim
        end = start + head_dim
        attn_out[:, :, start:end] *= scale
        return (attn_out,) + output[1:]
    return hook

def run_per_language_sweep(model, tokenizer):
    """
    For each head, compute switch_rate broken down by language.
    This gives us a (144 heads) x (5 languages) matrix of causal influence.
    """
    print("\n=== PER-LANGUAGE ABLATION SWEEP ===")
    print(f"Testing {NUM_LAYERS * NUM_HEADS} heads across {len(LANGUAGES)} languages...")
    print("This will take ~30-40 minutes.\n")

    # First get baseline per language
    baseline = {lang: [] for lang in LANGUAGES}
    for prompt, lang in PARALLEL_PROMPTS:
        generated = generate_text(model, tokenizer, prompt)
        detected = detect_language(generated)
        baseline[lang].append(detected)

    print("Baseline per language:")
    for lang in LANGUAGES:
        acc = np.mean([d == lang for d in baseline[lang]])
        print(f"  {lang}: {acc:.0%}")

    records = []

    for layer in tqdm(range(NUM_LAYERS), desc="Layers"):
        for head in range(NUM_HEADS):
            # Attach ablation hook
            attn = model.transformer.h[layer].attn
            handle = attn.register_forward_hook(make_ablation_hook(head, scale=0.0))

            # Run prompts and collect per-language results
            lang_results = {lang: [] for lang in LANGUAGES}
            for prompt, lang in PARALLEL_PROMPTS:
                generated = generate_text(model, tokenizer, prompt)
                detected = detect_language(generated)
                lang_results[lang].append(detected)

            handle.remove()

            # Compute switch rate per language
            row = {"layer": layer, "head": head, "head_id": f"L{layer}H{head}"}
            for lang in LANGUAGES:
                baseline_langs = baseline[lang]
                ablated_langs = lang_results[lang]
                switch_rate = np.mean([
                    a != b for a, b in zip(ablated_langs, baseline_langs)
                ])
                accuracy = np.mean([d == lang for d in ablated_langs])
                row[f"switch_{lang}"] = switch_rate
                row[f"acc_{lang}"] = accuracy

            # Overall switch rate
            all_baseline = [d for lang in LANGUAGES for d in baseline[lang]]
            all_ablated = [d for lang in LANGUAGES for d in lang_results[lang]]
            row["switch_overall"] = np.mean([a != b for a, b in zip(all_ablated, all_baseline)])

            records.append(row)
            tqdm.write(
                f"L{layer:02d}H{head:02d} | "
                + " ".join([f"{l}:{row[f'switch_{l}']:.2f}" for l in LANGUAGES])
            )

    df = pd.DataFrame(records)
    df = df.sort_values("switch_overall", ascending=False)
    df.to_csv(f"{RESULTS_DIR}/per_language_ablation.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/per_language_ablation.csv")
    return df

def print_specialization_analysis(df):
    """
    Print the key finding: are heads general or language-specific?
    """
    print("\n=== SPECIALIZATION ANALYSIS ===")

    # For each language, find the most impactful head
    print("\nMost impactful head per language:")
    for lang in LANGUAGES:
        top = df.nlargest(1, f"switch_{lang}").iloc[0]
        print(f"  {lang}: {top['head_id']} (switch_rate={top[f'switch_{lang}']:.2f})")

    # Check overlap — are the top heads the same across languages?
    print("\nTop 5 heads per language:")
    for lang in LANGUAGES:
        top5 = df.nlargest(5, f"switch_{lang}")["head_id"].tolist()
        print(f"  {lang}: {top5}")

    # Compute specialization score: std of switch rates across languages
    # High std = this head affects some languages much more than others (specialized)
    # Low std = affects all languages equally (general)
    switch_cols = [f"switch_{lang}" for lang in LANGUAGES]
    df["specialization"] = df[switch_cols].std(axis=1)
    df["mean_switch"] = df[switch_cols].mean(axis=1)

    print("\nMost SPECIALIZED heads (high impact on specific languages):")
    specialized = df.nlargest(5, "specialization")[["head_id", "specialization", "mean_switch"] + switch_cols]
    print(specialized.to_string(index=False))

    print("\nMost GENERAL heads (impact spread across all languages):")
    # High mean switch AND low specialization
    df["generality_score"] = df["mean_switch"] / (df["specialization"] + 0.01)
    general = df.nlargest(5, "generality_score")[["head_id", "mean_switch", "specialization"] + switch_cols]
    print(general.to_string(index=False))

    df.to_csv(f"{RESULTS_DIR}/per_language_ablation.csv", index=False)
    print("\n→ This general vs specialized finding is a key paper result")

if __name__ == "__main__":
    model, tokenizer = load_model()
    df = run_per_language_sweep(model, tokenizer)
    print_specialization_analysis(df)
    print("\n✓ Done. Next: run attention_viz.py")
