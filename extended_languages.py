"""
extended_languages.py — Extend the ablation study to Chinese and Russian.

Adds 5 prompts per language (zh, ru) to the existing dataset and runs:
1. Baseline for new languages through unmodified GPT-2
2. Ablation of the existing top 10 heads on new language prompts
3. Per-language switch rates for new languages

This tests whether the first-token broadcasting mechanism generalizes
to typologically distant languages with different scripts.

Chinese (zh): tonal, logographic, no spaces between words
Russian (ru): Slavic, Cyrillic script, rich morphology

Note: GPT-2 has limited Chinese/Russian capability but that's fine —
we're studying the mechanism, not performance.

Run time: ~20 minutes on GPU (targeted sweep only)
"""

import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect, LangDetectException
from tqdm import tqdm
import os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "gpt2"
MAX_NEW_TOKENS = 40
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Extended prompts: Chinese and Russian ─────────────────────────────────────
# These are naturally occurring sentence starters, not machine-translated.
# Chinese uses simplified characters (mainland standard).
# Russian uses Cyrillic.

EXTENDED_PROMPTS = [
    # Chinese (zh) — sentence starters
    ("今天的天气非常", "zh"),           # The weather today is very
    ("我想告诉你关于", "zh"),           # I want to tell you about
    ("科学家们发现了", "zh"),           # Scientists have discovered
    ("生活中最重要的事是", "zh"),       # The most important thing in life is
    ("从前有一个", "zh"),               # Once upon a time there was a

    # Russian (ru) — sentence starters
    ("Сегодня погода очень", "ru"),     # Today the weather is very
    ("Я хотел бы рассказать вам о", "ru"),  # I would like to tell you about
    ("Учёные обнаружили, что", "ru"),   # Scientists have discovered that
    ("Самое важное в жизни — это", "ru"),   # The most important thing in life is
    ("Однажды жил-был", "ru"),          # Once upon a time there was
]

# Original prompts for comparison
ORIGINAL_PROMPTS = [
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

ALL_PROMPTS = ORIGINAL_PROMPTS + EXTENDED_PROMPTS

# Top heads from original sweep to test on new languages
TOP_HEADS = [
    (6, 1),   # L6H1  — highest switch rate
    (10, 4),  # L10H4
    (7, 3),   # L7H3
    (3, 1),   # L3H1
    (1, 10),  # L1H10
    (1, 7),   # L1H7
    (3, 4),   # L3H4
    (6, 8),   # L6H8
    (9, 11),  # L9H11
    (3, 10),  # L3H10
]

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
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
        end   = start + head_dim
        attn_out[:, :, start:end] *= scale
        return (attn_out,) + output[1:]
    return hook

# ── Step 1: Baseline for new languages ───────────────────────────────────────
def run_extended_baseline(model, tokenizer):
    print("\n=== EXTENDED LANGUAGE BASELINE ===")
    results = []

    for prompt, expected_lang in tqdm(EXTENDED_PROMPTS, desc="Extended baseline"):
        generated = generate_text(model, tokenizer, prompt)
        detected  = detect_language(generated)
        correct   = detected == expected_lang
        results.append({
            "prompt":        prompt,
            "expected_lang": expected_lang,
            "generated":     generated[:120],
            "detected_lang": detected,
            "correct":       correct,
        })
        tqdm.write(f"[{expected_lang}→{detected}] {prompt[:40]}")

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/extended_baseline.csv", index=False)

    for lang in ["zh", "ru"]:
        lang_df = df[df["expected_lang"] == lang]
        acc = lang_df["correct"].mean()
        print(f"{lang} baseline accuracy: {acc:.1%}")

    print("\nNote: Low accuracy expected — GPT-2 has limited zh/ru capability.")
    print("What matters is whether ablating top heads changes the OUTPUT language.")
    return df

# ── Step 2: Targeted ablation on new languages ────────────────────────────────
def run_extended_ablation(model, tokenizer, baseline_df):
    """
    For each of the top 10 heads, run only the new language prompts
    and measure switch rate. This is much faster than a full sweep.
    """
    print("\n=== EXTENDED LANGUAGE ABLATION (top 10 heads) ===")

    baseline_langs = baseline_df["detected_lang"].tolist()
    records = []

    for layer, head in tqdm(TOP_HEADS, desc="Heads"):
        head_id = f"L{layer}H{head}"
        attn    = model.transformer.h[layer].attn
        handle  = attn.register_forward_hook(make_ablation_hook(head, scale=0.0))

        langs_detected = []
        for prompt, _ in EXTENDED_PROMPTS:
            generated = generate_text(model, tokenizer, prompt)
            detected  = detect_language(generated)
            langs_detected.append(detected)

        handle.remove()

        # Overall switch rate for new languages
        switch_rate = np.mean([
            d != b for d, b in zip(langs_detected, baseline_langs)
        ])

        # Per-language switch rates
        zh_langs = langs_detected[:5]
        ru_langs = langs_detected[5:]
        zh_baseline = baseline_langs[:5]
        ru_baseline = baseline_langs[5:]

        zh_switch = np.mean([d != b for d, b in zip(zh_langs, zh_baseline)])
        ru_switch = np.mean([d != b for d, b in zip(ru_langs, ru_baseline)])

        records.append({
            "head_id":    head_id,
            "layer":      layer,
            "head":       head,
            "switch_rate_extended": switch_rate,
            "switch_zh":  zh_switch,
            "switch_ru":  ru_switch,
            "langs_zh":   str(zh_langs),
            "langs_ru":   str(ru_langs),
        })

        tqdm.write(
            f"{head_id} | overall={switch_rate:.2f} | zh={zh_switch:.2f} | ru={ru_switch:.2f}"
        )

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/extended_ablation.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/extended_ablation.csv")

    # Compare with original switch rates
    original_df = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
    print("\n=== CROSS-LANGUAGE COMPARISON ===")
    print(f"{'Head':<10} {'Original SR':>12} {'Extended SR':>13} {'ZH':>8} {'RU':>8}")
    print("-" * 55)
    for _, row in df.iterrows():
        orig = original_df[original_df["head_id"] == row["head_id"]]
        orig_sr = orig["switch_rate"].values[0] if len(orig) > 0 else 0.0
        print(f"{row['head_id']:<10} {orig_sr:>12.3f} {row['switch_rate_extended']:>13.3f} "
              f"{row['switch_zh']:>8.3f} {row['switch_ru']:>8.3f}")

    return df

# ── Step 3: Full sweep on new languages (optional, slower) ───────────────────
def run_full_extended_sweep(model, tokenizer, baseline_df):
    """
    Full 144-head sweep on just the new language prompts.
    Slower (~15 min) but gives complete picture.
    """
    print("\n=== FULL EXTENDED SWEEP (all 144 heads, zh+ru only) ===")
    baseline_langs = baseline_df["detected_lang"].tolist()
    records = []

    for layer in tqdm(range(12), desc="Layers"):
        for head in range(12):
            attn   = model.transformer.h[layer].attn
            handle = attn.register_forward_hook(make_ablation_hook(head, scale=0.0))

            langs = []
            for prompt, _ in EXTENDED_PROMPTS:
                generated = generate_text(model, tokenizer, prompt)
                langs.append(detect_language(generated))

            handle.remove()

            switch_rate = np.mean([d != b for d, b in zip(langs, baseline_langs)])
            zh_switch = np.mean([
                d != b for d, b in zip(langs[:5], baseline_langs[:5])
            ])
            ru_switch = np.mean([
                d != b for d, b in zip(langs[5:], baseline_langs[5:])
            ])

            records.append({
                "layer": layer, "head": head,
                "head_id": f"L{layer}H{head}",
                "switch_rate": switch_rate,
                "switch_zh": zh_switch,
                "switch_ru": ru_switch,
            })

            tqdm.write(f"L{layer:02d}H{head:02d} | {switch_rate:.2f} | zh={zh_switch:.2f} | ru={ru_switch:.2f}")

    df = pd.DataFrame(records)
    df = df.sort_values("switch_rate", ascending=False)
    df.to_csv(f"{RESULTS_DIR}/extended_full_sweep.csv", index=False)

    print("\nTop 10 heads for new languages:")
    print(df.head(10)[["head_id", "switch_rate", "switch_zh", "switch_ru"]].to_string(index=False))
    return df

if __name__ == "__main__":
    model, tokenizer = load_model()

    # Step 1: baseline
    baseline_df = run_extended_baseline(model, tokenizer)

    # Step 2: targeted sweep on top heads (~5 min)
    targeted_df = run_extended_ablation(model, tokenizer, baseline_df)

    # Step 3: full sweep on new languages (~15 min)
    # Comment this out if you want just the targeted results
    full_df = run_full_extended_sweep(model, tokenizer, baseline_df)

    print("\n✓ Extended language experiments complete.")
    print("Results saved to results/extended_*.csv")

    # ── Paper-ready cross-family specificity table ────────────────────────────
    print("\n=== PAPER-READY: Table 6 data (language-family specificity) ===")
    try:
        orig_df  = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
        full_ext = pd.read_csv(f"{RESULTS_DIR}/extended_full_sweep.csv")

        eu_top = orig_df.sort_values("switch_rate", ascending=False).head(5)
        print("\nLeft columns (European top heads on zh/ru):")
        print(f"{'Head':8s} {'EU SR':8s} {'ZH SR':8s} {'RU SR':8s}")
        print("-" * 36)
        for _, row in eu_top.iterrows():
            hid  = row["head_id"]
            eu_sr = row["switch_rate"]
            ext_row = full_ext[full_ext["head_id"] == hid]
            zh_sr = ext_row["switch_zh"].values[0] if len(ext_row) else float("nan")
            ru_sr = ext_row["switch_ru"].values[0] if len(ext_row) else float("nan")
            print(f"{hid:8s} {eu_sr:8.2f} {zh_sr:8.2f} {ru_sr:8.2f}")

        print("\nRight columns (top zh/ru heads):")
        ext_top = full_ext.sort_values("switch_rate", ascending=False).head(5)
        print(f"{'Head':8s} {'Overall':8s} {'ZH':8s} {'RU':8s}")
        print("-" * 36)
        for _, row in ext_top.iterrows():
            print(f"{row['head_id']:8s} {row['switch_rate']:8.2f} "
                  f"{row['switch_zh']:8.2f} {row['switch_ru']:8.2f}")

        # Verify the key claim: do European top heads produce SR=0.0 on zh/ru?
        eu_on_ext = []
        for _, row in eu_top.iterrows():
            ext_row = full_ext[full_ext["head_id"] == row["head_id"]]
            if len(ext_row):
                eu_on_ext.append(max(
                    ext_row["switch_zh"].values[0],
                    ext_row["switch_ru"].values[0]
                ))
        if eu_on_ext and max(eu_on_ext) == 0.0:
            print("\n✓ KEY CLAIM VERIFIED: European top heads produce SR=0.0 on zh/ru.")
        else:
            print(f"\n⚠ KEY CLAIM: max zh/ru SR for European heads = {max(eu_on_ext):.3f} "
                  f"(paper claims 0.0 — check data)")
    except FileNotFoundError as e:
        print(f"  (could not generate table: {e})")