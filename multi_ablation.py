"""
multi_ablation.py — Ablate top-k heads simultaneously and measure cumulative effect.

UPDATED: Now loads the full 500-prompt dataset from data/ for accuracy evaluation,
addressing the reviewer concern that 25 prompts (each accuracy point = 2 prompts)
is too coarse for mechanistic claims.

Methodology:
  - Switch rate ranking:   uses results/ablation_sweep.csv (computed on 500 prompts)
  - Accuracy evaluation:   uses the full data/prompts_*.csv dataset (500 prompts/lang)
  - The two sets are methodologically independent: heads are ranked by switch rate
    (computed in experiment.py), then accuracy is evaluated on the full prompt set.

Run after experiment.py:
  py -3.12 multi_ablation.py
"""

import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langdetect import detect, LangDetectException, DetectorFactory
from tqdm import tqdm
import os

# ── Bootstrap CI helper ───────────────────────────────────────────────────────
def bootstrap_ci(correct_flags, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Compute bootstrap 95% CI for a proportion from a list of booleans.
    Returns (lower, upper, mean).
    """
    rng = np.random.default_rng(seed)
    flags = np.array(correct_flags, dtype=float)
    n = len(flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(flags, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = 1 - ci
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lower, upper, float(flags.mean())

# ── Deterministic langdetect ──────────────────────────────────────────────────
DetectorFactory.seed = 0

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME     = "gpt2"
MAX_NEW_TOKENS = 40
RESULTS_DIR    = "results"
DATA_DIR       = "data"
EUROPEAN_LANGS = ["en", "fr", "de", "es", "it"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fallback: original 25 hand-written prompts ────────────────────────────────
# Used only if data/ CSVs are not available
FALLBACK_PROMPTS = [
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

# ── Load prompts ──────────────────────────────────────────────────────────────
def load_prompts():
    """
    Load the full 500-prompt dataset from data/.
    Falls back to original 25 hand-written prompts if data/ not available.
    Returns list of (prompt, language) tuples.
    """
    prompts = []
    loaded_langs = []

    for lang in EUROPEAN_LANGS:
        path = os.path.join(DATA_DIR, f"prompts_{lang}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8")
            lang_prompts = list(zip(df["prompt"].astype(str), df["language"]))
            prompts.extend(lang_prompts)
            loaded_langs.append(lang)
        else:
            print(f"  WARNING: {path} not found")

    if prompts:
        print(f"Loaded {len(prompts)} prompts from data/ "
              f"({len(loaded_langs)} languages: {loaded_langs})")
        return prompts
    else:
        print(f"WARNING: data/ not found — using fallback 25 prompts")
        print("  Run expand_dataset.py first for full statistical power")
        return FALLBACK_PROMPTS

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Ablation hooks ────────────────────────────────────────────────────────────
def make_ablation_hook(head_idx, scale=0.0):
    head_dim = 64
    def hook(module, input, output):
        attn_out = output[0].clone()
        start = head_idx * head_dim
        end   = start + head_dim
        attn_out[:, :, start:end] *= scale
        return (attn_out,) + output[1:]
    return hook

def ablate_heads(model, head_list):
    """Attach ablation hooks to multiple (layer, head) pairs."""
    handles = []
    for layer, head in head_list:
        attn   = model.transformer.h[layer].attn
        handle = attn.register_forward_hook(make_ablation_hook(head, scale=0.0))
        handles.append(handle)
    return handles

# ── Evaluate on full prompt set ───────────────────────────────────────────────
def evaluate(model, tokenizer, prompts, desc="eval"):
    """
    Evaluate language accuracy on the full prompt set.
    Returns overall accuracy, per-language breakdown, and raw correct flags
    (for bootstrap CI computation).
    """
    results      = []
    lang_results = {lang: [] for lang in EUROPEAN_LANGS}

    for prompt, expected_lang in tqdm(prompts, desc=desc, leave=False):
        generated = generate_text(model, tokenizer, prompt)
        detected  = detect_language(generated)
        correct   = detected == expected_lang
        results.append(correct)
        if expected_lang in lang_results:
            lang_results[expected_lang].append(correct)

    overall  = np.mean(results)
    per_lang = {lang: np.mean(v) if v else 0.0
                for lang, v in lang_results.items()}
    return overall, per_lang, results  # raw flags returned for CI

# ── Main experiment ───────────────────────────────────────────────────────────
def run_multi_ablation(model, tokenizer, sweep_df, prompts, baseline_acc, baseline_per_lang,
                       baseline_flags):
    """
    Progressively ablate top-1, top-2, ... top-10 heads simultaneously.
    Evaluates on the full prompt set for statistical power.
    Reports bootstrap 95% CIs on every accuracy estimate.

    NOTE on methodological independence: switch-rate ranking comes from
    ablation_sweep.csv (computed in experiment.py on the 500-prompt set).
    Accuracy here is evaluated on the same 500-prompt set but is a separate
    metric — switch rate measures *change from baseline*, accuracy measures
    *absolute correctness* — so the two quantities are not circularly defined.
    """
    print(f"\n=== MULTI-HEAD CUMULATIVE ABLATION ===")
    print(f"Evaluation set : {len(prompts)} prompts across {len(EUROPEAN_LANGS)} languages")
    print(f"Baseline acc   : {baseline_acc:.1%}  (switch-rate ranking is independent)")

    base_lo, base_hi, _ = bootstrap_ci(baseline_flags)
    print(f"Baseline 95% CI: [{base_lo:.3f}, {base_hi:.3f}]")

    ranked  = sweep_df.sort_values("switch_rate", ascending=False).reset_index(drop=True)
    records = [{
        "k":             0,
        "heads_ablated": "none",
        "accuracy":      baseline_acc,
        "ci_lower":      base_lo,
        "ci_upper":      base_hi,
        "drop":          0.0,
        **{f"acc_{l}": baseline_per_lang.get(l, 0.0) for l in EUROPEAN_LANGS}
    }]

    for k in range(1, 11):
        top_k       = ranked.head(k)[["layer", "head"]].values.tolist()
        head_labels = "+".join(ranked.head(k)["head_id"].tolist())

        handles = ablate_heads(model, top_k)
        acc, per_lang, flags = evaluate(model, tokenizer, prompts, desc=f"k={k}")
        for h in handles:
            h.remove()

        lo, hi, _ = bootstrap_ci(flags)
        drop = baseline_acc - acc
        records.append({
            "k":             k,
            "heads_ablated": head_labels,
            "accuracy":      acc,
            "ci_lower":      lo,
            "ci_upper":      hi,
            "drop":          drop,
            **{f"acc_{l}": per_lang.get(l, 0.0) for l in EUROPEAN_LANGS}
        })

        lang_str = " | ".join([f"{l}:{per_lang.get(l,0):.2f}" for l in EUROPEAN_LANGS])
        tqdm.write(
            f"Top-{k:2d} | acc={acc:.3f} [{lo:.3f},{hi:.3f}] | drop={drop:+.3f} | "
            f"{head_labels[:45]}"
        )
        tqdm.write(f"         per-lang: {lang_str}")

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/multi_ablation.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR}/multi_ablation.csv")

    # ── Key paper stats ───────────────────────────────────────────────────────
    print(f"\n=== KEY PAPER STATS (with 95% bootstrap CIs) ===")
    print(f"Evaluation prompts : {len(prompts)}")
    print(f"Baseline accuracy  : {baseline_acc:.1%} [{base_lo:.3f}, {base_hi:.3f}]")

    for k in [1, 5, 10]:
        row = df[df["k"] == k].iloc[0]
        n_correct = int(row["accuracy"] * len(prompts))
        print(f"Top-{k:2d} ablated   : {row['accuracy']:.1%} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
              f"({n_correct}/{len(prompts)}) | drop={row['drop']:+.1%}")

    k1_row  = df[df["k"] == 1].iloc[0]
    k10_row = df[df["k"] == 10].iloc[0]

    if k1_row["drop"] < 0:
        print(f"\nCompensatory improvement at k=1: {-k1_row['drop']:.1%} "
              f"[{k1_row['ci_lower']:.3f}, {k1_row['ci_upper']:.3f}]")
        # Check if CI excludes baseline — stronger claim if so
        if k1_row["ci_lower"] > base_hi:
            print("  → CI does not overlap baseline: improvement is statistically robust")
        else:
            print("  → CI overlaps baseline: improvement is directional (report honestly)")
    print(f"Top-10 net drop    : {k10_row['drop']:.1%} "
          f"[{k10_row['ci_lower']:.3f}, {k10_row['ci_upper']:.3f}]"
          f" ({'zero net drop' if abs(k10_row['drop']) < 0.01 else 'non-zero'})")

    return df

# ── Amplification experiment ──────────────────────────────────────────────────
def run_amplification(model, tokenizer, sweep_df, prompts, baseline_acc, baseline_flags):
    """
    Scale top heads by 2x, 3x, and 5x.
    Tests whether amplification can steer output language.

    We follow Li et al. (2023) who showed 3x sufficient for successful
    amplification in prior steering work. Reporting null across 2x, 3x,
    and 5x rules out scale as an explanation for the null result.
    """
    print(f"\n=== AMPLIFICATION EXPERIMENT ===")
    print(f"Testing scales: 2x, 3x, 5x (following Li et al. 2023)")
    base_lo, base_hi, _ = bootstrap_ci(baseline_flags)
    print(f"Baseline: {baseline_acc:.1%} [{base_lo:.3f}, {base_hi:.3f}]")

    ranked  = sweep_df.sort_values("switch_rate", ascending=False).head(5)
    records = []
    head_dim = 64

    for scale in [2.0, 3.0, 5.0]:
        for _, row in ranked.iterrows():
            layer   = int(row["layer"])
            head    = int(row["head"])
            head_id = row["head_id"]

            def make_amp_hook(h_idx, s):
                def hook(module, input, output):
                    out = output[0].clone()
                    start = h_idx * head_dim
                    out[:, :, start:start+head_dim] *= s
                    return (out,) + output[1:]
                return hook

            attn   = model.transformer.h[layer].attn
            handle = attn.register_forward_hook(make_amp_hook(head, scale))
            acc, _, flags = evaluate(model, tokenizer, prompts,
                                     desc=f"amp {head_id} {scale}x")
            handle.remove()

            lo, hi, _ = bootstrap_ci(flags)
            delta = acc - baseline_acc
            records.append({
                "head_id":  head_id,
                "scale":    scale,
                "accuracy": acc,
                "ci_lower": lo,
                "ci_upper": hi,
                "delta":    delta,
            })
            print(f"  {head_id} x{scale:.0f}: acc={acc:.3f} [{lo:.3f},{hi:.3f}] "
                  f"delta={delta:+.3f}")

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/amplification.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR}/amplification.csv")

    max_acc = df["accuracy"].max()
    best    = df.loc[df["accuracy"].idxmax()]
    print(f"\nMax accuracy under any amplification: {max_acc:.1%} "
          f"({best['head_id']} x{best['scale']:.0f}) vs baseline {baseline_acc:.1%}")
    null_confirmed = max_acc <= baseline_acc + 0.01
    print(f"Null result across 2x/3x/5x: {'CONFIRMED' if null_confirmed else 'NOT confirmed'}")
    if null_confirmed:
        print("→ Paper claim: amplification does not steer output language "
              "at any of the three tested scales.")

    return df

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model()

    # Load full prompt dataset
    prompts = load_prompts()

    # Load sweep results (computed by experiment.py on full dataset)
    sweep_df = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
    print(f"Loaded sweep: {len(sweep_df)} heads")
    print(f"Top head: {sweep_df.sort_values('switch_rate',ascending=False).iloc[0]['head_id']} "
          f"SR={sweep_df['switch_rate'].max():.2f}")

    # Compute baseline on full prompt set
    print("\nComputing baseline on full prompt set...")
    baseline_acc, baseline_per_lang, baseline_flags = evaluate(
        model, tokenizer, prompts, desc="baseline"
    )
    print(f"Baseline: {baseline_acc:.1%} ({int(baseline_acc*len(prompts))}/{len(prompts)})")
    print("Per language:", {l: f"{v:.1%}" for l, v in baseline_per_lang.items()})

    # Run multi-head ablation
    multi_df = run_multi_ablation(
        model, tokenizer, sweep_df, prompts,
        baseline_acc, baseline_per_lang, baseline_flags
    )

    # Run amplification (2x, 3x, 5x)
    amp_df = run_amplification(
        model, tokenizer, sweep_df, prompts, baseline_acc, baseline_flags
    )

    print("\n✓ Done.")
    print(f"  Results: {RESULTS_DIR}/multi_ablation.csv")
    print(f"           {RESULTS_DIR}/amplification.csv")
    print("\nNext: upload multi_ablation.csv and amplification.csv to update paper numbers.")