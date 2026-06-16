"""
layer0_attention_comparison.py — Validates the layer-0 convergence hypothesis (W4).

The paper claims a "striking pattern": Chinese/Russian in GPT-2 are governed by
layer-0 heads, and Qwen-Instruct also localizes to layer 0. The reviewer
correctly notes this is correlational — two phenomena both happen to involve
layer 0. This script tests whether they share the same MECHANISM.

Test: Do the layer-0 heads that govern Chinese/Russian in GPT-2 (L0H0, L0H1,
L0H7 from Table 7) also show broadcaster-type attention patterns (high
first-token attention, low entropy) — the same functional signature as L6H1?

If yes: all three phenomena (EU broadcasting, non-EU layer-0, instruct layer-0)
share a unified "first-token broadcaster" mechanism. The layer-0 convergence
is mechanistic, not merely topological.

If no: the layer-0 co-occurrence is coincidental, and we should say so.

Reads:
  results/extended_full_sweep.csv  (from extended_languages.py)

Writes:
  results/layer0_attention_comparison.csv
  figures/fig_layer0_comparison.pdf

Runtime: ~10 min on RTX 3060
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
HEAD_DIM   = 64
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Heads to compare (Table 4 + Table 7 from paper)
HEADS_TO_COMPARE = {
    # EU broadcaster heads (known, from Table 4)
    "EU-broadcaster": [
        (6, 1,  "L6H1",  "EU-broadcaster", "#2E86AB"),
        (10, 4, "L10H4", "EU-broadcaster", "#2E86AB"),
        (7, 3,  "L7H3",  "EU-broadcaster", "#2E86AB"),
    ],
    # Layer-0 non-EU heads (from Table 7, right column)
    "non-EU-layer0": [
        (0, 0, "L0H0", "non-EU-layer0", "#E84855"),
        (0, 1, "L0H1", "non-EU-layer0", "#E84855"),
        (0, 7, "L0H7", "non-EU-layer0", "#E84855"),
    ],
    # Distributed head (control from Table 4)
    "distributed": [
        (1, 10, "L1H10", "distributed", "#AAAAAA"),
    ],
}

# Prompts for attention analysis
# Use both EU and non-EU so we can check script-specificity
ANALYSIS_PROMPTS = [
    # EU (French, German, Spanish, Italian)
    ("Le temps aujourd'hui est très", "fr"),
    ("Das Wetter heute ist sehr", "de"),
    ("El tiempo hoy es muy", "es"),
    ("Il tempo oggi è molto", "it"),
    ("Les scientifiques ont découvert que", "fr"),
    ("Wissenschaftler haben entdeckt, dass", "de"),
    # Chinese
    ("今天的天气非常", "zh"),
    ("我想告诉你关于", "zh"),
    ("科学家们发现了", "zh"),
    # Russian
    ("Сегодня погода очень", "ru"),
    ("Я хотел бы рассказать вам о", "ru"),
    ("Учёные обнаружили, что", "ru"),
]

SAMPLE_STEPS = [1, 5, 10, 20, 40]


def load_model():
    tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    return model, tok


def get_attention_stats(model, tok, prompt, layer, head, n_steps=10):
    """
    Generate n_steps tokens autoregressively.
    At each step, record:
      - first-token attention weight (head attends to token 0)
      - attention entropy

    Returns: (mean_first_tok, mean_entropy) averaged over steps.
    """
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = inputs["input_ids"].clone()

    ft_weights = []
    entropies  = []

    with torch.no_grad():
        for _ in range(n_steps):
            out = model(generated_ids, output_attentions=True)

            if out.attentions is not None and layer < len(out.attentions):
                attn = out.attentions[layer]  # (batch, heads, seq, seq)
                last_query = attn[0, head, -1, :]  # attending from last position

                ft_w = last_query[0].item()  # weight on token 0
                eps = 1e-9
                ent = -(last_query * (last_query + eps).log()).sum().item()

                ft_weights.append(ft_w)
                entropies.append(ent)

            # Greedy next token
            next_tok = out.logits[0, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat(
                [generated_ids, next_tok.unsqueeze(0)], dim=1
            )

    return (
        np.mean(ft_weights) if ft_weights else float("nan"),
        np.mean(entropies)  if entropies  else float("nan"),
    )


def run_comparison(model, tok):
    print("=== LAYER-0 ATTENTION PATTERN COMPARISON ===\n")
    records = []

    all_heads = []
    for group_heads in HEADS_TO_COMPARE.values():
        all_heads.extend(group_heads)

    for layer, head, head_id, group, color in tqdm(all_heads, desc="Heads"):
        print(f"\n  Analyzing {head_id} ({group})...")

        ft_eu, ent_eu = [], []
        ft_noneu, ent_noneu = [], []

        for prompt, lang in tqdm(ANALYSIS_PROMPTS, desc=f"  {head_id}", leave=False):
            ft_w, ent = get_attention_stats(model, tok, prompt, layer, head)
            is_eu = lang in ["fr", "de", "es", "it"]

            if is_eu:
                ft_eu.append(ft_w); ent_eu.append(ent)
            else:
                ft_noneu.append(ft_w); ent_noneu.append(ent)

            records.append({
                "head_id":   head_id,
                "layer":     layer,
                "head":      head,
                "group":     group,
                "prompt":    prompt[:40],
                "lang":      lang,
                "script":    "latin" if lang in ["fr","de","es","it","en"] else "non-latin",
                "ft_weight": ft_w,
                "entropy":   ent,
            })

        eu_ft   = np.mean(ft_eu)   if ft_eu   else float("nan")
        eu_ent  = np.mean(ent_eu)  if ent_eu  else float("nan")
        neu_ft  = np.mean(ft_noneu) if ft_noneu else float("nan")
        neu_ent = np.mean(ent_noneu) if ent_noneu else float("nan")

        print(f"    EU  prompts: ft_attn={eu_ft:.3f}  entropy={eu_ent:.3f}")
        print(f"    Non-EU:      ft_attn={neu_ft:.3f}  entropy={neu_ent:.3f}")

    return pd.DataFrame(records)


def analyze_and_plot(df):
    # Summary per head
    summary = df.groupby(["head_id", "group", "layer"]).agg(
        mean_ft=("ft_weight", "mean"),
        std_ft=("ft_weight", "std"),
        mean_ent=("entropy", "mean"),
        std_ent=("entropy", "std"),
    ).reset_index()

    # Per-script breakdown
    script_summary = df.groupby(["head_id", "group", "script"]).agg(
        mean_ft=("ft_weight", "mean"),
        mean_ent=("entropy", "mean"),
    ).reset_index()

    print("\n" + "="*65)
    print("RESULTS: First-Token Attention by Head Group")
    print("="*65)
    print(f"\n{'Head':8s} {'Group':16s} {'Mean FT Attn':>14s} {'Mean Entropy':>14s} {'Broadcaster?':>13s}")
    print("-"*67)

    for _, row in summary.sort_values("mean_ft", ascending=False).iterrows():
        is_broadcaster = row["mean_ft"] > 0.5
        b_label = "YES ✓" if is_broadcaster else "no"
        print(f"{row['head_id']:8s} {row['group']:16s} "
              f"{row['mean_ft']:>14.3f} {row['mean_ent']:>14.3f} "
              f"{b_label:>13s}")

    # Script-specific results
    print("\nFirst-Token Attention by Script:")
    print(f"{'Head':8s} {'Group':16s} {'Latin script':>14s} {'Non-Latin':>12s}")
    print("-"*55)
    for head_id in summary["head_id"]:
        sub = script_summary[script_summary["head_id"] == head_id]
        latin_ft = sub[sub["script"]=="latin"]["mean_ft"].values
        nonlatin_ft = sub[sub["script"]=="non-latin"]["mean_ft"].values
        group = sub["group"].values[0] if len(sub) else ""
        l_str = f"{latin_ft[0]:.3f}" if len(latin_ft) else "N/A"
        n_str = f"{nonlatin_ft[0]:.3f}" if len(nonlatin_ft) else "N/A"
        print(f"{head_id:8s} {group:16s} {l_str:>14s} {n_str:>12s}")

    # Key claim assessment
    eu_bcast = summary[summary["group"]=="EU-broadcaster"]["mean_ft"].mean()
    noneu_l0 = summary[summary["group"]=="non-EU-layer0"]["mean_ft"].mean()
    dist     = summary[summary["group"]=="distributed"]["mean_ft"].mean()

    print(f"\nKEY COMPARISON:")
    print(f"  EU broadcaster heads (mean ft_attn):      {eu_bcast:.3f}")
    print(f"  Non-EU layer-0 heads (mean ft_attn):      {noneu_l0:.3f}")
    print(f"  Distributed control head (mean ft_attn):  {dist:.3f}")

    if noneu_l0 > 0.5:
        verdict = "UNIFIED MECHANISM SUPPORTED"
        paper_claim = (
            "Layer-0 heads governing Chinese and Russian in GPT-2 share "
            f"the broadcaster signature of EU heads (mean first-token attention "
            f"{noneu_l0:.3f} vs {eu_bcast:.3f} for EU broadcaster heads), "
            "suggesting a unified first-token broadcasting mechanism across "
            "language families, localized at different layers depending on "
            "script-level ambiguity."
        )
    elif noneu_l0 > 0.3:
        verdict = "PARTIAL SUPPORT"
        paper_claim = (
            f"Layer-0 heads governing non-Latin scripts show moderate "
            f"first-token attention ({noneu_l0:.3f}), weaker than EU broadcaster "
            f"heads ({eu_bcast:.3f}). The layer-0 convergence across language "
            "families may reflect a shared mechanism with different strength, "
            "or distinct mechanisms that both resolve language identity early."
        )
    else:
        verdict = "MECHANISM DIFFERS"
        paper_claim = (
            f"Layer-0 heads for non-Latin scripts do not show the broadcaster "
            f"signature (ft_attn={noneu_l0:.3f}), suggesting the layer-0 "
            "convergence reflects different mechanisms for EU and non-EU "
            "language families, connected by locus but not by attention pattern."
        )

    print(f"\nVerdict: {verdict}")
    print(f"\nPaper sentence (Section 12 discussion):")
    print(f"  \"{paper_claim}\"")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    colors_map = {
        "EU-broadcaster": "#2E86AB",
        "non-EU-layer0":  "#E84855",
        "distributed":    "#AAAAAA",
    }

    for ax, metric, label in [
        (axes[0], "mean_ft",  "Mean First-Token Attention Weight"),
        (axes[1], "mean_ent", "Mean Attention Entropy"),
    ]:
        for _, row in summary.iterrows():
            c = colors_map.get(row["group"], "#888888")
            ax.bar(row["head_id"], row[metric], color=c,
                   yerr=row[metric.replace("mean","std")],
                   capsize=3, width=0.6, zorder=2)

        ax.set_ylabel(label)
        ax.set_xlabel("Head")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        if metric == "mean_ft":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                       label="Broadcaster threshold (0.5)")
            ax.legend(fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="EU broadcaster heads"),
        Patch(facecolor="#E84855", label="Non-EU layer-0 heads"),
        Patch(facecolor="#AAAAAA", label="Distributed (control)"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.suptitle(
        "Layer-0 Convergence: Do Non-EU Heads Share the Broadcaster Mechanism?",
        fontweight="bold", fontsize=10
    )
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        plt.savefig(f"{FIGURES_DIR}/fig_layer0_comparison.{ext}",
                    bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\nSaved → {FIGURES_DIR}/fig_layer0_comparison.pdf")

    return summary


if __name__ == "__main__":
    model, tok = load_model()
    df = run_comparison(model, tok)
    df.to_csv(f"{RESULTS_DIR}/layer0_attention_comparison.csv", index=False)
    summary = analyze_and_plot(df)
    print("\n✓ Done. Add fig_layer0_comparison.pdf to Overleaf.")
    print("Add paper sentence to Section 12 (Discussion).")