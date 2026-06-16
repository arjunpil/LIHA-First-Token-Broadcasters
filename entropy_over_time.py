"""
entropy_over_time.py — Broadcaster Head Entropy Over Generation Timesteps.

Shows that broadcaster heads (L0H4, L9H9, L6H1) maintain low entropy
(focused first-token attention) persistently throughout generation,
not just at the first token. This strengthens the "persistent broadcasting"
claim in the paper.

Method:
  For each generation step (token 1, 5, 10, 20, 40):
    - Record attention entropy of top broadcaster heads
    - Compare against random heads (baseline)
  Plot entropy over time for broadcasters vs random heads.

Run:
  py -3.12 entropy_over_time.py

Saves:
  results/entropy_over_time.csv
  figures/fig_entropy_over_time.pdf
  figures/fig_entropy_over_time.png
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "gpt2"
HEAD_DIM    = 64
NUM_LAYERS  = 12
NUM_HEADS   = 12
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Heads to track
BROADCASTER_HEADS = [
    (0, 4,  "L0H4",  "#E84855"),
    (9, 9,  "L9H9",  "#FF8C00"),
    (6, 1,  "L6H1",  "#2E86AB"),
]

# Random heads for comparison (middle-layer, mid-index — unremarkable)
RANDOM_HEADS = [
    (2, 5,  "L2H5 (rand)",  "#AAAAAA"),
    (4, 8,  "L4H8 (rand)",  "#CCCCCC"),
    (8, 3,  "L8H3 (rand)",  "#BBBBBB"),
]

# Generation steps to sample entropy at
SAMPLE_STEPS = [1, 5, 10, 20, 30, 40]

# Prompts — use non-English so language signal is in the first token
PROMPTS = [
    "Le temps aujourd'hui est très",
    "Das Wetter heute ist sehr",
    "El tiempo hoy es muy",
    "Il tempo oggi è molto",
    "Les scientifiques ont découvert que",
    "Wissenschaftler haben entdeckt, dass",
    "Los científicos han descubierto que",
    "Gli scienziati hanno scoperto che",
    "Je voudrais vous parler de",
    "Ich möchte Ihnen über",
    "La chose la plus importante dans la vie est",
    "Das Wichtigste im Leben ist",
    "Lo más importante en la vida es",
    "La cosa più importante nella vita è",
    "Il était une fois un",
]

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)
    model.eval()
    return model, tok

def attention_entropy(attn_weights):
    """Compute entropy of attention distribution. Lower = more focused."""
    # attn_weights: (heads, seq_len, seq_len) or (seq_len, seq_len)
    eps = 1e-9
    p = attn_weights + eps
    return -torch.sum(p * torch.log(p), dim=-1)  # (..., seq_len)

def get_entropy_at_step(model, tokenizer, prompt, step, target_heads):
    """
    Generate `step` tokens autoregressively, then extract attention
    entropy at the last generated token position for target heads.

    Returns: dict {head_id: entropy_value}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    # Generate step tokens one at a time to capture attention at each step
    generated_ids = inputs["input_ids"].clone()
    entropies = {h[2]: [] for h in target_heads}

    with torch.no_grad():
        for t in range(step):
            out = model(generated_ids, output_attentions=True)

            if out.attentions is not None:
                for l, h, name, _ in target_heads:
                    if l < len(out.attentions) and out.attentions[l] is not None:
                        # attn: (batch, heads, seq, seq)
                        attn = out.attentions[l][0, h, -1, :]  # last query pos
                        ent = (-torch.sum(
                            (attn + 1e-9) * torch.log(attn + 1e-9)
                        )).item()
                        entropies[name].append(ent)

            # Greedy next token
            next_token = out.logits[0, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat(
                [generated_ids, next_token.unsqueeze(0)], dim=1
            )

    # Return entropy at the last step only
    return {name: vals[-1] if vals else float("nan")
            for name, vals in entropies.items()}


def run_entropy_experiment(model, tokenizer):
    print("\n=== ENTROPY OVER TIME EXPERIMENT ===")
    all_heads = BROADCASTER_HEADS + RANDOM_HEADS
    records   = []

    for prompt in tqdm(PROMPTS, desc="Prompts"):
        for step in SAMPLE_STEPS:
            ents = get_entropy_at_step(model, tokenizer, prompt, step, all_heads)
            for _, _, name, _ in all_heads:
                records.append({
                    "prompt":   prompt[:30],
                    "step":     step,
                    "head_id":  name,
                    "entropy":  ents.get(name, float("nan")),
                    "is_broadcaster": name in [h[2] for h in BROADCASTER_HEADS],
                })

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/entropy_over_time.csv", index=False)
    print(f"Saved → {RESULTS_DIR}/entropy_over_time.csv")
    return df


def plot_entropy(df):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    all_heads = BROADCASTER_HEADS + RANDOM_HEADS

    for l, h, name, color in all_heads:
        sub     = df[df["head_id"] == name]
        by_step = sub.groupby("step")["entropy"].agg(["mean","std"]).reset_index()

        ls = "-" if name in [x[2] for x in BROADCASTER_HEADS] else "--"
        lw = 2.0 if name in [x[2] for x in BROADCASTER_HEADS] else 1.2
        ax.plot(by_step["step"], by_step["mean"],
                color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=4, label=name, zorder=3)
        ax.fill_between(
            by_step["step"],
            by_step["mean"] - by_step["std"],
            by_step["mean"] + by_step["std"],
            alpha=0.12, color=color
        )

    ax.set_xlabel("Generation step (tokens generated)")
    ax.set_ylabel("Attention entropy")
    ax.set_title("Broadcaster Heads Maintain Low Entropy Throughout Generation",
                 fontweight="bold", pad=5)
    ax.set_xticks(SAMPLE_STEPS)
    ax.grid(alpha=0.25, linewidth=0.5)

    # Separate broadcaster vs random in legend
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="upper left",
                    framealpha=0.9, ncol=2, handlelength=1.6)

    # Add annotation
    ax.annotate("Broadcaster heads\n(low, stable entropy)",
                xy=(40, df[df["is_broadcaster"]==True].groupby("step")["entropy"]
                    .mean().iloc[-1]),
                xytext=(28, 0.4), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#E84855", lw=1.0),
                color="#E84855")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_entropy_over_time.pdf",
                bbox_inches="tight", format="pdf")
    plt.savefig(f"{FIGURES_DIR}/fig_entropy_over_time.png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved → {FIGURES_DIR}/fig_entropy_over_time.pdf")


def print_summary(df):
    print("\n=== ENTROPY SUMMARY ===")
    summary = df.groupby(["head_id","is_broadcaster"])["entropy"].agg(
        ["mean","std","min","max"]
    ).round(3)
    print(summary.to_string())

    b_mean = df[df["is_broadcaster"]==True]["entropy"].mean()
    r_mean = df[df["is_broadcaster"]==False]["entropy"].mean()
    print(f"\nBroadcaster mean entropy: {b_mean:.3f}")
    print(f"Random head mean entropy: {r_mean:.3f}")
    print(f"Ratio: {r_mean/b_mean:.2f}x higher for random heads")


if __name__ == "__main__":
    model, tokenizer = load_model()
    df = run_entropy_experiment(model, tokenizer)
    plot_entropy(df)
    print_summary(df)
    print("\nDone. Add fig_entropy_over_time.pdf to Overleaf.")