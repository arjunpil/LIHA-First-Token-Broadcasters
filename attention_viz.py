"""
attention_viz.py — Visualize what the critical heads are actually attending to.

Uses forward-method patching to extract attention weights, bypassing the
transformers version issue where output_attentions returns None.

Key finding from debug: L6H1 attends overwhelmingly to the first token
(0.6-0.96 weight on token[0] across all positions). Since the first token
is always the strongest language signal ("Le"=French, "Das"=German, etc.),
this head is broadcasting language identity through the entire sequence.
"""

import torch
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME    = "gpt2"
RESULTS_DIR   = "results"
FIGS_DIR      = "figures"
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Prompts ───────────────────────────────────────────────────────────────────
VIZ_PROMPTS = [
    # (prompt, language, condition)
    ("Le temps aujourd'hui est très",          "fr", "correct"),
    ("Das Wetter heute ist sehr",              "de", "correct"),
    ("El tiempo hoy es muy",                   "es", "correct"),
    ("Il tempo oggi è molto",                  "it", "correct"),
    ("The weather today is very",              "en", "correct"),
    ("Lo más importante en la vida es",        "es", "confused"),
    ("La cosa più importante nella vita è",    "it", "confused"),
    ("C'era una volta un",                     "it", "confused"),
]

TOP_HEADS = [
    (6,  1),   # L6H1  — highest switch rate
    (10, 4),   # L10H4
    (7,  3),   # L7H3
    (1,  10),  # L1H10
]

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return model, tokenizer

# ── Attention extraction via forward patching ─────────────────────────────────
def make_patched_forward(attn_module, store: dict):
    """
    Replace the attention module's forward with a version that saves the
    softmax attention weights to `store['weights']` before dropout.

    This works regardless of transformers version because we're replicating
    the attention math ourselves rather than relying on output_attentions.
    """
    def patched_forward(self, hidden_states, **kwargs):
        # Project to Q, K, V
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.split_size, dim=2)

        def split_heads(x):
            B, T, C = x.size()
            x = x.view(B, T, self.num_heads, C // self.num_heads)
            return x.permute(0, 2, 1, 3)  # (B, heads, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale = 1.0 / (q.size(-1) ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask (lower triangular)
        T = hidden_states.size(1)
        mask = torch.tril(torch.ones(T, T, device=hidden_states.device))
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # ← Save here, before dropout drops values
        store['weights'] = attn_weights.detach().cpu()

        attn_weights_dropped = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights_dropped, v)

        # Merge heads back
        B, H, T, D = out.size()
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out, None

    attn_module.forward = types.MethodType(patched_forward, attn_module)

def get_attention_weights(model, tokenizer, prompt, layer_idx, head_idx):
    """
    Extract attention weights for a specific (layer, head) using forward patching.

    Returns:
        tokens      — list of token strings
        attn_head   — numpy array (seq_len, seq_len)
                      attn_head[i, j] = how much token i attends to token j
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    store = {}
    attn_module = model.transformer.h[layer_idx].attn

    # Save original forward so we can restore it after
    original_forward = attn_module.forward
    make_patched_forward(attn_module, store)

    with torch.no_grad():
        model(**inputs)

    # Restore original forward
    attn_module.forward = original_forward

    if 'weights' not in store:
        raise ValueError(f"Attention weights not captured for layer {layer_idx}")

    # store['weights'] shape: (1, num_heads, seq_len, seq_len)
    attn_head = store['weights'][0, head_idx].numpy()
    return tokens, attn_head

# ── Analysis 1: First-token attention ─────────────────────────────────────────
def run_first_token_analysis(model, tokenizer):
    """
    For each top head, measure the average attention weight on the FIRST token
    across all query positions.

    The first token carries the strongest language identity signal:
      "Le" → French | "Das" → German | "El" → Spanish | "The" → English

    From debug: L6H1 puts 0.6-0.96 weight on token[0] at every position.
    This is the mechanistic explanation for why ablating it causes language switching.
    """
    print("\n=== FIRST-TOKEN ATTENTION ANALYSIS ===")
    print("Hypothesis: language-critical heads attend heavily to first token")
    print("(first token = strongest language identity cue)\n")

    results = []
    for layer, head in TOP_HEADS:
        head_id = f"L{layer}H{head}"
        for prompt, lang, condition in VIZ_PROMPTS:
            tokens, attn = get_attention_weights(model, tokenizer, prompt, layer, head)

            # Column 0 = attention TO first token, averaged over all query rows
            first_token_attn = float(attn[:, 0].mean())
            last_token_attn  = float(attn[:, -1].mean())

            # Entropy: low = focused, high = diffuse
            flat    = attn.mean(axis=0)
            flat    = flat / (flat.sum() + 1e-9)
            entropy = float(-np.sum(flat * np.log(flat + 1e-9)))

            results.append({
                "head_id":           head_id,
                "layer":             layer,
                "head":              head,
                "prompt":            prompt[:35],
                "lang":              lang,
                "condition":         condition,
                "first_token_attn":  first_token_attn,
                "last_token_attn":   last_token_attn,
                "entropy":           entropy,
                "n_tokens":          len(tokens),
            })
            print(f"{head_id} | {lang} | {condition:8s} | "
                  f"first_tok={first_token_attn:.3f} | entropy={entropy:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/attention_stats.csv", index=False)

    print("\n--- Mean first-token attention by head (descending) ---")
    print(df.groupby("head_id")["first_token_attn"].mean()
            .sort_values(ascending=False).to_string())

    print("\n--- L6H1: correct vs confused ---")
    l6h1 = df[df["head_id"] == "L6H1"]
    print(l6h1.groupby("condition")[["first_token_attn", "entropy"]].mean().to_string())

    return df

# ── Analysis 2: Side-by-side heatmaps (main paper figure) ────────────────────
def plot_comparative_heatmaps(model, tokenizer):
    """
    L6H1 on a correct (French) vs confused (Italian) prompt side by side.
    Visually shows that on the correct prompt every row attends heavily
    to 'Le' (col 0), while on the confused prompt attention is more scattered.
    """
    print("\n=== COMPARATIVE HEATMAPS: L6H1 correct vs confused ===")

    correct_prompt  = "Le temps aujourd'hui est très"
    confused_prompt = "La cosa più importante nella vita è"

    tokens_c, attn_c = get_attention_weights(model, tokenizer, correct_prompt,  6, 1)
    tokens_x, attn_x = get_attention_weights(model, tokenizer, confused_prompt, 6, 1)

    vmax = max(attn_c.max(), attn_x.max())
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, tokens, attn, title in [
        (axes[0], tokens_c, attn_c,
         f'L6H1 — French (language-correct)\n"{correct_prompt}"'),
        (axes[1], tokens_x, attn_x,
         f'L6H1 — Italian (language-confused)\n"{confused_prompt}"'),
    ]:
        n  = len(tokens)
        im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel("Key (attended to)", fontsize=9)
        ax.set_ylabel("Query (attending from)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Highlight first column (first-token attention)
        for row in range(n):
            val = attn[row, 0]
            if val > 0.1:
                ax.text(0, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "black",
                        fontweight="bold")

    plt.suptitle(
        "L6H1 Attention Patterns: Correct vs. Confused Generation\n"
        "Note strong first-token (col 0) attention on correct French prompt",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = f"{FIGS_DIR}/fig_attention_comparison.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    plt.close()

# ── Analysis 3: First-token bar chart ─────────────────────────────────────────
def plot_first_token_bar(df):
    """
    Bar chart: mean first-token attention per head, split by correct vs confused.
    Shows whether the language-critical heads consistently attend to first token
    more on prompts the model gets right.
    """
    heads          = list(df["head_id"].unique())
    correct_means  = [df[(df["head_id"] == h) & (df["condition"] == "correct" )]["first_token_attn"].mean() for h in heads]
    confused_means = [df[(df["head_id"] == h) & (df["condition"] == "confused")]["first_token_attn"].mean() for h in heads]

    x     = np.arange(len(heads))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width/2, correct_means,  width, label="Language-correct",  color="#2980b9", alpha=0.85)
    ax.bar(x + width/2, confused_means, width, label="Language-confused", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(heads)
    ax.set_ylabel("Mean Attention to First Token")
    ax.set_title("First-Token Attention by Head: Correct vs. Confused Prompts", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = f"{FIGS_DIR}/fig_first_token_bar.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    plt.close()

# ── Analysis 4: Per-language heatmaps for L6H1 ───────────────────────────────
def plot_per_language_heatmaps(model, tokenizer):
    """
    One heatmap per language for L6H1 on correct prompts.
    Lets you see visually whether first-token attention strength varies
    across languages — supports or challenges the specialization claim.
    """
    print("\n=== PER-LANGUAGE HEATMAPS (L6H1) ===")
    for prompt, lang, condition in VIZ_PROMPTS[:5]:
        tokens, attn = get_attention_weights(model, tokenizer, prompt, 6, 1)
        n   = len(tokens)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n * 0.8)))
        im  = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("Key (attended to)")
        ax.set_ylabel("Query (attending from)")
        ax.set_title(f"L6H1 | {lang.upper()} | \"{prompt}\"", fontsize=10)
        plt.colorbar(im, ax=ax, label="Attention weight")
        plt.tight_layout()
        p = f"{FIGS_DIR}/attn_L6H1_{lang}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved: {p}")

# ── Analysis 5: First-token attention across all 5 languages for L6H1 ────────
def plot_first_token_by_language(model, tokenizer):
    """
    Bar chart: mean first-token attention for L6H1 broken down by language.
    If L6H1 attends more strongly to the first token for some languages than
    others, that helps explain per-language accuracy differences.
    """
    print("\n=== FIRST-TOKEN ATTENTION BY LANGUAGE (L6H1) ===")
    correct_prompts = [(p, l) for p, l, c in VIZ_PROMPTS if c == "correct"]

    lang_attn = {}
    for prompt, lang in correct_prompts:
        tokens, attn = get_attention_weights(model, tokenizer, prompt, 6, 1)
        lang_attn[lang] = float(attn[:, 0].mean())
        print(f"  {lang}: first_tok_attn = {lang_attn[lang]:.3f}")

    langs  = list(lang_attn.keys())
    values = [lang_attn[l] for l in langs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(langs, values, color="#2980b9", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Mean Attention to First Token")
    ax.set_title("L6H1 First-Token Attention Strength by Language", fontsize=11)
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = f"{FIGS_DIR}/fig_first_token_by_language.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_model()

    stats_df = run_first_token_analysis(model, tokenizer)
    plot_comparative_heatmaps(model, tokenizer)
    plot_first_token_bar(stats_df)
    plot_per_language_heatmaps(model, tokenizer)
    plot_first_token_by_language(model, tokenizer)

    print(f"\n✓ All done. Figures in {FIGS_DIR}/")
    print("Next: python phi3_experiment.py")
