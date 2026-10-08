"""
probing_experiment.py — Train linear probes on hidden states to predict language.

For each layer in GPT-2 (and BLOOM), we extract the hidden state at the
last token position and train a logistic regression classifier to predict
the prompt's language. High accuracy = that layer encodes language identity
representationally.

Key question: Do the layers containing our causally identified heads
(layers 1, 3, 6, 7, 10 in GPT-2) also show the highest probing accuracy?
If yes, causal and representational evidence converge — strong support
for our claims. If no, that's an interesting dissociation worth discussing.

This adds a second, independent line of evidence to the paper.
Run time: ~15-20 minutes on GPU
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import os

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"
FIGS_DIR    = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

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

# ── GPT-2 probing ─────────────────────────────────────────────────────────────
def extract_gpt2_hidden_states(model, tokenizer):
    """
    For each prompt, extract hidden states at every layer at the last token.
    Returns: dict {layer_idx: np.array of shape (n_prompts, hidden_dim)}
    """
    print("Extracting GPT-2 hidden states...")
    model.eval()
    n_layers = model.config.n_layer  # 12 for GPT-2 small

    all_hidden = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer
    labels = []

    for prompt, lang in PARALLEL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (n_layers+1) tensors
        # Each: (batch, seq_len, hidden_dim)
        # We take the last token position
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            vec = hidden[0, -1, :].cpu().numpy()  # last token, all dims
            all_hidden[layer_idx].append(vec)

        labels.append(lang)

    # Stack into arrays
    hidden_arrays = {
        i: np.stack(all_hidden[i]) for i in range(n_layers + 1)
    }
    return hidden_arrays, labels

def probe_layers(hidden_arrays, labels, model_name):
    """
    Train a logistic regression probe at each layer and measure accuracy
    via 5-fold cross-validation.
    """
    print(f"\nProbing {model_name} layers...")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    records = []
    n_layers = len(hidden_arrays)

    for layer_idx in range(n_layers):
        X = hidden_arrays[layer_idx]

        # Logistic regression with L2 regularization
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            solver="lbfgs",
        )

        # 5-fold CV — with only 25 samples use leave-one-out style
        # Use 5-fold since 25/5 = 5 samples per fold
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc  = scores.std()

        records.append({
            "layer":    layer_idx,
            "accuracy": mean_acc,
            "std":      std_acc,
            "layer_name": f"L{layer_idx}" if layer_idx > 0 else "Embed",
        })

        print(f"  Layer {layer_idx:2d}: acc={mean_acc:.3f} ± {std_acc:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/{model_name.lower().replace('-','_')}_probing.csv", index=False)
    return df

def plot_probing_results(gpt2_df, bloom_df=None):
    """
    Plot probing accuracy by layer for GPT-2 (and BLOOM if available).
    Overlay markers at layers containing our top causal heads.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size":   11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2 if bloom_df is not None else 1,
                              figsize=(14 if bloom_df is not None else 7, 5))

    if bloom_df is None:
        axes = [axes]

    # GPT-2
    ax = axes[0]
    layers = gpt2_df["layer"].values
    accs   = gpt2_df["accuracy"].values
    stds   = gpt2_df["std"].values

    ax.plot(layers, accs, "o-", color="#2980b9", linewidth=2, markersize=5, label="Probing accuracy")
    ax.fill_between(layers, accs - stds, accs + stds, alpha=0.15, color="#2980b9")

    # Mark layers with top causal heads
    causal_layers = [1, 3, 6, 7, 10]
    for cl in causal_layers:
        if cl < len(accs):
            ax.axvline(cl, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)

    ax.axvline(causal_layers[0], color="#e74c3c", linestyle="--", alpha=0.5,
               linewidth=1, label="Layers with top causal heads")
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1, label="Chance (5 langs)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probing Accuracy (5-fold CV)")
    ax.set_title("GPT-2: Language Identity Probing\nby Layer", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # BLOOM
    if bloom_df is not None:
        ax2 = axes[1]
        layers2 = bloom_df["layer"].values
        accs2   = bloom_df["accuracy"].values
        stds2   = bloom_df["std"].values

        ax2.plot(layers2, accs2, "o-", color="#27ae60", linewidth=2,
                 markersize=5, label="Probing accuracy")
        ax2.fill_between(layers2, accs2 - stds2, accs2 + stds2,
                         alpha=0.15, color="#27ae60")

        bloom_causal = [6, 9, 15, 18]
        for cl in bloom_causal:
            if cl < len(accs2):
                ax2.axvline(cl, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
        ax2.axvline(bloom_causal[0], color="#e74c3c", linestyle="--", alpha=0.5,
                    linewidth=1, label="Layers with top causal heads")
        ax2.axhline(0.2, color="gray", linestyle=":", linewidth=1, label="Chance (5 langs)")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Probing Accuracy (5-fold CV)")
        ax2.set_title("BLOOM-1b7: Language Identity Probing\nby Layer", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)

    plt.suptitle(
        "Linear Probe Accuracy for Language Identity Prediction by Layer\n"
        "Red dashed lines mark layers containing top causal heads from ablation sweep",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = f"{FIGS_DIR}/fig_probing.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.close()

def print_convergence_analysis(gpt2_df):
    """
    Key analysis: do layers with top causal heads also have high probing accuracy?
    """
    print("\n=== CONVERGENCE ANALYSIS ===")
    print("Do causally identified layers also show high probing accuracy?\n")

    causal_layers    = [1, 3, 6, 7, 10]
    non_causal       = [l for l in range(13) if l not in causal_layers]

    causal_acc    = gpt2_df[gpt2_df["layer"].isin(causal_layers)]["accuracy"].mean()
    non_causal_acc = gpt2_df[gpt2_df["layer"].isin(non_causal)]["accuracy"].mean()

    print(f"Mean probing accuracy in causal layers {causal_layers}: {causal_acc:.3f}")
    print(f"Mean probing accuracy in other layers:                  {non_causal_acc:.3f}")
    print(f"Difference: {causal_acc - non_causal_acc:+.3f}")

    if causal_acc > non_causal_acc:
        print("\n→ CONVERGENT EVIDENCE: Causally identified layers also show")
        print("  higher representational language identity. Both causal")
        print("  ablation and linear probing point to the same layers.")
    else:
        print("\n→ DISSOCIATION: Causally identified layers do NOT have the")
        print("  highest probing accuracy. This is an interesting finding:")
        print("  the heads that cause language switching may not be the ones")
        print("  that most strongly represent language identity.")

    peak_layer = gpt2_df.loc[gpt2_df["accuracy"].idxmax(), "layer"]
    peak_acc   = gpt2_df["accuracy"].max()
    print(f"\nPeak probing accuracy: Layer {peak_layer} ({peak_acc:.3f})")
    print(f"Is peak layer a causal layer? {peak_layer in causal_layers}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== PROBING EXPERIMENT ===")
    print(f"Device: {DEVICE}\n")

    # GPT-2
    print("Loading GPT-2...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    gpt2_model.eval()

    gpt2_hidden, gpt2_labels = extract_gpt2_hidden_states(gpt2_model, gpt2_tokenizer)
    gpt2_probe_df = probe_layers(gpt2_hidden, gpt2_labels, "GPT-2")

    # Free GPT-2 memory
    del gpt2_model
    torch.cuda.empty_cache()

    # BLOOM
    bloom_probe_df = None
    try:
        print("\nLoading BLOOM-1b7...")
        bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
        bloom_model = AutoModelForCausalLM.from_pretrained(
            "bigscience/bloom-1b7",
            dtype=torch.float16,
            device_map="auto",
        )
        bloom_model.eval()

        # BLOOM hidden states
        print("Extracting BLOOM hidden states...")
        bloom_hidden = {i: [] for i in range(bloom_model.config.n_layer + 1)}
        bloom_labels = []

        for prompt, lang in PARALLEL_PROMPTS:
            inputs = bloom_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = bloom_model(**inputs, output_hidden_states=True)
            for layer_idx, hidden in enumerate(outputs.hidden_states):
                vec = hidden[0, -1, :].cpu().float().numpy()
                bloom_hidden[layer_idx].append(vec)
            bloom_labels.append(lang)

        bloom_hidden_arrays = {
            i: np.stack(bloom_hidden[i])
            for i in range(bloom_model.config.n_layer + 1)
        }
        bloom_probe_df = probe_layers(bloom_hidden_arrays, bloom_labels, "BLOOM-1b7")

        del bloom_model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"BLOOM probing failed: {e}")
        print("Continuing with GPT-2 results only")

    # Analysis and figures
    print_convergence_analysis(gpt2_probe_df)
    plot_probing_results(gpt2_probe_df, bloom_probe_df)

    print("\n✓ Probing experiment complete.")
    print(f"Results saved to {RESULTS_DIR}/*_probing.csv")
    print(f"Figure saved to {FIGS_DIR}/fig_probing.pdf")