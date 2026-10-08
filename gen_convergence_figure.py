"""
gen_convergence_figure.py
Generates figures/fig_convergence.pdf — the dual-axis convergence plot
showing mean ablation switch rate per layer overlaid with probing accuracy.

Run from your project folder:
    py -3.12 gen_convergence_figure.py

Reads from:
    results/gpt_2_probing.csv
    results/bloom_1b7_probing.csv
    results/ablation_sweep.csv
    results/bloom_ablation_sweep.csv

Saves:
    figures/fig_convergence.pdf
    figures/fig_convergence.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
gpt2_probe  = pd.read_csv(f"{RESULTS_DIR}/gpt_2_probing.csv")
bloom_probe = pd.read_csv(f"{RESULTS_DIR}/bloom_1b7_probing.csv")
gpt2_sweep  = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
bloom_sweep = pd.read_csv(f"{RESULTS_DIR}/bloom_ablation_sweep.csv")

# Mean switch rate per layer
gpt2_layer_sr  = gpt2_sweep.groupby("layer")["switch_rate"].mean()
bloom_layer_sr = bloom_sweep.groupby("layer")["switch_rate"].mean()

# Top causal head layers (threshold = population mean + 1 std)
gpt2_mean  = gpt2_sweep["switch_rate"].mean()
gpt2_std   = gpt2_sweep["switch_rate"].std()
bloom_mean = bloom_sweep["switch_rate"].mean()
bloom_std  = bloom_sweep["switch_rate"].std()

# Use mean + 2.0*std for both models — highlights only genuinely elevated layers
# GPT-2: layers 0, 6, 9  |  BLOOM: layers 12, 21
gpt2_top_layers  = sorted(gpt2_sweep[
    gpt2_sweep["switch_rate"] >= gpt2_mean + 2.0 * gpt2_std
]["layer"].unique())
bloom_top_layers = sorted(bloom_sweep[
    bloom_sweep["switch_rate"] >= bloom_mean + 2.0 * bloom_std
]["layer"].unique())

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2E86AB"   # bars  — ablation switch rate
ORANGE = "#E84855"   # line  — probing accuracy
MARKER_COLOR = "#FF8C00"

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
})

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 2.8))
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)

# ── Helper ────────────────────────────────────────────────────────────────────
def plot_panel(ax, layer_sr, probe_df, top_layers, title, x_max):
    layers      = probe_df["layer"].values
    probe_acc   = probe_df["accuracy"].values

    # ── Bar: mean switch rate per layer ───────────────────────────────────────
    bar_layers = layer_sr.index.values
    bar_vals   = layer_sr.values

    ax.bar(bar_layers, bar_vals, color=BLUE, alpha=0.75,
           width=0.6, zorder=2, label="Mean switch rate (ablation)")
    ax.set_ylim(0, max(bar_vals) * 2.2)
    ax.set_ylabel("Mean switch rate", color=BLUE, labelpad=4)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Layer")
    ax.set_xlim(-0.8, x_max + 0.8)
    ax.set_xticks(range(0, x_max + 1, 2 if x_max > 14 else 1))

    # ── Line: probing accuracy ─────────────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.plot(layers, probe_acc, color=ORANGE, linewidth=1.8,
             marker="o", markersize=3.5, zorder=3, label="Probing accuracy")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Probing accuracy", color=ORANGE, labelpad=4)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # ── Vertical markers for top causal head layers ────────────────────────────
    for i, tl in enumerate(top_layers):
        ax.axvline(x=tl, color=MARKER_COLOR, linewidth=1.8,
                   linestyle="--", alpha=1.0, zorder=5,
                   label="Top causal head layer" if i == 0 else None)

    ax.set_title(title, pad=5, fontweight="bold")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # ── Combined legend ────────────────────────────────────────────────────────
    bars,   bar_labels  = ax.get_legend_handles_labels()
    lines,  line_labels = ax2.get_legend_handles_labels()
    ax.legend(bars + lines, bar_labels + line_labels,
              loc="upper left", framealpha=0.85, fontsize=7.5,
              handlelength=1.4, borderpad=0.6)

    return ax, ax2

# ── GPT-2 panel ───────────────────────────────────────────────────────────────
ax1, ax1b = plot_panel(
    fig.add_subplot(gs[0]),
    gpt2_layer_sr,
    gpt2_probe,
    gpt2_top_layers,
    "GPT-2",
    x_max=12
)

# ── BLOOM panel ───────────────────────────────────────────────────────────────
ax2, ax2b = plot_panel(
    fig.add_subplot(gs[1]),
    bloom_layer_sr,
    bloom_probe,
    bloom_top_layers,
    "BLOOM-1b7",
    x_max=24
)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Convergence of Causal Ablation and Linear Probing by Layer",
    fontsize=10, fontweight="bold", y=1.01
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = os.path.join(FIGURES_DIR, "fig_convergence.pdf")
out_png = os.path.join(FIGURES_DIR, "fig_convergence.png")

plt.savefig(out_pdf, bbox_inches="tight", format="pdf")
plt.savefig(out_png, bbox_inches="tight", dpi=200, format="png")
plt.close()

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done — upload figures/fig_convergence.pdf to Overleaf.")