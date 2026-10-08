"""
fix_fig2_xaxis.py
Regenerates figures/fig3_accuracy_curve.pdf with the correct x-axis (0-10).
Run from your project folder:
    py -3.12 fix_fig2_xaxis.py

Reads:  results/multi_ablation.csv
Saves:  figures/fig3_accuracy_curve.pdf
        figures/fig3_accuracy_curve.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(f"{RESULTS_DIR}/multi_ablation.csv")

# Keep only k=0..10
df = df[df["k"] <= 10].copy()

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
})

fig, ax = plt.subplots(figsize=(3.4, 2.6))

ax.plot(df["k"], df["accuracy"],
        color="#2E86AB", linewidth=2.0,
        marker="o", markersize=5, zorder=3,
        label="Language accuracy")

# Chance line
ax.axhline(y=0.2, color="#999999", linewidth=1.0,
           linestyle=":", zorder=1, label="Chance (0.20)")

ax.set_xlim(-0.3, 10.3)
ax.set_ylim(0.0, 1.05)
ax.set_xticks(range(0, 11))
ax.set_xlabel("Heads ablated (ranked by switch rate)", labelpad=6)
ax.set_ylabel("Language Accuracy")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
ax.legend(loc="upper right", framealpha=0.85, handlelength=1.4)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)

plt.tight_layout()

out_pdf = os.path.join(FIGURES_DIR, "fig3_accuracy_curve.pdf")
out_png = os.path.join(FIGURES_DIR, "fig3_accuracy_curve.png")
plt.savefig(out_pdf, bbox_inches="tight", format="pdf")
plt.savefig(out_png, bbox_inches="tight", dpi=200, format="png")
plt.close()

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done — upload figures/fig3_accuracy_curve.pdf to Overleaf.")
