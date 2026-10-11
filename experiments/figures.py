import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import ci, same

# figures are drawn at the ACL column width (7.7 cm), so included at \columnwidth their text prints at 8 to 9 pt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "figure.constrained_layout.use": True})
OUT = Path("figures")
WIDTH = 3.0


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def heatmap(table, key, label, vmax, name, max_dnll=None):
    m = np.zeros((12, 12))
    for h, v in table.items():
        layer, head = map(int, h[1:].split("H"))
        m[layer, head] = v["full"][key]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=vmax)
    if max_dnll is not None:
        for h, v in table.items():
            if v["dnll"] > max_dnll:
                layer, head = map(int, h[1:].split("H"))
                ax.plot(head, layer, "x", color="black", ms=4, mew=0.8)
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label=label)
    save(fig, name)


def pair_heatmap(runs, titles, name, vmax=0.5):
    tables = [json.load(open(f"results/{r}/summary.json"))["modes"]["head"]["table"] for r in runs]
    heads = [tuple(map(int, h[1:].split("H"))) for h in tables[0]]
    shape = (max(l for l, _ in heads) + 1, max(h for _, h in heads) + 1)
    fig, axes = plt.subplots(1, len(runs), figsize=(WIDTH, 3.2), sharey=True)
    for ax, table, title in zip(axes, tables, titles):
        m = np.zeros(shape)
        for h, v in table.items():
            layer, head = map(int, h[1:].split("H"))
            m[layer, head] = v["full"]["c2w"]
        im = ax.imshow(m, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        layer, head = np.unravel_index(m.argmax(), m.shape)
        ax.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, lw=1, ec="black"))
        ax.annotate(f"L{layer}H{head}  {m[layer, head]:.2f}", (head, layer), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(range(0, shape[1], 3))
        ax.set_xlabel("Head")
    axes[0].set_yticks(range(0, shape[0], 3))
    axes[0].set_ylabel("Layer")
    fig.colorbar(im, ax=axes, label="Correct→Wrong Rate", shrink=0.8)
    save(fig, name)


def curves(name):
    expected = [r["language"] for r in csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8"))]
    lab = {**json.load(open("results/gpt2-multi/labels.json")),
           **json.load(open("results/gpt2-multi-lowloss/labels.json"))}
    base_nll = mean(lab["base"]["nll"].values())

    def point(cond):
        correct = [int(same(x, e)) for x, e in zip(lab[cond]["labels"], expected) if e != "en"]
        return mean(correct), ci(correct), mean(lab[cond]["nll"].values()) - base_nll

    ks = list(range(11))
    series = {o: [point("base")] + [point(f"{o}:k{k}") for k in ks[1:]]
              for o in ("c2w-lowloss", "sr", "random0", "random1", "random2")}
    fig, (top, bot) = plt.subplots(2, 1, figsize=(WIDTH, 3.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    styles = {"c2w-lowloss": ("#2a7fa0", "by c2w, ΔNLL ≤ 0.1"), "sr": ("#d9822b", "by switch rate")}
    for o, (color, label) in styles.items():
        acc = [p[0] for p in series[o]]
        top.plot(ks, acc, "o-", color=color, ms=3.5, lw=1.5, label=label)
        top.fill_between(ks, [p[1][0] for p in series[o]], [p[1][1] for p in series[o]], color=color, alpha=0.15, lw=0)
        bot.plot(ks, [p[2] for p in series[o]], "o-", color=color, ms=3.5, lw=1.5)
    rand = np.array([[p[0] for p in series[f"random{s}"]] for s in range(3)])
    top.plot(ks, rand.mean(0), "--", color="gray", lw=1.2, label="random order (3)")
    top.fill_between(ks, rand.min(0), rand.max(0), color="gray", alpha=0.15, lw=0)
    bot.plot(ks, np.mean([[p[2] for p in series[f"random{s}"]] for s in range(3)], 0), "--", color="gray", lw=1.2)
    top.set_ylim(0, 1.1)
    top.set_ylabel("Accuracy, non-English")
    fig.legend(loc="outside upper center", ncol=2, handlelength=1.6, columnspacing=0.8)
    bot.set_ylabel("ΔNLL")
    bot.set_xlabel("Heads ablated")
    bot.set_xticks(ks)
    save(fig, name)


def swap_shares(name):
    # share of non-English replies that move into the swapped-in language when the head's output is replaced by that
    # language's mean (EXPERIMENTS.md sections 10 and 11); values are labeled only where the swap mostly fails
    models = [("Qwen2.5-1.5B", "qwen-instruct", "L22H6"), ("Qwen2.5-3B", "qwen2.5-3b-instruct", "L27H13"),
              ("Qwen2.5-7B", "qwen2.5-7b-instruct", "L19H1"), ("Qwen3-1.7B", "qwen3-1.7b-instruct", "L18H12"),
              ("Gemma-3-1B", "gemma3-1b-instruct", "L11H3"), ("Gemma-3-4B", "gemma3-4b-instruct", "L24H0"),
              ("OLMo-2-1B", "olmo2-1b-instruct", "L12H8")]
    flores, lcb = [], []
    for _, run, head in models:
        row = next(line for line in open(f"results/{run}-diag/summary.md", encoding="utf-8")
                   if line.startswith("| other-language mean |"))
        flores.append(float(row.strip(" |\n").split("|")[-1]))
        lcb.append(json.load(open(f"results/{run}-steer/summary.json"))["rows"][f"{head} swap"]["monolingual_swap"])
    y = np.arange(len(models))[::-1]
    h, gap = 0.36, 0.02
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    ax.barh(y + h / 2 + gap, flores, height=h, color="#2a78d6", label="FLORES")
    ax.barh(y - h / 2 - gap, lcb, height=h, color="#eb6834", label="LCB, monolingual")
    for yy, vals in zip(y, zip(flores, lcb)):
        for v, off in zip(vals, (h / 2 + gap, -h / 2 - gap)):
            if v < 0.5:
                ax.text(v + 0.015, yy + off, f"{v:.2f}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _, _ in models])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Share in the swapped-in language")
    ax.xaxis.grid(True, color="#e3e2de", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="outside upper center", ncol=2, handlelength=1.2, columnspacing=1.0, frameon=False)
    save(fig, name)


def screen_grid(name, key="c2w", label="c→w"):
    # every head of the twelve instruct models on the 125-prompt FLORES screen (key: c2w, w2c or sr). Each panel has
    # its own scale, its largest value or 0.1 if that is smaller; gray heads fail the head rule's checks (dNLL above 1,
    # or fewer than 0.9 of the English continuations kept in English). Square: the head of tab:models; circle: a head
    # found on the crosslingual LCB screen instead
    models = [
        ("Qwen2.5-1.5B", "qwen-instruct", "L22H6"),
        ("Qwen2.5-3B", "qwen2.5-3b-instruct-screen", "L27H13"),
        ("Qwen2.5-7B", "qwen2.5-7b-instruct-screen", "L19H1"),
        ("Qwen3-1.7B", "qwen3-1.7b-instruct-screen", "L18H12"),
        ("Qwen3-4B", "qwen3-4b-instruct-screen", "L24H27"),
        ("Gemma-3-1B", "gemma3-1b-instruct-screen", "L11H3"),
        ("Gemma-3-4B", "gemma3-4b-instruct-screen", "L24H0"),
        ("OLMo-2-1B", "olmo2-1b-instruct-screen", "L12H8"),
        ("OLMo-3-7B", "olmo3-7b-instruct-screen", "L14H25"),
        ("Llama-3.2-1B", "llama3.2-1b-instruct-screen", "L8H25"),
        ("Llama-3.2-3B", "llama3.2-3b-instruct-screen", "L13H19"),
        ("SmolLM3-3B", "smollm3-instruct-screen", None),
    ]
    from_lcb = {"Qwen3-4B", "Llama-3.2-1B", "Llama-3.2-3B"}
    expected, seen = [], {}
    for r in csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")):
        seen[r["language"]] = seen.get(r["language"], 0) + 1
        if seen[r["language"]] <= 25:  # the screen's prompts, as sweep.py picks them
            expected.append(r["language"])
    en = [i for i, e in enumerate(expected) if e == "en"]
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#d4d4d4")
    fig, axes = plt.subplots(3, 4, figsize=(2 * WIDTH + 0.3, 5.2))
    for i, (ax, (model, run, head)) in enumerate(zip(axes.flat, models)):
        table = json.load(open(f"results/{run}/summary.json"))["modes"]["head"]["table"]
        labels = json.load(open(f"results/{run}/labels.json"))
        heads = {h: tuple(map(int, h[1:].split("H"))) for h in table}
        m = np.zeros((max(l for l, _ in heads.values()) + 1, max(k for _, k in heads.values()) + 1))
        bad = np.zeros(m.shape, bool)
        for h, (layer, k) in heads.items():
            m[layer, k] = table[h]["full"][key]
            kept = mean(same(labels[f"head:{h}"]["labels"][j], "en") for j in en)
            bad[layer, k] = table[h]["dnll"] > 1 or kept < 0.9
        top = m[~bad].max()
        im = ax.imshow(np.ma.masked_array(m / max(top, 0.1), bad), cmap=cmap, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        if head:
            layer, k = heads[head]
            if model in from_lcb:
                ax.plot(k, layer, "o", mfc="none", mec="black", ms=5, mew=0.9)
            else:
                ax.add_patch(plt.Rectangle((k - 0.5, layer - 0.5), 1, 1, fill=False, lw=0.9, ec="black"))
        ax.set_title(f"{model}, max {top:.3f}", fontsize=7)
        ax.set_xticks([0, m.shape[1] - 1])
        ax.set_yticks([0, m.shape[0] - 1])
        ax.tick_params(labelsize=6, length=2)
        if i % 4 == 0:
            ax.set_ylabel("Layer", fontsize=7)
        if i >= 8:
            ax.set_xlabel("Head", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.5, label=f"{label}, share of the panel's scale")
    save(fig, name)


def main():
    table = json.load(open("results/gpt2/summary.json"))["modes"]["head"]["table"]
    heatmap(table, "sr", "Language Switch Rate", 0.6, "fig1_ablation_heatmap")
    heatmap(table, "c2w", "Correct→Wrong Rate", 0.25, "fig1_c2w_heatmap", max_dnll=0.1)
    curves("fig3_accuracy_curve")
    pair_heatmap(["qwen-instruct-full", "qwen-base-full"], ["Instruct", "Base"], "fig1_qwen_c2w")
    swap_shares("fig_swap")
    screen_grid("fig_screen_grid")


if __name__ == "__main__":
    main()
