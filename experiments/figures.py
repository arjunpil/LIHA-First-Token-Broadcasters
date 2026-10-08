import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import ci, same

plt.rcParams.update({"font.family": "serif", "font.size": 9})
OUT = Path("figures")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def heatmap(table, key, label, vmax, name, max_dnll=None):
    m = np.zeros((12, 12))
    for h, v in table.items():
        layer, head = map(int, h[1:].split("H"))
        m[layer, head] = v["full"][key]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
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
    shape = max(tuple(int(x) + 1 for x in h[1:].split("H")) for h in tables[0])
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4, 3.6), sharey=True)
    for ax, table, title in zip(axes, tables, titles):
        m = np.zeros(shape)
        for h, v in table.items():
            layer, head = map(int, h[1:].split("H"))
            m[layer, head] = v["full"]["c2w"]
        im = ax.imshow(m, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        layer, head = np.unravel_index(m.argmax(), m.shape)
        ax.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, lw=1, ec="black"))
        ax.annotate(f"L{layer}H{head}  {m[layer, head]:.2f}", (head, layer), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9)
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
    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
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
    top.legend(fontsize=6.5, loc="upper center", ncol=2, handlelength=1.6, columnspacing=0.8)
    bot.set_ylabel("ΔNLL")
    bot.set_xlabel("Heads ablated")
    bot.set_xticks(ks)
    save(fig, name)


def main():
    table = json.load(open("results/gpt2/summary.json"))["modes"]["head"]["table"]
    heatmap(table, "sr", "Language Switch Rate", 0.6, "fig1_ablation_heatmap")
    heatmap(table, "c2w", "Correct→Wrong Rate", 0.25, "fig1_c2w_heatmap", max_dnll=0.1)
    curves("fig3_accuracy_curve")
    pair_heatmap(["qwen-instruct-full", "qwen-base-full"], ["Instruct", "Base"], "fig1_qwen_c2w")


if __name__ == "__main__":
    main()
