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


if __name__ == "__main__":
    main()
