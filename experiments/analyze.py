import csv
import json
import random
import sys
from statistics import mean, pstdev

REFERENCE = {
    "gpt2": {"L6H1": 0.32, "L0H4": 0.28, "L3H1": 0.28, "L9H9": 0.28, "L7H3": 0.24},
    "qwen-instruct": {"L0H5": 0.224, "L0H11": 0.144, "L1H9": 0.136, "L0H7": 0.120, "L1H7": 0.112},
    "qwen-base": {"L0H0": 0.016},
    "bloom": {},
    "gpt2-zhru": {"L0H0": 0.40, "L4H5": 0.30, "L4H2": 0.30, "L0H1": 0.20, "L0H7": 0.20, "L1H10": 0.20,
                  "L6H1": 0.0, "L0H4": 0.0, "L3H1": 0.0, "L9H9": 0.0},
}


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


def ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def stats(base, abl, expected, idx):
    sw = [int(abl[i] != base[i]) for i in idx]
    cw = [int(same(base[i], expected[i]) and not same(abl[i], expected[i])) for i in idx]
    wc = [int(not same(base[i], expected[i]) and same(abl[i], expected[i])) for i in idx]
    return {"sr": mean(sw), "sr_ci": ci(sw), "c2w": mean(cw), "w2c": mean(wc),
            "acc": mean(int(same(abl[i], expected[i])) for i in idx)}


def main(model="gpt2"):
    labels, prompts, out, ref = f"out/{model}/labels.json", f"out/{model}/prompts.csv", f"out/{model}/summary", model
    PAPER = REFERENCE.get(ref, {})
    lab = json.load(open(labels))
    rows = list(csv.DictReader(open(prompts, encoding="utf-8")))
    expected = [r["language"] for r in rows]
    full = list(range(len(rows)))
    orig = [i for i, r in enumerate(rows) if r["source"] == "original"]
    base = lab["base"]["labels"]
    base_nll = mean(lab["base"]["nll"].values())
    res = {"baseline_acc_full": mean(int(same(b, e)) for b, e in zip(base, expected)),
           "baseline_acc_orig": mean(int(same(base[i], expected[i])) for i in orig), "modes": {}}
    for mode in ("head", "paper"):
        heads = {c.split(":")[1]: v for c, v in lab.items() if c.startswith(mode + ":")}
        if not heads:
            continue
        table = {}
        for h, v in heads.items():
            table[h] = {"full": stats(base, v["labels"], expected, full),
                        "orig": stats(base, v["labels"], expected, orig),
                        "dnll": mean(v["nll"].values()) - base_nll}
        srs = [t["full"]["sr"] for t in table.values()]
        res["modes"][mode] = {"table": table, "sr_mean": mean(srs), "sr_sd": pstdev(srs),
                              "orig_sr_mean": mean(t["orig"]["sr"] for t in table.values())}
    if {"head", "paper"} <= set(res["modes"]):
        hs = sorted(res["modes"]["head"]["table"])
        res["spearman_head_vs_paper_full"] = spearman([res["modes"]["head"]["table"][h]["full"]["sr"] for h in hs],
                                                      [res["modes"]["paper"]["table"][h]["full"]["sr"] for h in hs])
    json.dump(res, open(out + ".json", "w"), indent=1)

    lines = [f"baseline accuracy: {res['baseline_acc_full']:.3f} on {len(full)} prompts, "
             f"{res['baseline_acc_orig']:.3f} on the {len(orig)} original prompts", ""]
    if "paper" in res["modes"]:
        t = res["modes"]["paper"]["table"]
        key = "orig" if ref.startswith("gpt2") else "full"
        lines += [f"paper-style hook vs the paper's reported switch rates ({key} prompt set):"]
        lines += [f"  {h}: {t[h][key]['sr']:.3f} (paper {p:.3f})" for h, p in PAPER.items() if h in t]
        lines += [f"  population mean {res['modes']['paper']['orig_sr_mean' if ref.startswith('gpt2') else 'sr_mean']:.3f}", ""]
    for mode, m in res["modes"].items():
        t = m["table"]
        top = sorted(t, key=lambda h: -t[h]["full"]["sr"])[:10]
        lines += [f"{mode} hook, {len(full)} prompts: mean SR {m['sr_mean']:.3f} (sd {m['sr_sd']:.3f})",
                  "| head | SR [95% CI] | correct→wrong | wrong→correct | acc | ΔNLL | ΔNLL same-layer mean |",
                  "|---|---|---|---|---|---|---|"]
        for h in top:
            layer = h.split("H")[0]
            peers = [t[o]["dnll"] for o in t if o.split("H")[0] == layer and o != h]
            f = t[h]["full"]
            lines.append(f"| {h} | {f['sr']:.3f} [{f['sr_ci'][0]:.3f}, {f['sr_ci'][1]:.3f}] | {f['c2w']:.3f} | "
                         f"{f['w2c']:.3f} | {f['acc']:.3f} | {t[h]['dnll']:+.4f} | {mean(peers):+.4f} |")
        lines.append("")
        lines.append("paper's top heads under this hook: " + ", ".join(
            f"{h} SR {t[h]['full']['sr']:.3f} rank {sorted(t, key=lambda x: -t[x]['full']['sr']).index(h) + 1}"
            for h in PAPER if h in t))
        lines.append("")
    if "spearman_head_vs_paper_full" in res:
        lines.append(f"Spearman between head-hook and paper-hook SR over heads: {res['spearman_head_vs_paper_full']:.3f}")
    open(out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
