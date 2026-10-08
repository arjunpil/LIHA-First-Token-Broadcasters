import argparse
import csv
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import langid
import numpy as np

from analyze import same, spearman

ft = None


def init(lid):
    global ft
    import fasttext
    ft = fasttext.load_model(lid)


def relabel(text):
    if not text.strip():
        return "unknown", "unknown"
    label = ft.f.predict(text.replace("\n", " "), 1, 0.0, "strict")[0][1].replace("__label__", "")
    return langid.classify(text)[0], label


def vote(*labels):
    valid = [l for l in labels if l != "unknown"]
    top, n = Counter(valid).most_common(1)[0] if valid else ("unknown", 0)
    return top if n >= 2 else "unknown"


def c2w_matrix(labels, heads, expected):
    ok = lambda xs: np.array([same(x, e) for x, e in zip(xs, expected)])
    base = ok(labels["base"])
    return base, np.stack([base & ~ok(labels["head:" + h]) for h in heads])


def rank_of(scores, heads, head, keep):
    order = [heads[i] for i in np.argsort(-scores) if heads[i] in keep]
    return order.index(head) + 1, order[:5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", default="out/gpt2/gens.jsonl")
    p.add_argument("--labels", default="results/gpt2/labels.json")
    p.add_argument("--summary", default="results/gpt2/summary.json")
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--splits", type=int, default=200)
    p.add_argument("--head", default="L6H10")
    a = p.parse_args()

    expected = [r["language"] for r in csv.DictReader(open("prompts_european.csv", encoding="utf-8"))]
    table = json.load(open(a.summary))["modes"]["head"]["table"]
    heads = sorted(table)
    keep = {h for h in heads if table[h]["dnll"] <= 0.1}
    ld = json.load(open(a.labels))
    runs = [r for r in map(json.loads, open(a.gens, encoding="utf-8")) if r["cond"] == "base" or r["cond"].startswith("head:")]
    uniq = sorted({t for r in runs for t in r["texts"]})
    with Pool(4, initializer=init, initargs=(a.lid,)) as pool:
        other = dict(zip(uniq, pool.map(relabel, uniq, chunksize=500)))

    det = {d: {} for d in ("langdetect", "langid", "fasttext", "vote")}
    for r in runs:
        li, fx = zip(*(other[t] for t in r["texts"]))
        det["langdetect"][r["cond"]] = ld[r["cond"]]["labels"]
        det["langid"][r["cond"]], det["fasttext"][r["cond"]] = list(li), list(fx)
        det["vote"][r["cond"]] = [vote(*x) for x in zip(ld[r["cond"]]["labels"], li, fx)]

    mats = {d: c2w_matrix(labels, heads, expected) for d, labels in det.items()}
    ref = mats["langdetect"][1].mean(1)
    lines = ["# Detector and prompt-split checks for the GPT-2 head sweep", "",
             f"Same {len(runs) - 1} head-hook generations as results/gpt2, relabeled with langid and fastText (lid.176). "
             "vote = at least two of langdetect, langid and fastText agree, otherwise unknown. Ranks are among the "
             f"{len(keep)} heads with dNLL <= 0.1.", "",
             f"| detector | baseline acc | Spearman c->w vs langdetect | {a.head} c->w (rank) | top 5 c->w |",
             "|---|---|---|---|---|"]
    out = {}
    for d, (base, m) in mats.items():
        c2w = m.mean(1)
        rank, top = rank_of(c2w, heads, a.head, keep)
        rho = spearman(list(ref), list(c2w))
        out[d] = {"baseline_acc": float(base.mean()), "spearman_vs_langdetect": rho,
                  "c2w": dict(zip(heads, map(float, c2w)))}
        lines.append(f"| {d} | {base.mean():.3f} | {rho:.3f} | {c2w[heads.index(a.head)]:.3f} ({rank}) | {', '.join(top)} |")

    rng = random.Random(0)
    by_lang = {l: [i for i, e in enumerate(expected) if e == l] for l in sorted(set(expected))}
    lines += ["", f"Split halves: {a.splits} random splits, stratified by language, c->w per head on each half.", "",
              f"| detector | Spearman between halves, mean [5th pct] | {a.head} rank 1 in both halves | top-5 overlap, mean |",
              "|---|---|---|---|"]
    for d in ("langdetect", "vote"):
        m = mats[d][1]
        rhos, firsts, overlaps = [], 0, []
        for _ in range(a.splits):
            mask = np.zeros(len(expected), bool)
            for idx in by_lang.values():
                mask[rng.sample(idx, len(idx) // 2)] = True
            x, y = m[:, mask].mean(1), m[:, ~mask].mean(1)
            rhos.append(spearman(list(x), list(y)))
            (rx, tx), (ry, ty) = rank_of(x, heads, a.head, keep), rank_of(y, heads, a.head, keep)
            firsts += rx == 1 and ry == 1
            overlaps.append(len(set(tx) & set(ty)))
        out[d]["split"] = {"spearman_mean": float(np.mean(rhos)), "spearman_p5": float(np.percentile(rhos, 5)),
                           "head_first_both": firsts / a.splits, "top5_overlap": float(np.mean(overlaps))}
        lines.append(f"| {d} | {np.mean(rhos):.3f} [{np.percentile(rhos, 5):.3f}] | {firsts}/{a.splits} | "
                     f"{np.mean(overlaps):.1f} / 5 |")
    Path(a.summary).with_name("robustness.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(Path(a.summary).with_name("robustness.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
