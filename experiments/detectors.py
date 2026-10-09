"""Detector check and bootstrap CIs for the instruct heads on the 2,500 FLORES prompts: the continuations relabeled
with langid and fastText (lid.176) and a 2-of-3 vote, c->w per head under each, and a bootstrap CI for the head's
c->w."""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from analyze import same, spearman
from robustness import init, relabel, vote

RUNS = ("qwen-instruct-full:L22H6,qwen2.5-3b-instruct:L27H13,qwen3-1.7b-instruct:L18H12,gemma3-1b-instruct:L11H3,"
        "gemma3-4b-instruct:L24H0,olmo2-1b-instruct:L12H8")


def flips(base, labels, expected):
    b = np.array([same(x, e) for x, e in zip(base, expected)])
    o = np.array([same(x, e) for x, e in zip(labels, expected)])
    return b & ~o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=RUNS)
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--out", default="out/detectors")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {}
    lines = ["# Detector check and CIs for the instruct heads", "",
             "The 2,500-prompt continuations of each model relabeled with langid and fastText (lid.176); vote = at "
             "least two of langdetect, langid and fastText agree, otherwise unknown. c->w with a bootstrap 95% CI over "
             "prompts; rank and the largest other head are among the heads run on 2,500 prompts; Spearman compares "
             "c->w per head with langdetect's.", "",
             "| model | head | detector | baseline non-English retention | c->w [95% CI] | rank | largest other head | "
             "Spearman vs langdetect |", "|---|---|---|---|---|---|---|---|"]
    for spec in a.runs.split(","):
        m, head = spec.split(":")
        expected = [r["language"] for r in csv.DictReader(open(f"out/{m}/prompts.csv", encoding="utf-8"))]
        ld = json.load(open(f"out/{m}/labels.json"))
        runs = [json.loads(line) for line in open(f"out/{m}/gens.jsonl", encoding="utf-8")]
        uniq = sorted({t for r in runs for t in r["texts"]})
        with Pool(a.procs, initializer=init, initargs=(a.lid,)) as pool:
            other = dict(zip(uniq, pool.map(relabel, uniq, chunksize=500)))
        det = {d: {} for d in ("langdetect", "langid", "fasttext", "vote")}
        for r in runs:
            li, fx = zip(*(other[t] for t in r["texts"]))
            det["langdetect"][r["cond"]] = ld[r["cond"]]["labels"]
            det["langid"][r["cond"]], det["fasttext"][r["cond"]] = list(li), list(fx)
            det["vote"][r["cond"]] = [vote(*x) for x in zip(ld[r["cond"]]["labels"], li, fx)]
        heads = [c for c in ld if c.startswith("head:")]
        target = f"head:{head}"
        non_en = [i for i, e in enumerate(expected) if e != "en"]
        ref = None
        res[m] = {}
        for d, labs in det.items():
            f = {h: flips(labs["base"], labs[h], expected) for h in heads}
            rate = {h: float(x.mean()) for h, x in f.items()}
            boots = rng.choice(f[target].astype(float), (2000, len(expected))).mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            order = sorted(rate, key=rate.get, reverse=True)
            best_other = max((h for h in heads if h != target), key=rate.get)
            ret = float(np.mean([same(labs["base"][i], expected[i]) for i in non_en]))
            vals = [rate[h] for h in heads]
            ref = ref or vals
            rho = spearman(ref, vals)
            res[m][d] = {"head": head, "c2w": rate[target], "ci": [float(lo), float(hi)],
                         "rank": order.index(target) + 1, "heads": len(heads), "largest_other": best_other[5:],
                         "largest_other_c2w": rate[best_other], "retention": ret, "spearman_vs_langdetect": rho}
            lines.append(f"| {m} | {head} | {d} | {ret:.3f} | {rate[target]:.3f} [{lo:.3f}, {hi:.3f}] | "
                         f"{order.index(target) + 1} of {len(heads)} | {best_other[5:]} {rate[best_other]:.3f} | "
                         f"{rho:.3f} |")
        print(m, "done", flush=True)
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
