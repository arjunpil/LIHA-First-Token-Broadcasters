import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from analyze import same
from checks import controls


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x):
    m, lo, hi = boot(x)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def reference(rows, enc, seed=0):
    blocks = {l: [r["prompt"] for r in rows if r["language"] == l and r["source"] == "flores200"]
              for l in ("en", "fr", "de", "es", "it")}
    n = min(map(len, blocks.values()))
    perm = np.random.default_rng(seed).permutation(n)
    e = {l: enc(v[:n]) for l, v in blocks.items()}
    out = {k: [] for k in ("same language, next sentence", "same language, random sentence",
                           "English, next sentence", "English, random sentence")}
    for l in ("fr", "de", "es", "it"):
        x, en = e[l], e["en"]
        out["same language, next sentence"] += list((x[:-1] * x[1:]).sum(1))
        out["same language, random sentence"] += list((x * x[perm]).sum(1))
        out["English, next sentence"] += list((x[:-1] * en[1:]).sum(1))
        out["English, random sentence"] += list((x * en[perm]).sum(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", default="out/gpt2/gens.jsonl")
    p.add_argument("--labels", default="results/gpt2/labels.json")
    p.add_argument("--summary", default="results/gpt2/summary.json")
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--head", default="L6H10")
    p.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.prompts, encoding="utf-8")))
    lab = json.load(open(a.labels))
    ctrl = controls(a.summary, [a.head])
    conds = ["base", f"head:{a.head}"] + [f"head:{h}" for h in ctrl]
    texts = {}
    for line in open(a.gens, encoding="utf-8"):
        r = json.loads(line)
        if r["cond"] in conds:
            texts[r["cond"]] = r["texts"]

    ne = [i for i, r in enumerate(rows) if r["language"] != "en"]
    model = SentenceTransformer(a.encoder, device="cpu")
    enc = lambda xs: model.encode(xs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    prompt_emb = enc([rows[i]["prompt"] for i in ne])
    emb = {c: enc([texts[c][i].strip() for i in ne]) for c in conds}
    sim = {c: (prompt_emb * emb[c]).sum(1) for c in conds}
    lang = {c: [lab[c]["labels"][i] for i in ne] for c in conds}
    exp = [rows[i]["language"] for i in ne]
    nonempty = {c: [bool(texts[c][i].strip()) for i in ne] for c in conds}
    kept = [k for k in range(len(ne)) if same(lang["base"][k], exp[k]) and nonempty["base"][k]]
    drift = [k for k in range(len(ne)) if lang["base"][k] == "en" and nonempty["base"][k]]

    lines = [f"# Does the content stay when {a.head} is removed? ({a.encoder})", "",
             f"Non-English prompts, cosine similarity between the prompt and the 40-token continuation with a "
             f"multilingual encoder. Controls: {', '.join(ctrl)}. Means with bootstrap 95% CIs.", "",
             "| group | n | prompt vs continuation |", "|---|---|---|",
             f"| baseline, continues in the prompt language | {len(kept)} | {fmt(sim['base'][kept])} |",
             f"| baseline, drifts to English | {len(drift)} | {fmt(sim['base'][drift])} |"]
    ref = reference(rows, enc)
    lines += [f"| reference: FLORES sentence vs {k} | {len(v)} | {fmt(v)} |" for k, v in ref.items()]
    res = {"controls": ctrl, "kept": boot(sim["base"][kept]), "drift": boot(sim["base"][drift]),
           "reference": {k: boot(v) for k, v in ref.items()}, "heads": {}}
    for c in conds[1:]:
        flip = [k for k in kept if lang[c][k] == "en" and nonempty[c][k]]
        stay = [k for k in kept if same(lang[c][k], exp[k]) and nonempty[c][k]]
        if len(flip) < 10:
            lines.append(f"| {c[5:]}: only {len(flip)} prompts flip to English | | |")
            continue
        cross_flip = (emb["base"][flip] * emb[c][flip]).sum(1)
        cross_stay = (emb["base"][stay] * emb[c][stay]).sum(1)
        lines += [f"| {c[5:]} flips to English, before | {len(flip)} | {fmt(sim['base'][flip])} |",
                  f"| {c[5:]} flips to English, after | {len(flip)} | {fmt(sim[c][flip])} |",
                  f"| {c[5:]}: before vs after continuation, flipped | {len(flip)} | {fmt(cross_flip)} |",
                  f"| {c[5:]}: before vs after continuation, stayed | {len(stay)} | {fmt(cross_stay)} |"]
        res["heads"][c[5:]] = {"n_flip": len(flip), "before": boot(sim["base"][flip]), "after": boot(sim[c][flip]),
                               "cross_flip": boot(cross_flip), "cross_stay": boot(cross_stay)}
    out = Path(a.out or f"results/gpt2-content/{a.encoder.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
