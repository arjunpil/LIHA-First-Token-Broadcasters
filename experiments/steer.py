"""Steer one head on LCB with its per-language mean output on FLORES (the means of the diagnosis run). steer: the
mean of the language the reply should be in; swap: the mean of another language (diagnose.SWAP). With --add, the
head also gets the difference between the language mean and its mean over all languages added instead of replaced."""
import argparse
import csv
import json
import random
from pathlib import Path

import fasttext
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose import LANGS, SWAP, head_stats
from followup import patched
from lcb import TASKS, batches, load_lcb, lpr, paired_ci, repetition, score
from multi import parse
from sweep import MODELS, as_prompts, generate, load_kwargs


def language_means(model, tok, key, heads, dh, bs):
    # the same tokens as the diagnosis: the user's text and the baseline continuation, template left out
    d = Path(f"out/{key}-diag")
    rows = list(csv.DictReader(open(d / "prompts.csv", encoding="utf-8")))
    base = json.loads(open(d / "gens.jsonl", encoding="utf-8").readline())
    assert base["cond"] == "base"
    prompts = as_prompts(tok, key, rows)
    means = {}
    for l, h in heads:
        _, mu = head_stats(model, tok, key, rows, prompts, base["texts"], l, h, dh, bs)
        means[(l, h)] = {g: mu[("text", g)].float() for g in LANGS} | {"all": mu["text"].float()}
    return means


def controls_of(key, head, H, n, seed):
    # the control heads of the earlier LCB run, so the comparison is with the same heads
    s = Path(f"out/{key}-lcb/summary.json")
    if s.exists():
        c = [parse(x) for x in json.load(open(s))["controls"] if parse(x)[0] == head[0]]
        if c:
            return c
    pool = [h for h in range(H) if (head[0], h) != head]
    return [(head[0], h) for h in sorted(random.Random(seed).sample(pool, min(n, len(pool))))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=[m for m in MODELS if m.endswith("instruct")])
    p.add_argument("--head", required=True)
    p.add_argument("--controls", type=int, default=3)
    p.add_argument("--add", action="store_true", help="also add the language direction instead of replacing")
    p.add_argument("--langs", default="fr,de,es,it,en")
    p.add_argument("--lcb", default="lcb/test_sets")
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--bs", type=int, default=None, help="default: the earlier LCB run's")
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--mean-bs", type=int, default=250)
    p.add_argument("--dtype", default=None)
    p.add_argument("--limit", type=int, default=None, help="prompts per task, for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or f"out/{a.model}-steer")
    out.mkdir(parents=True, exist_ok=True)
    prev = Path(f"out/{a.model}-lcb/summary.json")
    prev = json.load(open(prev)) if prev.exists() else None
    a.bs = a.bs or (prev["args"]["bs"] if prev else 125)
    a.token_budget = a.token_budget or (prev["args"]["token_budget"] if prev else 24000)

    items = load_lcb(a.lcb, a.langs.split(","))
    if a.limit:
        rng = random.Random(a.seed)
        items = [it for t in TASKS for it in rng.sample([x for x in items if x["task"] == t], a.limit)]
    lid = fasttext.load_model(a.lid)

    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, a.dtype or dtype), **load_kwargs(a.model))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    cfg = model.config.get_text_config()
    H = cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    head = parse(a.head)
    controls = controls_of(a.model, head, H, a.controls, a.seed)
    means = language_means(model, tok, a.model, [head] + controls, dh, a.mean_bs)
    torch.save({f"L{l}H{h}": {k: v.cpu() for k, v in m.items()} for (l, h), m in means.items()}, out / "means.pt")

    prompts = as_prompts(tok, a.model, items)
    lengths = [len(tok(x).input_ids) for x in prompts]
    groups = batches(lengths, sorted(range(len(prompts)), key=lambda i: lengths[i]), a.bs, a.token_budget)
    langs = [it["language"] for it in items]

    def run_all(spec=None):
        texts = [None] * len(prompts)
        for g in groups:
            if spec is None:
                t = generate(model, tok, prompts, g, len(g), a.max_new_tokens)
            else:
                lh, target, add = spec
                m = means[lh]
                V = torch.stack([m[target(langs[i])] - (m["all"] if add else 0) for i in g])[:, None]
                fn = (lambda x: x + V.to(x.dtype)) if add else (lambda x: V.to(x.dtype).expand_as(x))
                with patched(model, a.model, *lh, dh, fn):
                    t = generate(model, tok, prompts, g, len(g), a.max_new_tokens)
            for i in g:
                texts[i] = t[i]
        return texts

    own, other = (lambda g: g), (lambda g: SWAP[g])
    conds = [("base", None), (f"{a.head} steer", (head, own, False)), (f"{a.head} swap", (head, other, False))]
    if a.add:
        conds += [(f"{a.head} add steer", (head, own, True)), (f"{a.head} add swap", (head, other, True))]
    for l, h in controls:
        conds += [(f"L{l}H{h} steer (control)", ((l, h), own, False)),
                  (f"L{l}H{h} swap (control)", ((l, h), other, False))]
        if a.add:
            conds += [(f"L{l}H{h} add steer (control)", ((l, h), own, True)),
                      (f"L{l}H{h} add swap (control)", ((l, h), other, True))]

    results = {}
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for cond, spec in conds:
            texts = run_all(spec)
            scores = [score(t, it["language"], lid) for t, it in zip(texts, items)]
            results[cond] = scores
            for it, t, s in zip(items, texts, scores):
                row = {"cond": cond, **it, "text": t, **s, "rep": repetition(t)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(cond, flush=True)
    summarize(a, out, items, results, texts_rep(out), controls, prev)


def texts_rep(out):
    rep = {}
    for line in open(out / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rep.setdefault(r["cond"], []).append(r["rep"])
    return rep


def summarize(a, out, items, results, rep, controls, prev):
    langs = a.langs.split(",")
    non_en = {t: [i for i, it in enumerate(items) if it["task"] == t and it["language"] != "en"] for t in TASKS}
    en = [i for i, it in enumerate(items) if it["task"] == "monolingual" and it["language"] == "en"]
    base = results["base"]

    def share(scores, keep, target, ci=False):
        kept = [all(l == target(i) for l in scores[i]["labels"]) for i in keep if not scores[i]["skipped"]]
        if ci:
            return paired_ci([0] * len(kept), kept) if kept else None
        return float(np.mean(kept)) if kept else float("nan")

    swapped = lambda i: SWAP[items[i]["language"]]
    lines = [f"# {a.model} {a.head}: steering on the Language Confusion Benchmark", "",
             f"Greedy {a.max_new_tokens} tokens, chat template. The head's output is replaced by its mean output on "
             "the 2,500 FLORES prompts in a language (user's text and baseline continuation, from the diagnosis run): "
             "steer = the language the reply should be in, swap = de for en, fr, es and it, fr for de. 'add' adds "
             "the language mean minus the mean over all languages instead of replacing. Controls are the same-layer "
             "heads of the earlier LCB run. LPR as in the benchmark, averaged over sources; Δ = paired change against "
             "base on the non-English prompts, bootstrap 95% CI. 'in swap language' = share of scored replies whose "
             "lines are all in the swap language. The head is replaced or shifted at every position, template and "
             "prompt included, as in the diagnosis. 'skipped' = share of replies with no line of 5+ words, which LPR "
             "leaves out.", "",
             "| condition | mono LPR | Δ mono | cross LPR | Δ cross | in swap language (mono / cross) | "
             "English lines (mono / cross) | en prompts: LPR / in German | repetition | skipped (mono / cross) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for cond, scores in results.items():
        row, cells = {}, []
        for t in TASKS:
            row[t] = lpr(items, scores, non_en[t])
            both = [i for i in non_en[t] if not scores[i]["skipped"] and not base[i]["skipped"]]
            d = paired_ci([base[i]["pass"] for i in both], [scores[i]["pass"] for i in both]) if both else None
            row[f"{t}_delta"] = d
            row[f"{t}_swap"] = share(scores, non_en[t], swapped)
            row[f"{t}_swap_ci"] = share(scores, non_en[t], swapped, ci=True)
            row[f"{t}_en"] = float(np.mean([scores[i]["en"] for i in non_en[t] if not scores[i]["skipped"]]))
            row[f"{t}_skipped"] = float(np.mean([scores[i]["skipped"] for i in non_en[t]]))
            cells += [f"{row[t]:.3f}", "" if cond == "base" or d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"]
        row["en_lpr"] = lpr(items, scores, en)
        row["en_in_german"] = share(scores, en, lambda i: "de")
        row["repetition"] = float(np.mean(rep[cond]))
        sel = {(t, g): [i for i, it in enumerate(items) if it["task"] == t and it["language"] == g]
               for t in TASKS for g in langs}
        row["per_language"] = {f"{t}/{g}": lpr(items, scores, idx) for (t, g), idx in sel.items() if idx}
        lines.append(f"| {cond} | " + " | ".join(cells) + f" | {row['monolingual_swap']:.3f} / "
                     f"{row['crosslingual_swap']:.3f} | {row['monolingual_en']:.2f} / {row['crosslingual_en']:.2f} | "
                     f"{row['en_lpr']:.3f} / {row['en_in_german']:.3f} | {row['repetition']:.3f} | "
                     f"{row['monolingual_skipped']:.3f} / {row['crosslingual_skipped']:.3f} |")
        rows[cond] = row
    if prev:
        lines += ["", f"Baseline of the earlier LCB run: mono {prev['rows']['base']['monolingual']:.3f}, "
                      f"cross {prev['rows']['base']['crosslingual']:.3f}."]
    keys = list(rows["base"]["per_language"])
    lines += ["", "LPR per task and language", "", "| condition | " + " | ".join(keys) + " |",
              "|---|" + "---|" * len(keys)]
    lines += [f"| {c} | " + " | ".join(f"{r['per_language'][k]:.2f}" for k in keys) + " |" for c, r in rows.items()]
    args = {**vars(a), "lcb": Path(a.lcb).name, "lid": Path(a.lid).name}
    json.dump({"args": args, "controls": [f"L{l}H{h}" for l, h in controls], "rows": rows,
               "n": {t: len(non_en[t]) for t in TASKS}}, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
