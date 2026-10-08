# line-level pass rate (LPR) follows compute_metrics.py from github.com/for-ai/language-confusion (Apache-2.0)
import argparse
import csv
import io
import json
import random
import string
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import fasttext
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from followup import patched, prompt_means
from multi import parse
from sweep import MODELS, as_prompts, generate, load_kwargs

TASKS = ["monolingual", "crosslingual"]
LCB_URL = "https://raw.githubusercontent.com/for-ai/language-confusion/HEAD/test_sets.zip"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
PUNCT = str.maketrans("", "", string.punctuation)


def load_lcb(path, langs):
    path = Path(path)
    if not path.exists():
        zipfile.ZipFile(io.BytesIO(urlopen(LCB_URL).read())).extractall(path.parent)
    rows = []
    for task in TASKS:
        for f in sorted((path / task).glob("*/*.csv")):
            if f.stem in langs:
                rows += [{"task": task, "source": f.parent.name, "language": f.stem, "prompt": r["prompt"]}
                         for r in csv.DictReader(open(f, encoding="utf-8", newline=""))]
    return rows


def score(text, lang, lid):
    text = text.split("\nQ:")[0].strip().translate(PUNCT).replace("—", " ").replace("،", "")
    lines = [line for line in text.split("\n") if len(line.split()) >= 5]
    if not lines:
        return {"skipped": True}
    labels = []
    for line in lines:
        # same as lid.predict, which breaks under numpy 2: it appends the newline before predicting
        (p, label), = lid.f.predict(line + "\n", 1, 0.0, "strict")
        labels.append(label[9:] if p > 0.3 else "unknown")
    return {"skipped": False, "labels": labels, "pass": all(l == lang for l in labels),
            "en": sum(l == "en" for l in labels) / len(labels)}


def repetition(text):
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def lpr(items, scores, keep):
    # the benchmark averages over sources
    by = {}
    for i in keep:
        if not scores[i]["skipped"]:
            by.setdefault(items[i]["source"], []).append(scores[i]["pass"])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def batches(lengths, order, bs, budget):
    # order is sorted by length, so the last prompt added sets the padded length of the batch
    out, cur = [], []
    for i in order:
        if cur and (len(cur) == bs or (len(cur) + 1) * lengths[i] > budget):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


@torch.no_grad()
def sample(model, tok, prompts, idx, max_new, temperature, seed):
    tok.padding_side = "left"
    b = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    # top-p, top-k and repetition penalty stay at the model's generation config
    ids = model.generate(**b, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                         pad_token_id=tok.eos_token_id)
    return dict(zip(idx, tok.batch_decode(ids[:, b["input_ids"].shape[1]:], skip_special_tokens=True)))


def paired_ci(a, b, n=2000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    means = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=[m for m in MODELS if m.endswith("instruct")])
    p.add_argument("--heads", default="", help="comma separated, each gets zero and mean ablation")
    p.add_argument("--scale", default="", help="e.g. L17H8:3,L17H7:2")
    p.add_argument("--controls", type=int, default=3, help="random zero-ablated heads per layer of --heads")
    p.add_argument("--langs", default="fr,de,es,it,en")
    p.add_argument("--lcb", default="lcb/test_sets")
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--bs", type=int, default=125)
    p.add_argument("--token-budget", type=int, default=24000, help="prompt tokens per batch, LCB prompts vary a lot")
    p.add_argument("--dtype", default=None)
    p.add_argument("--limit", type=int, default=None, help="prompts per task, for smoke tests")
    p.add_argument("--tasks", default="monolingual,crosslingual")
    p.add_argument("--per-lang", type=int, default=None, help="random prompts per task and language")
    p.add_argument("--screen", action="store_true", help="zero every head in turn, LPR only")
    p.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--report-only", action="store_true", help="rescore the saved replies without generating")
    a = p.parse_args()
    out = Path(a.out or f"out/{a.model}-lcb")
    out.mkdir(parents=True, exist_ok=True)

    if not Path(a.lid).exists():
        urlretrieve(LID_URL, a.lid)
    lid = fasttext.load_model(a.lid)
    results = {}

    def record(f, cond, texts):
        scores = [score(t, it["language"], lid) for t, it in zip(texts, items)]
        results[cond] = scores
        for it, t, s in zip(items, texts, scores):
            f.write(json.dumps({"cond": cond, **it, "text": t, **s, "rep": repetition(t)}, ensure_ascii=False) + "\n")
        f.flush()

    if a.report_only:
        saved = [json.loads(line) for line in open(out / "samples.jsonl", encoding="utf-8")]
        items = [{k: r[k] for k in ("task", "source", "language", "prompt")} for r in saved
                 if r["cond"] == saved[0]["cond"]]
        texts = {}
        for r in saved:
            texts.setdefault(r["cond"], []).append(r["text"])
        controls = json.load(open(out / "summary.json"))["controls"]
        with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
            for cond, t in texts.items():
                record(f, cond, t)
        return summarize(a, out, items, results, controls)

    items = [it for it in load_lcb(a.lcb, a.langs.split(",")) if it["task"] in a.tasks.split(",")]
    rng = random.Random(a.seed)
    if a.limit:
        items = [it for t in TASKS for it in rng.sample([x for x in items if x["task"] == t], a.limit)]
    if a.per_lang:
        groups = {}
        for it in items:
            groups.setdefault((it["task"], it["language"]), []).append(it)
        items = [it for g in groups.values() for it in rng.sample(g, min(a.per_lang, len(g)))]

    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, a.dtype or dtype), **load_kwargs(a.model))
    model = model.to(device).eval()
    if a.temperature:
        a.top_p, a.top_k = model.generation_config.top_p, model.generation_config.top_k
    cfg = model.config.get_text_config()
    H = cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    prompts = as_prompts(tok, a.model, items)
    lengths = [len(tok(p).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    groups = batches(lengths, order, a.bs, a.token_budget)

    def run_all():
        texts = [None] * len(prompts)
        for j, g in enumerate(groups):
            if a.temperature > 0:  # the same seed per batch in every condition, so the comparison stays paired
                out = sample(model, tok, prompts, g, a.max_new_tokens, a.temperature, a.sample_seed * 100000 + j)
            else:
                out = generate(model, tok, prompts, g, len(g), a.max_new_tokens)
            for i in g:
                texts[i] = out[i]
        return texts

    if a.screen:
        with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
            record(f, "base", run_all())
            for l in range(cfg.num_hidden_layers):
                for h in range(H):
                    with patched(model, a.model, l, h, dh, lambda x: torch.zeros_like(x)):
                        record(f, f"L{l}H{h}", run_all())
                print(f"layer {l} done", flush=True)
        return screen_report(out, items, results)

    heads = [parse(h) for h in a.heads.split(",")]
    rng = random.Random(a.seed)
    controls = []
    for l in sorted({l for l, _ in heads}):
        pool = [h for h in range(H) if (l, h) not in heads]
        controls += [(l, h) for h in sorted(rng.sample(pool, min(a.controls, len(pool))))]
    flores_rows = list(csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")))
    mu = prompt_means(model, tok, a.model, as_prompts(tok, a.model, flores_rows), heads, dh, 64)

    conds = [("base", None)]
    for l, h in heads:
        conds += [(f"L{l}H{h} zero", (l, h, lambda x: torch.zeros_like(x))),
                  (f"L{l}H{h} mean", (l, h, lambda x, m=mu[(l, h)]: m.to(x.dtype).expand_as(x)))]
    for spec in filter(None, a.scale.split(",")):
        hd, s = spec.split(":")
        l, h = parse(hd)
        conds.append((f"{hd} x{s}", (l, h, lambda x, s=float(s): x * s)))
    conds += [(f"L{l}H{h} zero (control)", (l, h, lambda x: torch.zeros_like(x))) for l, h in controls]

    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for cond, spec in conds:
            if spec is None:
                record(f, cond, run_all())
            else:
                l, h, fn = spec
                with patched(model, a.model, l, h, dh, fn):
                    record(f, cond, run_all())
            print(cond, flush=True)
    summarize(a, out, items, results, [f"L{l}H{h}" for l, h in controls])


def screen_report(out, items, results):
    non_en = [i for i, it in enumerate(items) if it["language"] != "en"]

    def rate(scores):
        kept = [scores[i]["pass"] for i in non_en if not scores[i]["skipped"]]
        return float(np.mean(kept)) if kept else float("nan")

    base = rate(results["base"])
    rows = sorted(((c, rate(s)) for c, s in results.items() if c != "base"), key=lambda r: r[1])
    lines = [f"# LCB head screen, {len(non_en)} non-English prompts", "",
             "Every head zero-ablated in turn. Pooled pass rate (all lines in the expected language); "
             f"baseline {base:.3f}.",
             "", "| head | pass rate | Δ |", "|---|---|---|"]
    lines += [f"| {c} | {r:.3f} | {r - base:+.3f} |" for c, r in rows[:20]]
    json.dump({"base": base, "heads": dict(rows)}, open(out / "screen.json", "w"), indent=1)
    (out / "screen.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def summarize(a, out, items, results, controls):
    langs = a.langs.split(",")
    sel = {(t, l): [i for i, it in enumerate(items) if it["task"] == t and it["language"] == l]
           for t in TASKS for l in langs}
    non_en = {t: [i for l in langs if l != "en" for i in sel[(t, l)]] for t in TASKS}
    base = results["base"]
    rows = {}
    lines = [f"# {a.model}: Language Confusion Benchmark", "",
             ("Greedy" if not a.temperature else f"Sampling at temperature {a.temperature}, top-p "
              f"{getattr(a, 'top_p', None)}, top-k {getattr(a, 'top_k', None)} (seed {a.sample_seed}),")
             + f" {a.max_new_tokens} tokens, chat template. LPR = share of replies whose lines (5+ words) are all "
             "in the expected language, averaged over sources as in the benchmark. Δ = paired change against base "
             "on the non-English prompts both runs score, pooled, with a bootstrap 95% CI. Mean ablation uses the "
             "head's mean over the 2,500 FLORES prompts. Controls are random heads from the same layers, "
             "zero-ablated.", "",
             "| condition | mono LPR | Δ mono | cross LPR | Δ cross | mono en LPR | English lines (mono / cross) |",
             "|---|---|---|---|---|---|---|"]
    for cond, scores in results.items():
        row = {}
        cells = []
        for t in TASKS:
            row[t] = lpr(items, scores, non_en[t])
            both = [i for i in non_en[t] if not scores[i]["skipped"] and not base[i]["skipped"]]
            d = paired_ci([base[i]["pass"] for i in both], [scores[i]["pass"] for i in both]) if both else None
            row[f"{t}_delta"] = d
            row[f"{t}_en"] = float(np.mean([scores[i]["en"] for i in non_en[t] if not scores[i]["skipped"]]))
            cells += [f"{row[t]:.3f}", "" if cond == "base" or d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"]
        row["per_language"] = {f"{t}/{l}": lpr(items, scores, sel[(t, l)]) for t in TASKS for l in langs
                               if sel[(t, l)]}
        en_mono = row["per_language"].get("monolingual/en", float("nan"))
        lines.append(f"| {cond} | " + " | ".join(cells) + f" | {en_mono:.3f} | "
                     f"{row['monolingual_en']:.2f} / {row['crosslingual_en']:.2f} |")
        rows[cond] = row
    keys = [k for k in rows["base"]["per_language"]]
    lines += ["", "LPR per task and language", "", "| condition | " + " | ".join(keys) + " |",
              "|---|" + "---|" * len(keys)]
    lines += [f"| {c} | " + " | ".join(f"{r['per_language'][k]:.2f}" for k in keys) + " |" for c, r in rows.items()]
    args = {**vars(a), "lcb": Path(a.lcb).name, "lid": Path(a.lid).name}  # no local paths
    json.dump({"args": args, "controls": controls, "rows": rows,
               "n": {t: len(non_en[t]) for t in TASKS}}, open(out / "summary.json", "w"), indent=1)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
