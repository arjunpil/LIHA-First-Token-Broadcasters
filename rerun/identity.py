import argparse
import csv
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import same
from multi import parse
from prompts import flores
from sweep import MODELS, blocks, generate

LANGS = ["en", "fr", "de", "es", "it"]


@torch.no_grad()
def head_means(model, tok, layer, head, dh, n, bs=50):
    s = slice(head * dh, (head + 1) * dh)
    dev = flores("dev")
    store = {}
    proj = blocks(model, "gpt2")[layer][1]
    handle = proj.register_forward_pre_hook(lambda m, args: store.update(x=args[0][..., s].float()))
    tok.padding_side = "right"
    means = {}
    try:
        for lang in LANGS:
            total, count = 0, 0
            sents = [x.strip() for x in dev[lang][:n]]
            for i in range(0, len(sents), bs):
                b = tok(sents[i:i + bs], return_tensors="pt", padding=True).to(model.device)
                model(**b)
                mask = b["attention_mask"][..., None].float()
                total = total + (store["x"] * mask).sum((0, 1))
                count += mask.sum().item()
            means[lang] = total / count
    finally:
        handle.remove()
    return means


@contextmanager
def patched(model, layer, head, dh, fn):
    s = slice(head * dh, (head + 1) * dh)

    def hook(m, args):
        x = args[0].clone()
        x[..., s] = fn(x[..., s])
        return (x,) + args[1:]

    handle = blocks(model, "gpt2")[layer][1].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run(a):
    rows = list(csv.DictReader(open(a.prompts, encoding="utf-8")))
    prompts = [r["prompt"] for r in rows]
    name, dtype = MODELS["gpt2"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype),
                                                 attn_implementation="eager").cuda().eval()
    dh = model.config.hidden_size // model.config.num_attention_heads
    layer, head = parse(a.head)
    mu = head_means(model, tok, layer, head, dh, a.n_mean)
    mu_all = sum(mu.values()) / len(mu)
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    conds = [("base", None)]
    conds += [(f"set:{l}", lambda x, l=l: mu[l].to(x.dtype).expand_as(x)) for l in LANGS]
    conds += [(f"add:{l}", lambda x, l=l: x + a.scale * (mu[l] - mu_all).to(x.dtype)) for l in LANGS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
        for cond, fn in conds:
            if fn is None:
                texts = generate(model, tok, prompts, order, a.bs, 40)
            else:
                with patched(model, layer, head, dh, fn):
                    texts = generate(model, tok, prompts, order, a.bs, 40)
            f.write(json.dumps({"cond": cond, "texts": texts, "nll": {}}, ensure_ascii=False) + "\n")
            f.flush()
            print(cond, flush=True)


def report(a):
    out = Path(a.out)
    expected = [r["language"] for r in csv.DictReader(open(out / "prompts.csv", encoding="utf-8"))]
    lab = json.load(open(out / "labels.json"))
    lines = [f"# {a.head}: replacing (set) or shifting (add) the head output with a language's mean", "",
             "Share of outputs in the prompt's own language / the patched-in language / English, by prompt language.", "",
             "| condition | " + " | ".join(LANGS) + " |", "|---|" + "---|" * len(LANGS)]
    for cond, v in lab.items():
        target = cond.split(":")[1] if ":" in cond else None
        cells = []
        for p in LANGS:
            labels = [x for x, e in zip(v["labels"], expected) if e == p]
            c = Counter("own" if same(x, p) else "target" if target and same(x, target) else "en" if x == "en" else "other"
                        for x in labels)
            n = len(labels)
            cells.append(f"{c['own'] / n:.2f} / {c['target'] / n:.2f} / {c['en'] / n:.2f}" if target and target != p
                         else f"{c['own'] / n:.2f} / – / {c['en'] / n:.2f}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--prompts", default="prompts_european.csv")
    p.add_argument("--head", default="L6H10")
    p.add_argument("--n-mean", type=int, default=200)
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--bs", type=int, default=500)
    p.add_argument("--out", default="out/gpt2-identity")
    a = p.parse_args()
    if a.step == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
