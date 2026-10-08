import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import same
from multi import parse
from prompts import flores
from sweep import MODELS, NO_EOS, blocks, generate, nll

LANGS = ["en", "fr", "de", "es", "it"]


@contextmanager
def patched(model, key, layer, head, dh, fn):
    s = slice(head * dh, (head + 1) * dh)

    def hook(m, args):
        x = args[0].clone()
        x[..., s] = fn(x[..., s])
        return (x,) + args[1:]

    handle = blocks(model, key)[layer][1].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def prompt_means(model, tok, key, prompts, heads, dh, bs):
    store = {}
    layers = sorted({l for l, _ in heads})
    hooks = [blocks(model, key)[l][1].register_forward_pre_hook(lambda m, args, l=l: store.__setitem__(l, args[0]))
             for l in layers]
    tok.padding_side = "left"
    total, n = {lh: 0 for lh in heads}, 0
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        model(**b)
        mask = b["attention_mask"][..., None].float()
        for l, h in heads:
            total[(l, h)] = total[(l, h)] + (store[l][..., h * dh:(h + 1) * dh].float() * mask).sum((0, 1))
        n += mask.sum().item()
    for hook in hooks:
        hook.remove()
    return {lh: t / n for lh, t in total.items()}


def run(a):
    table = json.load(open(f"out/{a.model}/summary.json"))["modes"]["head"]["table"]
    heads = a.heads.split(",") if a.heads else sorted(
        (h for h in table if table[h]["dnll"] <= a.max_dnll), key=lambda h: -table[h]["full"]["c2w"])[:a.top]
    rows = list(csv.DictReader(open(f"out/{a.model}/prompts.csv", encoding="utf-8")))
    prompts = [r["prompt"] for r in rows]
    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    kwargs = {"attn_implementation": "eager"} if a.model.startswith("gpt2") else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), **kwargs).cuda().eval()
    cfg = model.config
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    min_new = 40 if a.model in NO_EOS else 0
    prefix = tok.bos_token if a.model.startswith("gpt2") else ""
    dev = flores("dev")
    loss_sents = {l: [x.strip() for x in dev[l][:100]] for l in LANGS}
    mu = prompt_means(model, tok, a.model, prompts, [parse(h) for h in heads], dh, a.bs)

    conds = [("base", None)]
    for h in heads:
        l, k = parse(h)
        conds.append((f"{h}:mean", (l, k, lambda x, m=mu[(l, k)]: m.to(x.dtype).expand_as(x))))
        conds += [(f"{h}:x{s:g}", (l, k, lambda x, s=s: x * s)) for s in map(float, a.scales.split(","))]
    out = Path(f"out/{a.model}-followup")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
        for cond, spec in conds:
            if spec is None:
                texts = generate(model, tok, prompts, order, a.bs, 40, min_new)
                losses = {l: nll(model, tok, s, prefix) for l, s in loss_sents.items()}
            else:
                l, k, fn = spec
                with patched(model, a.model, l, k, dh, fn):
                    texts = generate(model, tok, prompts, order, a.bs, 40, min_new)
                    losses = {l: nll(model, tok, s, prefix) for l, s in loss_sents.items()}
            f.write(json.dumps({"cond": cond, "texts": texts, "nll": losses}, ensure_ascii=False) + "\n")
            f.flush()
            print(cond, flush=True)


def report(a):
    out = Path(f"out/{a.model}-followup")
    expected = [r["language"] for r in csv.DictReader(open(out / "prompts.csv", encoding="utf-8"))]
    lab = json.load(open(out / "labels.json"))
    table = json.load(open(f"out/{a.model}/summary.json"))["modes"]["head"]["table"]
    base = lab["base"]["labels"]
    base_nll = mean(lab["base"]["nll"].values())
    non_en = [i for i, e in enumerate(expected) if e != "en"]
    lines = [f"# {a.model}: follow-up on the top correct->wrong heads", "",
             "zero = the sweep's head ablation; mean = the head's mean over prompt tokens; xN = the head scaled by N.", "",
             "| condition | accuracy | non-English acc | c->w | w->c | " + " | ".join(LANGS) + " | dNLL |",
             "|---|---|---|---|---|" + "---|" * len(LANGS) + "---|"]

    def row(name, labels, dnll):
        ok = [same(x, e) for x, e in zip(labels, expected)]
        bok = [same(x, e) for x, e in zip(base, expected)]
        by = [mean(o for o, e in zip(ok, expected) if e == l) for l in LANGS]
        return (f"| {name} | {mean(ok):.3f} | {mean(ok[i] for i in non_en):.3f} | "
                f"{mean(b and not o for b, o in zip(bok, ok)):.3f} | {mean(o and not b for b, o in zip(bok, ok)):.3f} | "
                + " | ".join(f"{x:.2f}" for x in by) + f" | {dnll:+.4f} |")

    for cond, v in lab.items():
        lines.append(row(cond, v["labels"], mean(v["nll"].values()) - base_nll))
        if cond.endswith(":mean"):
            h = cond.split(":")[0]
            lines.append(f"| {h}:zero | {table[h]['full']['acc']:.3f} | | {table[h]['full']['c2w']:.3f} | "
                         f"{table[h]['full']['w2c']:.3f} | " + " | " * len(LANGS) + f"{table[h]['dnll']:+.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--heads", default=None, help="comma separated, default the top c->w heads")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--max-dnll", type=float, default=0.1)
    p.add_argument("--scales", default="2,3,5")
    p.add_argument("--bs", type=int, default=250)
    a = p.parse_args()
    if a.step == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
