import argparse
import csv
import json
import random
from contextlib import ExitStack
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import ci, same
from sweep import MODELS, ablated, flores, generate, nll


def ranked(summary, key, max_dnll=float("inf")):
    t = json.load(open(summary))["modes"]["head"]["table"]
    return sorted((h for h in t if t[h]["dnll"] <= max_dnll), key=lambda h: -t[h]["full"][key])


def parse(h):
    layer, head = h[1:].split("H")
    return int(layer), int(head)


def run(a):
    rows = list(csv.DictReader(open(a.prompts, encoding="utf-8")))
    prompts = [r["prompt"] for r in rows]
    name, dtype = MODELS["gpt2"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype),
                                                 attn_implementation="eager").cuda().eval()
    dh = model.config.hidden_size // model.config.num_attention_heads
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    dev = flores("dev")
    loss_sents = {l: [x.strip() for x in dev[l][:100]] for l in ("en", "fr", "de", "es", "it")}
    every = [f"L{l}H{h}" for l in range(model.config.num_hidden_layers) for h in range(model.config.num_attention_heads)]
    orders = {"sr": ranked(a.summary, "sr"), "c2w": ranked(a.summary, "c2w"),
              "c2w-lowloss": ranked(a.summary, "c2w", a.max_dnll)}
    for s in range(a.n_random):
        orders[f"random{s}"] = random.Random(s).sample(every, len(every))
    orders = {o: hs for o, hs in orders.items() if o.rstrip("0123456789") in a.orders.split(",")}
    with open(a.out, "w", encoding="utf-8") as f:
        for name, heads in [("base", [])] + [(f"{o}:k{k}", hs[:k]) for o, hs in orders.items() for k in range(1, a.k + 1)]:
            with ExitStack() as stack:
                for h in heads:
                    stack.enter_context(ablated(model, "gpt2", "head", *parse(h), dh))
                texts = generate(model, tok, prompts, order, a.bs, 40)
                losses = {l: nll(model, tok, x, tok.bos_token) for l, x in loss_sents.items()}
            f.write(json.dumps({"cond": name, "heads": heads, "texts": texts, "nll": losses}, ensure_ascii=False) + "\n")
            f.flush()
            print(name, heads[-1] if heads else "", flush=True)


def report(a):
    rows = list(csv.DictReader(open(a.prompts, encoding="utf-8")))
    expected = [r["language"] for r in rows]
    lab = json.load(open(a.labels))
    heads = {json.loads(l)["cond"]: json.loads(l)["heads"] for l in open(a.out, encoding="utf-8")}
    base_nll = mean(lab["base"]["nll"].values())
    lines = ["| ablated | k | last head | accuracy [95% CI] | dNLL |", "|---|---|---|---|---|"]
    for cond, v in lab.items():
        correct = [int(same(x, e)) for x, e in zip(v["labels"], expected)]
        lo, hi = ci(correct)
        order, k = (cond.split(":k") + ["0"])[:2] if cond != "base" else ("none", "0")
        last = heads[cond][-1] if heads[cond] else "-"
        lines.append(f"| {order} | {k} | {last} | {mean(correct):.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{mean(v['nll'].values()) - base_nll:+.4f} |")
    open(a.labels.replace("labels.json", "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--summary", default="results/gpt2/summary.json")
    p.add_argument("--orders", default="sr,c2w,random")
    p.add_argument("--max-dnll", type=float, default=0.1, help="for c2w-lowloss")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--bs", type=int, default=500)
    p.add_argument("--out", default="out/gpt2-multi/gens.jsonl")
    p.add_argument("--labels", default="out/gpt2-multi/labels.json")
    a = p.parse_args()
    if a.step == "run":
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
