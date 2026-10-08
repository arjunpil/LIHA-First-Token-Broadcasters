import argparse
import csv
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import ci, same, stats
from multi import parse, ranked
from sweep import MODELS, blocks, flores, generate, nll


@contextmanager
def scaled(model, layer, head, dh, scale):
    s = slice(head * dh, (head + 1) * dh)

    def hook(m, args):
        x = args[0].clone()
        x[..., s] *= scale
        return (x,) + args[1:]

    handle = blocks(model, "gpt2")[layer][1].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def heads(a):
    hs = ranked(a.summary, "c2w", 0.1)[:a.top] + ranked(a.summary, "sr")[:a.top] + [h for h in a.extra.split(",") if h]
    return list(dict.fromkeys(hs))


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
    conds = [("base", None, 1.0)] + [(f"{h}:x{s:g}", h, s) for h in heads(a) for s in map(float, a.scales.split(","))]
    with open(a.out, "w", encoding="utf-8") as f:
        for cond, h, s in conds:
            with ExitStack() as stack:
                if h:
                    stack.enter_context(scaled(model, *parse(h), dh, s))
                texts = generate(model, tok, prompts, order, a.bs, 40)
                losses = {l: nll(model, tok, x, tok.bos_token) for l, x in loss_sents.items()}
            f.write(json.dumps({"cond": cond, "texts": texts, "nll": losses}, ensure_ascii=False) + "\n")
            f.flush()
            print(cond, flush=True)


def report(a):
    expected = [r["language"] for r in csv.DictReader(open(a.prompts, encoding="utf-8"))]
    lab = json.load(open(a.labels))
    base = lab["base"]["labels"]
    base_nll = mean(lab["base"]["nll"].values())
    base_acc = mean(int(same(x, e)) for x, e in zip(base, expected))
    idx = range(len(expected))
    langs = ["en", "fr", "de", "es", "it"]
    lines = ["| head | scale | accuracy [95% CI] | Δacc | wrong→correct | correct→wrong | " + " | ".join(langs) + " | dNLL |",
             "|---|---|---|---|---|---|" + "---|" * len(langs) + "---|"]
    for cond, v in lab.items():
        st = stats(base, v["labels"], expected, idx)
        correct = [int(same(x, e)) for x, e in zip(v["labels"], expected)]
        lo, hi = ci(correct)
        head, scale = cond.split(":x") if cond != "base" else ("none", "1")
        by_lang = [mean(c for c, e in zip(correct, expected) if e == l) for l in langs]
        lines.append(f"| {head} | {scale} | {st['acc']:.3f} [{lo:.3f}, {hi:.3f}] | {st['acc'] - base_acc:+.3f} | "
                     f"{st['w2c']:.3f} | {st['c2w']:.3f} | " + " | ".join(f"{x:.2f}" for x in by_lang) +
                     f" | {mean(v['nll'].values()) - base_nll:+.4f} |")
    Path(a.labels).with_name("summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--summary", default="results/gpt2/summary.json")
    p.add_argument("--top", type=int, default=5, help="top heads by c2w (dNLL <= 0.1) and by SR")
    p.add_argument("--extra", default="L6H1")
    p.add_argument("--scales", default="2,3,5")
    p.add_argument("--bs", type=int, default=500)
    p.add_argument("--out", default="out/gpt2-amp/gens.jsonl")
    p.add_argument("--labels", default="out/gpt2-amp/labels.json")
    a = p.parse_args()
    if a.step == "run":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
