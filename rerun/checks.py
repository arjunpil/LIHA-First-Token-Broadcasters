import argparse
import csv
import json
import random
from contextlib import ExitStack
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import same
from multi import parse
from sweep import MODELS, ablated, generate

GPT2_HEADS = ["L6H10", "L2H5", "L4H8", "L6H1"]
QWEN_HEADS = ["L22H6", "L17H7", "L17H8", "L0H6"]


def controls(summary, exclude, n=3, seed=0):
    t = json.load(open(summary))["modes"]["head"]["table"]
    top = set(sorted(t, key=lambda h: -t[h]["full"]["c2w"])[:10]) | set(exclude)
    pool = sorted(h for h in t if t[h]["dnll"] <= 0.1 and h not in top)
    return random.Random(seed).sample(pool, n)


def rows_for(path, per_lang=None):
    seen, rows = {}, []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        seen[r["language"]] = seen.get(r["language"], 0) + 1
        if per_lang is None or seen[r["language"]] <= per_lang:
            rows.append(r)
    return rows


def truncate(rows):
    out = []
    for r in rows:
        words = r["prompt"].split()
        cut = " ".join(words[:max(4, len(words) // 2)]) if r["source"] == "flores200" else r["prompt"]
        out.append({**r, "prompt": cut})
    return out


@torch.no_grad()
def sample(model, tok, prompts, bs, max_new, seed):
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    texts = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed * 100000 + s)
        ids = model.generate(**b, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
        for i, t in zip(idx, tok.batch_decode(ids[:, b["input_ids"].shape[1]:], skip_special_tokens=True)):
            texts[i] = t
    return texts


def load(key):
    name, dtype = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    kwargs = {"attn_implementation": "eager"} if key == "gpt2" else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), **kwargs).cuda().eval()
    cfg = model.config
    return tok, model, getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def run_conditions(f, model, key, dh, heads, gen):
    for cond, h in [("base", None)] + [(f"head:{h}", h) for h in heads]:
        with ExitStack() as stack:
            if h:
                stack.enter_context(ablated(model, key, "head", *parse(h), dh))
            texts = gen()
        f.write(json.dumps({"cond": cond, "texts": texts, "nll": {}}, ensure_ascii=False) + "\n")
        f.flush()
        print(cond, flush=True)


def write_prompts(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(a):
    out = Path(a.out or f"out/{a.check}")
    if a.check in ("gpt2-sampling", "gpt2-truncated"):
        rows = rows_for("prompts_european.csv", a.per_lang)
        if a.check == "gpt2-truncated":
            rows = truncate(rows)
        write_prompts(out, rows)
        heads = GPT2_HEADS + controls("results/gpt2/summary.json", GPT2_HEADS)
        tok, model, dh = load("gpt2")
        prompts = [r["prompt"] for r in rows]
        order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
        with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
            if a.check == "gpt2-truncated":
                run_conditions(f, model, "gpt2", dh, heads, lambda: generate(model, tok, prompts, order, a.bs, 40))
            else:
                for seed in range(a.seeds):
                    for cond, h in [("base", None)] + [(f"head:{h}", h) for h in heads]:
                        with ExitStack() as stack:
                            if h:
                                stack.enter_context(ablated(model, "gpt2", "head", *parse(h), dh))
                            texts = sample(model, tok, prompts, a.bs, 40, seed)
                        f.write(json.dumps({"cond": f"s{seed}:{cond}", "texts": texts, "nll": {}},
                                           ensure_ascii=False) + "\n")
                        f.flush()
                        print(seed, cond, flush=True)
    else:
        rows = rows_for("prompts_european.csv", a.per_lang or 25)
        write_prompts(out, rows)
        heads = QWEN_HEADS + controls("results/qwen-instruct/summary.json", QWEN_HEADS)
        template = AutoTokenizer.from_pretrained(MODELS["qwen-instruct"][0])
        chat = [template.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                             add_generation_prompt=True) for r in rows]
        raw = [r["prompt"] for r in rows]
        with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
            for key, prompts, tag in (("qwen-instruct", raw, "instruct-raw"), ("qwen-base", chat, "base-chat")):
                tok, model, dh = load(key)
                order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
                for cond, h in [("base", None)] + [(f"head:{h}", h) for h in heads]:
                    with ExitStack() as stack:
                        if h:
                            stack.enter_context(ablated(model, key, "head", *parse(h), dh))
                        texts = generate(model, tok, prompts, order, a.bs, 40)
                    f.write(json.dumps({"cond": f"{tag}:{cond}", "texts": texts, "nll": {}}, ensure_ascii=False) + "\n")
                    f.flush()
                    print(tag, cond, flush=True)
                del model
                torch.cuda.empty_cache()


def report(a):
    out = Path(a.out or f"out/{a.check}")
    expected = [r["language"] for r in csv.DictReader(open(out / "prompts.csv", encoding="utf-8"))]
    lab = json.load(open(out / "labels.json"))
    non_en = [i for i, e in enumerate(expected) if e != "en"]
    lines = [f"# {a.check}", "", "| setting | head | accuracy | non-English acc | c->w | w->c |", "|---|---|---|---|---|---|"]
    for cond, v in lab.items():
        prefix = cond[:-len("base")] if cond.endswith("base") else cond[:cond.index("head:")]
        base = lab[prefix + "base"]["labels"]
        x = v["labels"]
        ok = [same(l, e) for l, e in zip(x, expected)]
        bok = [same(l, e) for l, e in zip(base, expected)]
        lines.append(f"| {prefix.rstrip(':') or '-'} | {cond[len(prefix):]} | {mean(ok):.3f} | "
                     f"{mean(ok[i] for i in non_en):.3f} | {mean(b and not o for b, o in zip(bok, ok)):.3f} | "
                     f"{mean(o and not b for b, o in zip(bok, ok)):.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("check", choices=["gpt2-sampling", "gpt2-truncated", "qwen-format"])
    p.add_argument("--per-lang", type=int, default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.step == "run":
        run(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
