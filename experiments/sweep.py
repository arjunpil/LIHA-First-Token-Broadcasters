import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
from prompts import flores  # noqa: E402

MODELS = {
    "gpt2": ("gpt2", "float32"),
    "qwen-base": ("Qwen/Qwen2.5-1.5B", "float16"),
    "qwen-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "float16"),
    "bloom": ("bigscience/bloom-1b7", "float32"),  # fp16 gives NaN on left-padded rows
    "gpt2-medium": ("gpt2-medium", "float32"),
    "olmo2-1b": ("allenai/OLMo-2-0425-1B", "bfloat16"),
    "pythia-1b": ("EleutherAI/pythia-1b", "float32"),
}
NO_EOS = {"olmo2-1b"}  # ends the document after most complete FLORES sentences, so the end token is blocked
PAPER_HEAD_DIM = {"bloom": 64}  # bloom_experiment.py assumed hidden 1024


def blocks(model, key):
    if key.startswith("gpt2"):
        return [(h.attn, h.attn.c_proj) for h in model.transformer.h]
    if key == "bloom":
        return [(h.self_attention, h.self_attention.dense) for h in model.transformer.h]
    if key.startswith("pythia"):
        return [(l.attention, l.attention.dense) for l in model.gpt_neox.layers]
    return [(l.self_attn, l.self_attn.o_proj) for l in model.model.layers]


@contextmanager
def ablated(model, key, mode, layer, head, dh):
    attn, proj = blocks(model, key)[layer]
    if mode == "paper":
        dh = PAPER_HEAD_DIM.get(key, dh)
    s = slice(head * dh, (head + 1) * dh)

    def zero(x):
        x = x.clone()
        x[..., s] = 0
        return x

    if mode == "head":
        handle = proj.register_forward_pre_hook(lambda m, args: (zero(args[0]),) + args[1:])
    elif key.startswith("qwen") or key.startswith("olmo"):
        handle = proj.register_forward_hook(lambda m, inp, out: zero(out))
    else:
        handle = attn.register_forward_hook(lambda m, inp, out: (zero(out[0]),) + tuple(out[1:]))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(model, tok, prompts, order, bs, max_new, min_new=0):
    tok.padding_side = "left"
    texts = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        ids = model.generate(**b, max_new_tokens=max_new, min_new_tokens=min_new or None, do_sample=False,
                             pad_token_id=tok.eos_token_id, temperature=None, top_p=None, top_k=None)
        for i, t in zip(idx, tok.batch_decode(ids[:, b["input_ids"].shape[1]:], skip_special_tokens=True)):
            texts[i] = t
    return texts


@torch.no_grad()
def nll(model, tok, sents, prefix, bs=64):
    tok.padding_side = "right"
    total, n = 0.0, 0
    for s in range(0, len(sents), bs):
        b = tok([prefix + x for x in sents[s:s + bs]], return_tensors="pt", padding=True).to(model.device)
        logits = model(**b).logits[:, :-1].float()
        target, mask = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, target[..., None])[..., 0]
        total -= lp[mask].sum().item()
        n += mask.sum().item()
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--per-lang", type=int, default=500)
    p.add_argument("--modes", default="head,paper")
    p.add_argument("--layers", default=None, help="comma separated, default all")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-loss", type=int, default=100)
    p.add_argument("--dtype", default=None, help="default per model, see MODELS")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or f"out/{a.model}")
    out.mkdir(parents=True, exist_ok=True)

    seen, rows = {}, []
    for r in csv.DictReader(open(a.prompts, encoding="utf-8")):
        seen[r["language"]] = seen.get(r["language"], 0) + 1
        if seen[r["language"]] <= a.per_lang:
            rows.append(r)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    name, dtype = MODELS[a.model]
    dtype = a.dtype or dtype
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    kwargs = {"attn_implementation": "eager"} if a.model.startswith("gpt2") else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), **kwargs).cuda().eval()
    cfg = model.config
    H = cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    layers = [int(x) for x in a.layers.split(",")] if a.layers else range(cfg.num_hidden_layers)
    if a.model == "qwen-instruct":
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                           add_generation_prompt=True) for r in rows]
    else:
        prompts = [r["prompt"] for r in rows]
    order = sorted(range(len(prompts)), key=lambda i: len(tok(prompts[i]).input_ids))
    prefix = tok.bos_token if a.model.startswith("gpt2") else ""
    min_new = a.max_new_tokens if a.model in NO_EOS else 0
    dev = flores("dev")
    loss_sents = {l: [x.strip() for x in dev[l][:a.n_loss]] for l in ("en", "fr", "de", "es", "it")}

    def record(f, cond, texts, losses):
        f.write(json.dumps({"cond": cond, "texts": texts, "nll": losses}, ensure_ascii=False) + "\n")
        f.flush()

    t0 = time.time()
    with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
        record(f, "base", generate(model, tok, prompts, order, a.bs, a.max_new_tokens, min_new),
               {l: nll(model, tok, s, prefix) for l, s in loss_sents.items()})
        for mode in a.modes.split(","):
            for layer in layers:
                for head in range(H):
                    with ablated(model, a.model, mode, layer, head, dh):
                        texts = generate(model, tok, prompts, order, a.bs, a.max_new_tokens, min_new)
                        losses = {l: nll(model, tok, s, prefix) for l, s in loss_sents.items()}
                    record(f, f"{mode}:L{layer}H{head}", texts, losses)
                print(f"{mode} layer {layer} done, {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
