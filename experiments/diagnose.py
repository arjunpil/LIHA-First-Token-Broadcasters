"""Why zero and mean ablation of a head differ. Statistics of the head's contribution on the FLORES prompts and their
baseline continuations, and replacements that separate an out-of-distribution input, the post-attention norm
rescaling the other heads, a constant signal, and language-specific information."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import same
from followup import patched
from multi import parse
from sweep import MODELS, NO_EOS, as_prompts, blocks, generate, load_kwargs

LANGS = ["en", "fr", "de", "es", "it"]
SWAP = {"en": "de", "fr": "de", "de": "fr", "es": "de", "it": "de"}  # away from English and from Spanish
POST_NORM = ("gemma", "olmo2")  # attention output goes through an RMSNorm before the residual add


def layer_of(model, key, l):
    inner = getattr(model.model, "language_model", model.model)
    return inner.layers[l]


def prefix_len(tok, key, row, prompt):
    # template tokens before the user's text see no user text, so the head's output there is the same for every prompt
    text = row["prompt"] if row["prompt"] in prompt else row["prompt"].strip()
    head = prompt.split(text)[0] if key.endswith("instruct") and text in prompt else ""
    return len(tok(head).input_ids)


@torch.no_grad()
def head_stats(model, tok, key, rows, prompts, conts, layer, head, dh, bs):
    """Means of the head's output (before the output projection, for replacing it) and statistics of its
    contribution after the projection, over: all prompt tokens (the follow-up's mean ablation), the user's text and
    the continuation, and the continuation only."""
    proj = blocks(model, key)[layer][1]
    W = proj.weight.float()
    s = slice(head * dh, (head + 1) * dh)
    Wh = W[:, s]
    store = {}
    hook = proj.register_forward_pre_hook(lambda m, args: store.__setitem__("x", args[0]))
    langs = [r["language"] for r in rows]
    n_pre = [prefix_len(tok, key, r, p) for r, p in zip(rows, prompts)]
    n_prompt = [len(tok(p).input_ids) for p in prompts]
    sums, counts, energy = defaultdict(float), Counter(), Counter()
    n_heads = W.shape[1] // dh
    head_norm, head_n = torch.zeros(n_heads, device=model.device), 0
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        idx = list(range(i, min(i + bs, len(prompts))))
        b = tok([prompts[j] + conts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
        model.base_model(**b)
        x = store["x"].float()
        T = x.shape[1]
        lengths = b["attention_mask"].sum(1).tolist()
        c = x[..., s] @ Wh.T
        text_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for r, j in enumerate(idx):
            off = T - lengths[r]
            segs = {"prompt": (off, off + n_prompt[j]), "text": (off + n_pre[j], T), "gen": (off + n_prompt[j], T)}
            text_mask[r, segs["text"][0]:T] = True
            for name, (a, z) in segs.items():
                for k in (name, (name, langs[j])):
                    sums[k] = sums[k] + x[r, a:z, s].sum(0)
                    counts[k] += z - a
                    energy[k] += (c[r, a:z] ** 2).sum().item()
        for h in range(n_heads):
            hs = slice(h * dh, (h + 1) * dh)
            head_norm[h] += ((x[..., hs] @ W[:, hs].T).norm(dim=-1) * text_mask).sum()
        head_n += text_mask.sum().item()
    hook.remove()
    mu = {k: sums[k] / counts[k] for k in sums}

    def summary(seg):
        m = mu[seg] @ Wh.T
        e = energy[seg] / counts[seg]
        var = e - (m ** 2).sum().item()
        between = sum(counts[(seg, l)] * ((mu[(seg, l)] @ Wh.T - m) ** 2).sum().item()
                      for l in LANGS if counts[(seg, l)]) / counts[seg]
        return {"tokens": counts[seg], "mean_share_of_energy": (m ** 2).sum().item() / e,
                "language_share_of_variance": between / var}

    norms = (head_norm / head_n).tolist()
    mp, mg = mu["prompt"] @ Wh.T, mu["gen"] @ Wh.T
    stats = {"contribution_norm": norms[head], "layer_contribution_norms": norms,
             "norm_rank": sorted(norms, reverse=True).index(norms[head]) + 1,
             "text_and_continuation": summary("text"), "continuation": summary("gen"),
             "prompt_mean_vs_continuation_mean": {
                 "cosine": torch.nn.functional.cosine_similarity(mp, mg, dim=0).item(),
                 "relative_distance": ((mp - mg).norm() / mg.norm()).item()},
             "template_tokens": mean(n_pre)}
    return stats, mu


@contextmanager
def frozen_norm_zero(model, key, layer, head, dh):
    """Zero the head but divide by the RMS the post-attention norm would have seen without the ablation."""
    proj = blocks(model, key)[layer][1]
    norm = layer_of(model, key, layer).post_attention_layernorm
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))
    s = slice(head * dh, (head + 1) * dh)
    saved = {}

    def rms(y):
        return torch.sqrt(y.float().pow(2).mean(-1, keepdim=True) + eps)

    def on_proj(m, args, out):
        saved["clean"] = rms(out)
        return out - args[0][..., s] @ m.weight[:, s].T

    def on_norm(m, args, out):
        return (out.float() * rms(args[0]) / saved["clean"]).to(out.dtype)

    h1 = proj.register_forward_hook(on_proj)
    h2 = norm.register_forward_hook(on_norm)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()


def gen_by_language(model, tok, key, prompts, langs, layer, head, dh, bs, min_new, fn=None, per_lang=None, ctx=None):
    """Greedy 40 tokens, batched per language in every condition so that padding is the same across conditions.
    fn replaces the head's slice; per_lang(l) gives the replacement vector for prompts in language l."""
    texts = [None] * len(prompts)
    for g in LANGS:
        idx = [i for i, l in enumerate(langs) if l == g]
        if not idx:
            continue
        sub = [prompts[i] for i in idx]
        order = sorted(range(len(sub)), key=lambda i: len(tok(sub[i]).input_ids))
        f = fn
        if per_lang is not None:
            v = per_lang(g)
            f = lambda x, v=v: v.to(x.dtype).expand_as(x)
        if ctx is not None:
            with ctx():
                t = generate(model, tok, sub, order, bs, 40, min_new)
        elif f is None:
            t = generate(model, tok, sub, order, bs, 40, min_new)
        else:
            with patched(model, key, layer, head, dh, f):
                t = generate(model, tok, sub, order, bs, 40, min_new)
        for i, x in zip(idx, t):
            texts[i] = x
    return texts


def run(a):
    rows = list(csv.DictReader(open(f"out/{a.model}/prompts.csv", encoding="utf-8")))
    if a.limit:
        seen = Counter()
        rows = [r for r in rows if seen.update([r["language"]]) or seen[r["language"]] <= a.limit]
    langs = [r["language"] for r in rows]
    layer, head = parse(a.head)
    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, a.dtype or dtype), **load_kwargs(a.model))
    model = model.to(a.device).eval()
    cfg = model.config.get_text_config()
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    prompts = as_prompts(tok, a.model, rows)
    min_new = 40 if a.model in NO_EOS else 0
    out = Path(a.out or f"out/{a.model}-diag")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def gen(**kw):
        return gen_by_language(model, tok, a.model, prompts, langs, layer, head, dh, a.bs, min_new, **kw)

    base = gen()
    stats, mu = head_stats(model, tok, a.model, rows, prompts, base, layer, head, dh, a.bs)
    json.dump({"head": a.head, **stats}, open(out / "stats.json", "w"), indent=1)

    noise_gen = torch.Generator(device=model.device)

    def noise(x):
        r = torch.randn(x.shape, generator=noise_gen, device=x.device, dtype=torch.float32)
        return (r * (x.float().norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True))).to(x.dtype)

    const = lambda v: (lambda x: v.to(x.dtype).expand_as(x))
    conds = [("zero", {"fn": lambda x: torch.zeros_like(x)}),
             ("mean of all prompt tokens (follow-up)", {"fn": const(mu["prompt"])}),
             ("mean of continuation tokens", {"fn": const(mu["gen"])}),
             ("minus the continuation mean", {"fn": lambda x: x - mu["gen"].to(x.dtype)}),
             ("own-language mean", {"per_lang": lambda l: mu[("text", l)]}),
             ("English mean", {"per_lang": lambda l: mu[("text", "en")]}),
             ("other-language mean", {"per_lang": lambda l: mu[("text", SWAP[l])]}),
             ("random, same norm", {"fn": noise}),
             ("x0.5", {"fn": lambda x: x * 0.5})]
    if a.model.startswith(POST_NORM):
        conds.append(("zero, norm frozen", {"ctx": lambda: frozen_norm_zero(model, a.model, layer, head, dh)}))
    with open(out / "gens.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"cond": "base", "texts": base, "nll": {}}, ensure_ascii=False) + "\n")
        for cond, kw in conds:
            noise_gen.manual_seed(0)
            f.write(json.dumps({"cond": cond, "texts": gen(**kw), "nll": {}}, ensure_ascii=False) + "\n")
            f.flush()
            print(cond, flush=True)


def report(a):
    out = Path(a.out or f"out/{a.model}-diag")
    expected = [r["language"] for r in csv.DictReader(open(out / "prompts.csv", encoding="utf-8"))]
    lab = json.load(open(out / "labels.json"))
    st = json.load(open(out / "stats.json"))
    tc, cont, pm = st["text_and_continuation"], st["continuation"], st["prompt_mean_vs_continuation_mean"]
    norms = sorted(st["layer_contribution_norms"])
    base = lab["base"]["labels"]
    bok = [same(x, e) for x, e in zip(base, expected)]
    non_en = [i for i, e in enumerate(expected) if e != "en"]
    lines = [f"# {a.model} {st['head']}: why zero and mean ablation differ", "",
             f"The head's contribution after the output projection, on the user's text and the baseline "
             f"continuation of {len(expected)} FLORES prompts (the {st['template_tokens']:.0f} template tokens before "
             f"the text are left out, since the head's output there is the same for every prompt): mean norm "
             f"{st['contribution_norm']:.2f}, rank {st['norm_rank']} of {len(norms)} in its layer (layer median "
             f"{norms[len(norms) // 2]:.2f}).", "",
             "| tokens | n | mean vector's share of the energy | prompt language's share of the rest |",
             "|---|---|---|---|",
             f"| user's text and continuation | {tc['tokens']} | {tc['mean_share_of_energy']:.2f} | "
             f"{tc['language_share_of_variance']:.2f} |",
             f"| continuation only | {cont['tokens']} | {cont['mean_share_of_energy']:.2f} | "
             f"{cont['language_share_of_variance']:.2f} |", "",
             f"The follow-up's mean (all prompt tokens, template included) against the continuation mean: cosine "
             f"{pm['cosine']:.2f}, distance {pm['relative_distance']:.2f} of the continuation mean's norm.", "",
             "Greedy 40 tokens, batched per language in every condition. Language means are over the user's text "
             "and the continuation. other-language mean: de for en, fr, es and it prompts, fr for de prompts.", "",
             "| condition | non-English acc | c->w | w->c | non-English replies in English | "
             "in the other-language target |", "|---|---|---|---|---|---|"]
    for cond, v in lab.items():
        lb = v["labels"]
        ok = [same(x, e) for x, e in zip(lb, expected)]
        c2w = mean(b and not o for b, o in zip(bok, ok))
        w2c = mean(o and not b for b, o in zip(bok, ok))
        en = mean(same(lb[i], "en") for i in non_en)
        tgt = mean(same(lb[i], SWAP[expected[i]]) for i in non_en)
        lines.append(f"| {cond} | {mean(ok[i] for i in non_en):.3f} | {c2w:.3f} | {w2c:.3f} | {en:.3f} | {tgt:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["run", "report"])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--head", required=True)
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--dtype", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="prompts per language, for smoke tests")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a) if a.step == "run" else report(a)


if __name__ == "__main__":
    main()
