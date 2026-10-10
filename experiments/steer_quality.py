"""Quality of the steered LCB replies. Content: multilingual embedding similarity between a reply and the baseline
reply to the same prompt, against the baseline reply to another prompt of the same task and language. Fluency:
perplexity of the reply text alone under the unmodified model."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sweep import MODELS, load_kwargs

TASKS = ["monolingual", "crosslingual"]


@torch.no_grad()
def perplexities(model, tok, texts, bs):
    tok.padding_side = "right"
    out = []
    for s in range(0, len(texts), bs):
        b = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        logits = model(**b).logits[:, :-1].float()
        target, mask = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].float()
        lp = torch.log_softmax(logits, -1).gather(-1, target[..., None])[..., 0]
        nll = -(lp * mask).sum(1) / mask.sum(1).clamp(min=1)
        out += torch.exp(nll).tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--dir", default=None)
    p.add_argument("--encoder", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--conds", default=None, help="comma separated, for smoke tests")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--dtype", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    d = Path(a.dir or f"out/{a.model}-steer")
    rows = defaultdict(list)
    for line in open(d / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["cond"]].append(r)
    conds = ["base"] + [c for c in (a.conds.split(",") if a.conds else rows) if c != "base"]
    base = rows["base"]
    non_en = {t: [i for i, r in enumerate(base) if r["task"] == t and r["language"] != "en"] for t in TASKS}
    # another prompt of the same task and language, fixed once for every condition
    rng = random.Random(0)
    groups = defaultdict(list)
    for i, r in enumerate(base):
        groups[(r["task"], r["language"])].append(i)
    other = {}
    for g in groups.values():
        for i in g:
            pool = [j for j in g if j != i]
            other[i] = rng.choice(pool) if pool else None

    enc = SentenceTransformer(a.encoder, device=a.device)
    emb = {c: enc.encode([r["text"] for r in rows[c]], batch_size=a.bs, normalize_embeddings=True,
                         convert_to_numpy=True) for c in conds}
    del enc
    torch.cuda.empty_cache()

    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, a.dtype or dtype), **load_kwargs(a.model))
    model = model.to(a.device).eval()
    ppl = {c: perplexities(model, tok, [r["text"] for r in rows[c]], a.bs) for c in conds}

    def scored(c, i):
        return not rows[c][i]["skipped"]

    res = {}
    for c in conds:
        e, eb = emb[c], emb["base"]
        res[c] = {}
        for t in TASKS:
            keep = [i for i in non_en[t] if scored(c, i) and scored("base", i) and other[i] is not None]
            same = np.array([float(e[i] @ eb[i]) for i in keep])
            diff = np.array([float(e[i] @ eb[other[i]]) for i in keep])
            flip = [k for k, i in enumerate(keep) if rows[c][i]["pass"] != base[i]["pass"]]
            fixed = [k for k, i in enumerate(keep) if rows[c][i]["pass"] and not base[i]["pass"]]
            ok = [i for i in non_en[t] if scored(c, i) and rows[c][i]["pass"]]
            res[c][t] = {
                "n": len(keep), "same_prompt": float(same.mean()) if keep else None,
                "other_prompt": float(diff.mean()) if keep else None,
                "same_above_other": float((same > diff).mean()) if keep else None,
                "n_flipped": len(flip), "flipped_same_prompt": float(same[flip].mean()) if flip else None,
                "flipped_other_prompt": float(diff[flip].mean()) if flip else None,
                "n_fixed": len(fixed), "fixed_same_prompt": float(same[fixed].mean()) if fixed else None,
                "ppl_median": float(np.median([ppl[c][i] for i in non_en[t] if scored(c, i)])),
                "ppl_median_expected_language": float(np.median([ppl[c][i] for i in ok])) if ok else None}
    json.dump(res, open(d / "quality.json", "w"), indent=1)

    f = lambda x: "" if x is None else f"{x:.3f}"
    g = lambda x: "" if x is None else f"{x:.1f}"
    lines = [f"# {a.model}: quality of the steered replies", "",
             f"Content: cosine similarity ({a.encoder}) between a reply and the baseline reply to the same prompt, "
             "against the baseline reply to another prompt of the same task and language (one fixed pairing for "
             "every condition). 'flipped' = prompts whose pass/fail changed against the baseline, 'fixed' = fail to "
             "pass. Fluency: perplexity of the reply text alone under the unmodified model, median over the scored "
             "non-English replies and over those entirely in the expected language.", "",
             "| condition | task | same prompt | other prompt | same > other | flipped: same / other (n) | "
             "fixed: same (n) | PPL, all | PPL, in expected language |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        for t in TASKS:
            r = res[c][t]
            lines.append(f"| {c} | {t} | {f(r['same_prompt'])} | {f(r['other_prompt'])} | {f(r['same_above_other'])} "
                         f"| {f(r['flipped_same_prompt'])} / {f(r['flipped_other_prompt'])} ({r['n_flipped']}) "
                         f"| {f(r['fixed_same_prompt'])} ({r['n_fixed']}) | {g(r['ppl_median'])} "
                         f"| {g(r['ppl_median_expected_language'])} |")
    (d / "quality.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
