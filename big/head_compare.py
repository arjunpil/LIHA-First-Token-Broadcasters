"""What one head reads and writes on the LCB prompts, so heads can be compared across models.

For every LCB prompt (both tasks) the base reply saved by lcb.py is appended (teacher forcing) and the first K reply
positions plus the last prompt position are used as queries. Saved to --out:
  attn.npz     per head of every layer: attention mass on the target-language name in crosslingual prompts
               ("Reply in Japanese" -> the tokens of "Japanese"), mass on the first token (sink), uniform baseline
  head.npz     the chosen head's output (o_proj input slice) averaged over the query positions, per prompt,
               for every head of its layer (probe and controls)
  report.json  model-wide rank of the head on language-name attention, probe accuracy (requested language from the
               head output, 5-fold CV) for each head of the layer, top tokens the head writes per language

Usage: python big/head_compare.py --model qwen3-8b-instruct --head L24H27 --lcb-samples out/qwen3-8b-instruct-lcb
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "big"), str(ROOT / "experiments"), str(ROOT / "prompts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import run_big  # noqa: E402,F401  (adds the larger Qwen keys to sweep.MODELS)
from sweep import MODELS, as_prompts, blocks  # noqa: E402

NAMES = {"ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French", "hi": "Hindi", "id": "Indonesian",
         "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish",
         "vi": "Vietnamese", "zh": "Chinese"}


def parse(h):
    l, hh = h[1:].split("H")
    return int(l), int(hh)


def probe_acc(X, y, folds=5, seed=0, steps=300):
    """Multinomial logistic regression, standardized features, k-fold CV accuracy."""
    X = torch.tensor(X, dtype=torch.float32, device="cuda")
    classes = sorted(set(y))
    yy = torch.tensor([classes.index(v) for v in y], device="cuda")
    perm = torch.randperm(len(yy), generator=torch.Generator().manual_seed(seed)).cuda()
    accs = []
    for f in range(folds):
        te = perm[f::folds]
        tr = perm[torch.isin(perm, te, invert=True)]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        W = torch.zeros(X.shape[1], len(classes), device="cuda", requires_grad=True)
        b = torch.zeros(len(classes), device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, yy[tr]) + 1e-3 * (W ** 2).sum()
            loss.backward()
            opt.step()
        accs.append(((Xte @ W + b).argmax(1) == yy[te]).float().mean().item())
    return float(np.mean(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--lcb-samples", required=True, help="lcb.py output folder with samples.jsonl (base replies)")
    p.add_argument("--k", type=int, default=16, help="reply positions used as queries")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    L0, H0 = parse(a.head)
    out = Path(a.out or f"out/{a.model}-headcmp-{a.head}")
    out.mkdir(parents=True, exist_ok=True)

    items = []
    for line in open(Path(a.lcb_samples) / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["cond"] != "base":
            break
        items.append(r)
    name, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), attn_implementation="eager")
    model = model.cuda().eval()
    cfg = model.config.get_text_config()
    NL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H

    prompts = as_prompts(tok, a.model, items)
    name_mass = np.zeros((NL, H)); sink_mass = np.zeros((NL, H)); n_cross = 0; uniform = []
    z_layer = np.zeros((len(items), H, dh), dtype=np.float32)
    cur = {}

    def attn_hook(l):
        def f(m, args, kwargs, output):
            w = output[1]  # B, H, T, T (eager)
            q, nm = cur["q"], cur["name"]  # q: B, T bool query mask; nm: B, T bool name-token mask
            qw = w.float() * q[:, None, :, None]
            nq = q.sum(1).clamp(min=1)[:, None]  # B, 1
            name_mass[l] += ((qw * nm[:, None, None, :]).sum((2, 3)) / nq)[cur["cross"]].sum(0).cpu().numpy()
            sink_mass[l] += ((qw * cur["fmask"][:, None, None, :]).sum((2, 3)) / nq)[cur["cross"]].sum(0).cpu().numpy()
        return f

    def z_hook(m, args):
        x = args[0].float()  # B, T, H*dh
        q = cur["q"]
        zm = (x * q[..., None]).sum(1) / q.sum(1, keepdim=True)  # B, H*dh
        z_layer[cur["idx"]] = zm.view(len(cur["idx"]), H, dh).cpu().numpy()

    hs = [blocks(model, a.model)[l][0].register_forward_hook(attn_hook(l), with_kwargs=True) for l in range(NL)]
    hs.append(blocks(model, a.model)[L0][1].register_forward_pre_hook(z_hook))
    tok.padding_side = "left"
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    with torch.no_grad():
        for s in range(0, len(order), a.bs):
            idx = order[s:s + a.bs]
            texts, P, spans = [], [], []
            for i in idx:
                reply_ids = tok(items[i]["text"], add_special_tokens=False).input_ids[:a.k]
                full = prompts[i] + tok.decode(reply_ids)
                texts.append(full)
                P.append(len(tok(prompts[i], add_special_tokens=False).input_ids))
                spans.append(None)
                if items[i]["task"] == "crosslingual":
                    nm = NAMES[items[i]["language"]]
                    st = prompts[i].rfind(nm)  # the user turn; the system turn never names a language
                    spans[-1] = (st, st + len(nm)) if st >= 0 else None
            b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
            off = b.pop("offset_mapping")
            T = b["input_ids"].shape[1]
            lens = b["attention_mask"].sum(1)
            q = torch.zeros(len(idx), T, dtype=torch.bool)
            nmask = torch.zeros(len(idx), T, dtype=torch.bool)
            first = []
            for j, i in enumerate(idx):
                pad = T - lens[j].item()
                first.append(pad)
                q[j, pad + P[j] - 1: T] = True  # last prompt token and the reply tokens
                if spans[j]:
                    st, en = spans[j]
                    for t in range(pad, pad + P[j]):
                        o0, o1 = off[j, t].tolist()
                        if o1 > st and o0 < en:
                            nmask[j, t] = True
            cross = torch.tensor([spans[j] is not None for j in range(len(idx))])
            uniform += [nmask[j].sum().item() / P[j] for j in range(len(idx)) if spans[j]]
            n_cross += cross.sum().item()
            fmask = torch.zeros(len(idx), T)
            fmask[torch.arange(len(idx)), torch.tensor(first)] = 1
            cur.update(q=q.cuda(), name=nmask.cuda().float(), cross=cross.cuda(), fmask=fmask.cuda(), idx=idx)
            model(**{k: v.cuda() for k, v in b.items()})
            if s // a.bs % 50 == 0:
                print(f"{s}/{len(order)}", flush=True)
    for h in hs:
        h.remove()
    name_mass /= max(n_cross, 1); sink_mass /= max(n_cross, 1)
    np.savez(out / "attn.npz", name_mass=name_mass, sink_mass=sink_mass, uniform=np.mean(uniform))
    langs = [it["language"] for it in items]
    tasks = [it["task"] for it in items]
    np.savez_compressed(out / "head.npz", z=z_layer, langs=langs, tasks=tasks)

    flat = name_mass.ravel()
    rank = int((flat > name_mass[L0, H0]).sum()) + 1
    report = {"model": a.model, "head": a.head, "layers": NL, "heads": H, "head_dim": dh, "n_cross": n_cross,
              "uniform_name_mass": float(np.mean(uniform)),
              "name_mass": float(name_mass[L0, H0]), "name_mass_rank": rank, "n_heads_total": int(flat.size),
              "name_mass_layer": name_mass[L0].tolist(), "sink_mass_layer": sink_mass[L0].tolist(),
              "top_name_heads": [(f"L{i // H}H{i % H}", float(flat[i])) for i in np.argsort(-flat)[:15]]}

    cross = [i for i, t in enumerate(tasks) if t == "crosslingual"]
    mono = [i for i, t in enumerate(tasks) if t == "monolingual" and langs[i] != "en"]
    report["probe_cross"] = [probe_acc(z_layer[cross, h], [langs[i] for i in cross]) for h in range(H)]
    report["probe_mono"] = [probe_acc(z_layer[mono, h], [langs[i] for i in mono]) for h in range(H)]

    # what the head writes: language-mean of its output (crosslingual), centred, through o_proj, final norm, unembedding
    W_O = blocks(model, a.model)[L0][1].weight[:, H0 * dh:(H0 + 1) * dh].float()
    norm_w = model.model.norm.weight.float()
    W_U = model.get_output_embeddings().weight.float()
    zc = torch.tensor(z_layer[cross, H0], device="cuda")
    allm = zc.mean(0)
    top = {}
    means = {}
    for lang in sorted(set(langs[i] for i in cross)):
        sel = torch.tensor([langs[i] == lang for i in cross], device="cuda")
        m = zc[sel].mean(0)
        means[lang] = (m - allm).cpu().numpy()
        logits = W_U @ (norm_w * (W_O @ (m - allm)))
        top[lang] = [tok.decode([t]) for t in logits.topk(15).indices.tolist()]
    report["top_tokens_cross"] = top
    np.savez(out / "lang_means.npz", **means)
    json.dump(report, open(out / "report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("name_mass_layer", "sink_mass_layer")},
                     ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
