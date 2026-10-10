import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze import same
from checks import controls
from multi import parse

# drawn at the ACL column width (7.7 cm), so included at \columnwidth the text prints at 8 to 9 pt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "figure.constrained_layout.use": True})
WIDTH = 3.0
MAIN = ["L6H10", "L2H5", "L4H8", "L2H3", "L8H6"]
OLD = ["L6H1", "L0H4", "L9H9"]
PAIR = ["Le temps aujourd'hui est très", "La cosa più importante nella vita è"]


def entropy(a):
    return float(-(a * torch.log(a + 1e-9)).sum())


@torch.no_grad()
def collect(model, tok, prompts, heads, steps):
    hs = [parse(h) for h in heads]
    first, ent, prompt_rows = [], [], []
    for p in prompts:
        b = tok(p, return_tensors="pt").to(model.device)
        out = model.generate(**b, max_new_tokens=steps, do_sample=False, output_attentions=True,
                             return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        f = np.full((len(heads), steps), np.nan)
        e = np.full((len(heads), steps), np.nan)
        for t, att in enumerate(out.attentions):
            for j, (l, h) in enumerate(hs):
                a = att[l][0, h, -1].float()
                f[j, t], e[j, t] = float(a[0]), entropy(a)
        l, h = hs[0]
        prompt_rows.append(out.attentions[0][l][0, h, 1:, 0].float().cpu().tolist())
        first.append(f)
        ent.append(e)
    return np.stack(first), np.stack(ent), prompt_rows


@torch.no_grad()
def probe(model, tok, prompts, langs, bs=100):
    tok.padding_side = "left"
    feats = None
    for i in range(0, len(prompts), bs):
        b = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        hidden = model(**b, output_hidden_states=True).hidden_states
        x = [hs[:, -1].float().cpu().numpy() for hs in hidden]
        feats = x if feats is None else [np.concatenate([a, c]) for a, c in zip(feats, x)]
    y = np.array(langs)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return [float(cross_val_score(clf, f, y, cv=5).mean()) for f in feats]


def labels(tok, ids):
    out, buf = [], []
    for i in ids:
        buf.append(i)
        text = tok.decode(buf)
        out.append("·" if "\ufffd" in text else text.strip())
        if "\ufffd" not in text:
            buf = []
    return out


def heatmap(model, tok, head, path):
    l, h = parse(head)
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.4))
    for ax, p, title in zip(axes, PAIR, ["French, stays French", "Italian, switches"]):
        b = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            a = model(**b, output_attentions=True).attentions[l][0, h].float().cpu().numpy()
        toks = labels(tok, b["input_ids"][0].tolist())
        ax.imshow(a, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=60, ha="right")
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels(toks)
        ax.set_title(f"{head}, {title}")
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def entropy_plot(ent, heads, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    x = np.arange(1, ent.shape[2] + 1)
    for j, h in enumerate(heads):
        y = np.nanmean(ent[:, j], 0)
        if h == "L6H10":
            ax.plot(x, y, color="#2a7fa0", lw=2, label=h)
        elif h in MAIN:
            ax.plot(x, y, color="#2a7fa0", lw=0.8, alpha=0.6, label="other c→w heads" if h == MAIN[1] else None)
        elif h in OLD:
            ax.plot(x, y, color="#d9822b", lw=1, ls="-.", label="paper's heads" if h == OLD[0] else None)
        else:
            ax.plot(x, y, color="gray", lw=1, ls="--", label="random heads" if h == heads[-3] else None)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Attention entropy")
    fig.legend(loc="outside upper center", ncol=2, handlelength=1.8, columnspacing=0.8)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=100)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="results/gpt2-attention")
    p.add_argument("--figures-only", action="store_true",
                   help="redraw the two figures and keep summary.json; reuses entropy_steps.npz if it was made "
                        "with the same heads and settings")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8")))
    base = json.load(open("results/gpt2/labels.json"))["base"]["labels"]
    table = json.load(open("results/gpt2/summary.json"))["modes"]["head"]["table"]
    seen, idx = {}, []
    for i, r in enumerate(rows):
        seen[r["language"]] = seen.get(r["language"], 0) + 1
        if seen[r["language"]] <= a.per_lang:
            idx.append(i)
    heads = MAIN + OLD + controls("results/gpt2/summary.json", MAIN + OLD)

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32, attn_implementation="eager").cuda().eval()
    saved = out / "entropy_steps.npz"
    if a.figures_only and saved.exists():
        s = np.load(saved)
        if list(s["heads"]) == heads and int(s["per_lang"]) == a.per_lang and int(s["steps"]) == a.steps:
            heatmap(model, tok, "L6H10", out / "fig_attention_comparison")
            entropy_plot(s["ent"], heads, out / "fig_entropy_over_time")
            return
    first, ent, prompt_rows = collect(model, tok, [rows[i]["prompt"] for i in idx], heads, a.steps)
    np.savez_compressed(saved, ent=ent, heads=np.array(heads), per_lang=a.per_lang, steps=a.steps)
    heatmap(model, tok, "L6H10", out / "fig_attention_comparison")
    entropy_plot(ent, heads, out / "fig_entropy_over_time")
    if a.figures_only:
        return
    acc = probe(model, tok, [r["prompt"] for r in rows], [r["language"] for r in rows])

    non_en = [k for k, i in enumerate(idx) if rows[i]["language"] != "en"]
    ok = [k for k in non_en if same(base[idx[k]], rows[idx[k]]["language"])]
    bad = [k for k in non_en if k not in ok]
    j = heads.index("L6H10")
    pr = np.concatenate([np.array(prompt_rows[k]) for k in non_en if prompt_rows[k]])
    res = {"prompts": len(idx), "heads": heads,
           "first_token": {h: float(np.nanmean(first[:, k])) for k, h in enumerate(heads)},
           "entropy": {h: float(np.nanmean(ent[:, k])) for k, h in enumerate(heads)},
           "l6h10_first_token_correct": float(np.nanmean(first[ok, j])),
           "l6h10_first_token_confused": float(np.nanmean(first[bad, j])),
           "n_correct": len(ok), "n_confused": len(bad),
           "l6h10_prompt_rows_p5_p95": [float(np.percentile(pr, 5)), float(np.percentile(pr, 95))],
           "probing_by_hidden_state": acc}
    json.dump(res, open(out / "summary.json", "w"), indent=1)

    lines = ["# L6H10 versions of the section 5 / appendix B numbers", "",
             f"{len(idx)} prompts ({a.per_lang} per language), greedy, {a.steps} steps. First-token attention and entropy "
             "are at the newest position at each generation step, averaged over steps and prompts.", "",
             "Table 6 replacement:", "", "| head | c->w | first-token attn | entropy |", "|---|---|---|---|"]
    lines += [f"| {h} | {table[h]['full']['c2w']:.3f} | {res['first_token'][h]:.3f} | {res['entropy'][h]:.3f} |"
              for h in heads]
    lines += ["", f"L6H10 first-token attention on non-English prompts: {res['l6h10_first_token_correct']:.3f} when the "
              f"baseline output is in the right language ({len(ok)} prompts), {res['l6h10_first_token_confused']:.3f} "
              f"when it isn't ({len(bad)}). On the prompt itself, attention to the first token across query positions "
              f"is {res['l6h10_prompt_rows_p5_p95'][0]:.2f}-{res['l6h10_prompt_rows_p5_p95'][1]:.2f} (5th-95th percentile).",
              "", "Probing accuracy for the prompt language from the last prompt token, 2,500 prompts, 5-fold logistic "
              "regression, by hidden state (0 = embeddings, k = output of layer k-1):", "",
              "| hidden state | " + " | ".join(map(str, range(len(acc)))) + " |", "|---|" + "---|" * len(acc),
              "| accuracy | " + " | ".join(f"{x:.3f}" for x in acc) + " |"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
