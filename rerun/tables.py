import csv
import json
from pathlib import Path
from statistics import mean, stdev

from analyze import same

R = Path("results")
EU = ["en", "fr", "de", "es", "it"]
MODELS = [("gpt2", "GPT-2", None), ("bloom", "BLOOM-1b7", None), ("qwen-base", "Qwen-1.5B Base", 25),
          ("qwen-instruct", "Qwen-1.5B Instruct", 25)]


def load(name):
    if not (R / name / "summary.json").exists():
        return None, None
    return json.load(open(R / name / "summary.json"))["modes"]["head"]["table"], json.load(open(R / name / "labels.json"))


def languages(path, per_lang=None):
    seen, out = {}, []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        seen[r["language"]] = seen.get(r["language"], 0) + 1
        if per_lang is None or seen[r["language"]] <= per_lang:
            out.append(r["language"])
    return out


def rates(lab, head, exp, langs=None):
    b, a = lab["base"]["labels"], lab["head:" + head]["labels"]
    idx = [i for i, e in enumerate(exp) if langs is None or e in langs]
    return {"sr": mean(int(a[i] != b[i]) for i in idx),
            "c2w": mean(int(same(b[i], exp[i]) and not same(a[i], exp[i])) for i in idx),
            "c2other": mean(int(same(b[i], exp[i]) and not same(a[i], exp[i]) and a[i] != "unknown") for i in idx)}


def ranked(t, key, max_dnll=float("inf")):
    return sorted((h for h in t if t[h]["dnll"] <= max_dnll), key=lambda h: -t[h]["full"][key])


def f3(x):
    return f"{x:.3f}"


def signed(x):
    return f"{x:+.3f}"


def tex(c):
    for a, b in (("#", "\\#"), ("→", "$\\to$"), ("Δ", "$\\Delta$"), ("σ", "$\\sigma$"), ("≤", "$\\le$"), (">", "$>$")):
        c = c.replace(a, b)
    if c[:1] in "+-" and c[1:2].isdigit():
        c = f"${c[0]}${c[1:]}"
    return "{" + c + "}" if c.startswith("[") else c


def block(title, note, header, rows):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    tex_rows = [" & ".join(map(tex, r)) + " \\\\" for r in rows]
    return [f"## {title}", "", note, ""] + md + ["", "```latex"] + tex_rows + ["```", ""]


def head_rows(t, heads):
    return [[h, f3(t[h]["full"]["sr"]), "[{:.3f}, {:.3f}]".format(*t[h]["full"]["sr_ci"]), f3(t[h]["full"]["c2w"]),
             f3(t[h]["full"]["w2c"]), signed(t[h]["dnll"])] for h in heads]


def table1(t):
    header = ["Head", "SR", "95% CI", "c→w", "w→c", "ΔNLL"]
    mean_path = R / "gpt2-mean-ablation" / "summary.json"
    mean_t = json.load(open(mean_path))["table"] if mean_path.exists() else {}
    mu = mean(v["full"]["sr"] for v in t.values())
    out = block("Table 1a: top five GPT-2 heads by switch rate",
                f"Same selection rule as the paper, 2,500 prompts. Population mean SR {mu:.3f}. Only L2H5 mostly flips "
                "correct→wrong; the rest flip wrong→correct about as often or more, and L0H10 / L0H0 raise LM loss by "
                "more than 1 nat.", header, head_rows(t, ranked(t, "sr")[:5]))
    heads = ranked(t, "c2w", 0.1)[:5] + ["L6H1"]
    rows = [r + [f3(mean_t[h]["c2w"]) if h in mean_t else "–"] for r, h in zip(head_rows(t, heads), heads)]
    out += block("Table 1b: top five GPT-2 heads by correct→wrong, ΔNLL ≤ 0.1",
                 "Suggested replacement: heads that move correct outputs to the wrong language without much LM loss "
                 "change. L6H1, the paper's top head, for comparison. The last column is c→w under mean ablation "
                 "(Chaewon, results/gpt2-mean-ablation): only L6H10 keeps most of its effect.",
                 header + ["c→w, mean abl."], rows)
    return out


def table2():
    cols, rows, en_note = [], {k: [] for k in ("swept", "prompts", "max", "sigma", "n01", "n2sd", "layers", "c2w",
                                               "c2w_low", "c2other", "en")}, ""
    for key, name, per_lang in MODELS:
        t, lab = load(key)
        cols.append(name)
        if t is None:
            for k in rows:
                rows[k].append("pending")
            continue
        exp = languages("prompts_european.csv", per_lang)
        sr = {h: v["full"]["sr"] for h, v in t.items()}
        mu, sd = mean(sr.values()), stdev(sr.values())
        top = max(sr, key=sr.get)
        big = [h for h in sr if sr[h] > mu + 2 * sd]
        layers = sorted({int(h[1:].split("H")[0]) for h in big})
        c, cl = ranked(t, "c2w")[0], ranked(t, "c2w", 0.1)[0]
        en = sorted(((rates(lab, h, exp, ["en"])["c2other"], h) for h in t), reverse=True)
        other = max((rates(lab, h, exp)["c2other"], h) for h in t)
        rows["swept"].append(str(len(t)))
        rows["prompts"].append(f"{len(exp):,}")
        rows["max"].append(f"{sr[top]:.3f} ({top})")
        rows["sigma"].append(f"{(sr[top] - mu) / sd:.2f}")
        rows["n01"].append(str(sum(v > 0.1 for v in sr.values())))
        rows["n2sd"].append(str(len(big)))
        rows["layers"].append(", ".join(map(str, layers)) or "none")
        rows["c2w"].append(f"{c} {t[c]['full']['c2w']:.3f} / {t[c]['dnll']:+.3f}")
        rows["c2w_low"].append(f"{cl} {t[cl]['full']['c2w']:.3f} / {t[cl]['dnll']:+.3f}")
        rows["c2other"].append(f"{other[1]} {other[0]:.3f} / {t[other[1]]['dnll']:+.3f}")
        rows["en"].append("none" if en[0][0] == 0 else f"{en[0][0]:.3f} ({en[0][1]})")
        n_en = exp.count("en")
        if key == "gpt2":
            en_note += (f" In GPT-2 only {en[0][1]} flips English prompts, and its outputs degenerate (ΔNLL "
                        f"{t[en[0][1]]['dnll']:+.2f}); the next head is at {en[1][0]:.3f}.")
        elif en[0][0] > 0:
            en_note += f" In {name} that is {round(en[0][0] * n_en)} of {n_en} English prompts."
    labels = {"swept": "Heads swept", "prompts": "Prompts", "max": "Max SR (head)", "sigma": "Top head σ",
              "n01": "# heads SR > 0.1", "n2sd": "# heads SR > mean + 2 sd", "layers": "Layers of those heads",
              "c2w": "Top c→w head (c→w / ΔNLL)", "c2w_low": "Top c→w head, ΔNLL ≤ 0.1",
              "c2other": "Top head switching to another language",
              "en": "English → another language, max over heads"}
    note = ("σ is (max − mean) / sd over heads, as in the paper. GPT-2 and BLOOM use the 2,500 prompts, Qwen the "
            "paper's 125, so SR > 0.1 counts aren't comparable across columns (GPT-2's mean SR is already about "
            "0.11). The mean + 2 sd rows and the c→w rows are suggested replacements. \"Switching to another language\" "
            "counts c→w only when the new output is detected as some language, not empty or unknown; BLOOM's top c→w "
            "head makes the model stop right away instead of switching." + en_note)
    return block("Table 2: cross-model comparison", note, ["Property"] + cols,
                 [[labels[k]] + v for k, v in rows.items()])


def table3(t, lab):
    exp = languages("prompts_european.csv")
    base_acc = mean(int(same(b, e)) for b, e in zip(lab["base"]["labels"], exp))
    heads = [h for h in ranked(t, "sr") if t[h]["full"]["sr"] > 0.15]
    rows = [[h, f3(t[h]["full"]["sr"]), signed(t[h]["full"]["acc"] - base_acc), f3(t[h]["full"]["c2w"]),
             f3(t[h]["full"]["w2c"]), signed(t[h]["dnll"])] for h in heads]
    return block("Table 3: all GPT-2 heads with switch rate > 0.15",
                 f"{len(heads)} heads, 2,500 prompts. Δacc is against the baseline accuracy {base_acc:.3f}.",
                 ["Head", "SR", "Δacc", "c→w", "w→c", "ΔNLL"], rows)


def table5(t, lab):
    exp = languages("prompts_european.csv")
    heads = ranked(t, "c2w", 0.1)[:5] + ["L6H1"]
    rows = [[h] + [f"{rates(lab, h, exp, [l])['sr']:.2f}" for l in EU] for h in heads]
    base = lab["base"]["labels"]
    acc = [f"{mean(int(same(base[i], l)) for i, e in enumerate(exp) if e == l):.2f}" for l in EU]
    return block("Table 5: per-language switch rates",
                 "Heads from Table 1b, 500 prompts per language instead of the 5 hand-written ones. Baseline "
                 f"accuracy per language: {', '.join(f'{l} {a}' for l, a in zip(EU, acc))}. Under L6H10 the "
                 "es/de/it switches go to English.",
                 ["Head"] + [l.upper() for l in EU], rows)


def table7():
    t, _ = load("qwen-instruct")
    sr = [v["full"]["sr"] for v in t.values()]
    mu, sd = mean(sr), stdev(sr)
    top = ranked(t, "sr")
    rows = [[h, f3(t[h]["full"]["sr"]), "[{:.3f}, {:.3f}]".format(*t[h]["full"]["sr_ci"]),
             f"{(t[h]['full']['sr'] - mu) / sd:.2f}", f3(t[h]["full"]["c2w"]), signed(t[h]["dnll"])] for h in top[:5]]
    ties = [h for h in top[5:] if t[h]["full"]["sr"] == t[top[4]]["full"]["sr"]]
    return block("Table 7: top five Qwen2.5-1.5B-Instruct heads by switch rate",
                 f"125 prompts. Population mean SR {mu:.3f}." + (f" {', '.join(ties)} ties with {top[4]}." if ties else "") +
                 f" The paper's L0H5 is {t['L0H5']['full']['sr']:.3f}.",
                 ["Head", "SR", "95% CI", "σ above mean", "c→w", "ΔNLL"], rows)


def table8(t_eu):
    t, lab = load("gpt2-zhru")
    exp = languages("prompts_extended.csv")
    left = ["L6H1", "L0H4", "L3H1", "L9H9", "L1H10"]
    rows = [[h, f"{t_eu[h]['full']['sr']:.2f}", f"{rates(lab, h, exp, ['zh'])['sr']:.2f}",
             f"{rates(lab, h, exp, ['ru'])['sr']:.2f}"] for h in left]
    out = block("Table 8 left: the paper's European heads on zh/ru (switch rate)",
                "Not zero under the fixed hook. EU is the 2,500-prompt SR, ZH/RU are 100 prompts each.",
                ["Head", "EU", "ZH", "RU"], rows)
    right = ranked(t, "c2w")[:5]
    rows = [[h, f"{t[h]['full']['c2w']:.2f}", f"{rates(lab, h, exp, ['zh'])['c2w']:.2f}",
             f"{rates(lab, h, exp, ['ru'])['c2w']:.2f}", f"{t_eu[h]['full']['c2w']:.2f}"] for h in right]
    top_sr = ", ".join(f"{h} {t[h]['full']['sr']:.2f}" for h in ranked(t, "sr")[:3])
    out += block("Table 8 right: top zh/ru heads by correct→wrong",
                 f"c→w on zh/ru with the European c→w for comparison. By raw SR the top heads are {top_sr}, mostly "
                 "wrong→correct on zh where GPT-2 repeats Chinese characters from the prompt.",
                 ["Head", "All", "ZH", "RU", "EU"], rows)
    return out


def main():
    t, lab = load("gpt2")
    lines = ["# Paper tables under the fixed hook", "",
             "Generated by tables.py from results/. Columns marked pending fill in once those results are added.",
             "c→w and w→c are correct→wrong and wrong→correct switch rates; ΔNLL is the LM loss change on FLORES dev.",
             ""]
    lines += table1(t) + table2() + table3(t, lab) + table5(t, lab) + table7() + table8(t)
    Path("TABLES.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
