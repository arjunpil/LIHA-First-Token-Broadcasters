"""Main-text tables, built from results/: the head in each instruct model (EXPERIMENTS.md sections 5, 7 and 8) and,
for Qwen2.5-1.5B and Gemma-3-1B, what the head attends to and when it acts (section 6). Run from the repository
root: python tables/main_tables.py. Writes tables/main_tables.md with previews and the LaTeX to paste."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

R = Path("results")
# name, FLORES run, base run, head, LCB run; screen runs are marked, None means not run or no head
MODELS = [
    ("Qwen2.5-1.5B", "qwen-instruct-full", "qwen-base-full", "L22H6", "qwen-instruct-lcb"),
    ("Qwen2.5-3B", "qwen2.5-3b-instruct", "qwen2.5-3b", "L27H13", "qwen2.5-3b-instruct-lcb"),
    ("Qwen2.5-7B", "qwen2.5-7b-instruct", "qwen2.5-7b", "L19H1", "qwen2.5-7b-instruct-lcb"),
    ("Qwen3-1.7B", "qwen3-1.7b-instruct", "qwen3-1.7b", "L18H12", "qwen3-1.7b-instruct-lcb"),
    ("Qwen3-4B", "qwen3-4b-instruct-screen", None, None, None),
    ("Gemma-3-1B", "gemma3-1b-instruct", "gemma3-1b", "L11H3", "gemma3-1b-instruct-lcb"),
    ("Gemma-3-4B", "gemma3-4b-instruct", "gemma3-4b", "L24H0", "gemma3-4b-instruct-lcb"),
    ("OLMo-2-1B", "olmo2-1b-instruct", "olmo2-1b", "L12H8", "olmo2-1b-instruct-lcb"),
    ("OLMo-3-7B", "olmo3-7b-instruct", "olmo3-7b-screen", "L14H25", "olmo3-7b-instruct-lcb"),
    ("Llama-3.2-1B", "llama3.2-1b-instruct-screen", None, "L8H25", "llama3.2-1b-instruct-lcb"),
    ("Llama-3.2-3B", "llama3.2-3b-instruct-screen", None, "L13H19", "llama3.2-3b-instruct-lcb-top"),
    ("SmolLM3-3B", "smollm3-instruct-screen", None, None, None),
]
NOTES = {
    "Qwen3-4B": "none on FLORES (largest 0.024 on the screen)",
    "SmolLM3-3B": "none on FLORES (L1H12 breaks generation, dNLL +2.58)",
}
# where each family's attention block normalizes (transformers' decoder layers): its input (pre), its output before
# the residual stream (post), queries and keys (QK)
NORM = {"Qwen2.5": "pre", "Qwen3": "pre, QK", "Gemma-3": "pre+post, QK", "OLMo-2": "post, QK", "OLMo-3": "post, QK",
        "Llama-3.2": "pre", "SmolLM3": "pre"}


def table(run):
    return json.load(open(R / run / "summary.json"))["modes"]["head"]["table"]


def layer_of(h):
    return h.split("H")[0]


def flores(run, head):
    t = table(run)
    rest = [v["full"]["c2w"] for h, v in t.items() if layer_of(h) == layer_of(head) and h != head]
    return t[head]["full"]["c2w"], max(rest), t[head]["dnll"]


def mean_c2w(run, head):
    # c->w with the head replaced by its mean over prompt tokens, from the follow-up's summary
    lines = [line for line in open(R / f"{run.removesuffix('-full')}-followup" / "summary.md", encoding="utf-8")
             if line.startswith("|")]
    col = [c.strip() for c in lines[0].strip(" |\n").split("|")].index("c->w")
    return float(next(line for line in lines if line.startswith(f"| {head}:mean |")).strip(" |\n").split("|")[col])


def lcb(run, head):
    row = json.load(open(R / run / "summary.json"))["rows"][f"{head} zero"]
    return row["monolingual_delta"], row["crosslingual_delta"]


def rows():
    out = []
    for name, run, base, head, lcb_run in MODELS:
        screen = run.endswith("-screen")
        r = {"model": name, "heads": max(int(h.split("H")[1]) for h in table(run)) + 1,
             "norm": NORM[name.rsplit("-", 1)[0]]}
        if head is None:
            out.append({**r, "note": NOTES[name]})
            continue
        c2w, rest, dnll = flores(run, head)
        r.update(head=head, c2w=c2w, rest=rest, dnll=dnll, screen=screen, mean=None if screen else mean_c2w(run, head))
        if base:
            r["base"] = table(base)[head]["full"]["c2w"]
            r["base_screen"] = base.endswith("-screen")
        r["mono"], r["cross"] = lcb(lcb_run, head)
        out.append(r)
    return out


def num(x, signed=False):
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "$-$").replace("+", "$+$") if signed else s


def delta_tex(d):
    s = num(d[0], signed=True)
    return s if d[2] < 0 or d[1] > 0 else rf"\textcolor{{gray}}{{{s}}}"


def tex(rs):
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lcllccccccc@{}}", r"\toprule",
             r" & & & & \multicolumn{4}{c}{$C\to W$} & & \multicolumn{2}{c}{LCB} \\",
             r"\cmidrule(lr){5-8}\cmidrule(lr){10-11}",
             r"Model & Heads & Norm & Head & Zero & Mean & Others & Base & $\Delta$NLL & Mono & Cross \\", r"\midrule"]
    for r in rs:
        lead = f"{r['model']} & {r['heads']} & {r['norm']}"
        if "note" in r:
            lines.append(rf"{lead} & \multicolumn{{8}}{{l}}{{{r['note']}}} \\")
            continue
        s = r"$^{s}$" if r["screen"] else ""
        mean_ = "--" if r["mean"] is None else num(r["mean"])
        base = num(r["base"]) + (r"$^{s}$" if r.get("base_screen") else "") if "base" in r else "--"
        lines.append(f"{lead} & {r['head']} & {num(r['c2w'])}{s} & {mean_} & {num(r['rest'])}{s} & {base} & "
                     f"{num(r['dnll'], signed=True)} & {delta_tex(r['mono'])} & {delta_tex(r['cross'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The head in each instruct model. Heads: heads per layer. Norm: where the attention block "
              r"normalizes, on its input (pre), on its output before the residual stream (post), and on queries and "
              r"keys (QK). $C\to W$: share of the 2,500 FLORES prompts whose continuation is in the prompt language at "
              r"baseline and in another language with the head zeroed or replaced by its mean output; others: the "
              r"largest zero value among the other heads of its layer; base: the same head zeroed in the base model. "
              r"LCB: paired change in line-level pass rate with the head zeroed, five languages; gray where the 95\% "
              r"CI includes zero. $^{s}$: 125-prompt screen. The Llama heads come from a screen of every head on "
              r"crosslingual LCB prompts. Heads per layer and normalization differ, so each head is compared with the "
              r"other heads of its own layer, not with the heads of other models.}",
              r"\label{tab:models}", r"\end{table*}"]
    return "\n".join(lines)


def preview(rs):
    lines = ["| model | heads | norm | head | c→w zero | c→w mean | others | base | ΔNLL | LCB mono | LCB cross |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        lead = f"| {r['model']} | {r['heads']} | {r['norm']}"
        if "note" in r:
            lines.append(f"{lead} | {r['note']} | | | | | | | |")
            continue
        s = " (screen)" if r["screen"] else ""
        mean_ = "not run" if r["mean"] is None else f"{r['mean']:.3f}"
        base = f"{r['base']:.3f}" + (" (screen)" if r.get("base_screen") else "") if "base" in r else "not run"
        d = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
        lines.append(f"{lead} | {r['head']} | {r['c2w']:.3f}{s} | {mean_} | {r['rest']:.3f}{s} | {base} | "
                     f"{r['dnll']:+.3f} | {d(r['mono'])} | {d(r['cross'])} |")
    return "\n".join(lines)


QWEN, GEMMA = R / "qwen-l22h6-mechanism", R / "gemma-l11h3-mechanism"
# the runs behind the three columns (Qwen2.5-1.5B cross, Qwen2.5-1.5B mono, Gemma-3-1B cross) and how their rows
# are keyed to prompts; Qwen2.5-1.5B cross has its zeroing and its masking in two runs with the same clean replies
ZERO_RUNS = [(QWEN / "liha_l22h6_phase_validation.csv", ("language", "source", "prompt")),
             (QWEN / "monolingual_phase_outputs.csv", ("prompt_id",)),
             (GEMMA / "condition_outputs.csv", ("prompt_id",))]
MASK_RUNS = [(QWEN / "liha_l22h6_language_edge_validation.csv", ("language", "source", "prompt")), None,
             ZERO_RUNS[2]]
# row label and its condition in each column; None where it was not run
ZEROED = [("Head, prompt only", "L22H6_prefill", "L22H6_prefill", None),
          ("Head, generation only", "L22H6_decode", "L22H6_decode", "head_zero_decode"),
          ("Head, both", "L22H6_all", "L22H6_all", None),
          ("Control head, both", "L22H8_control", "L22H8_control", None)]
MASKED = [("Head to the name", "target_language", None, "target_word_mask"),
          ("Head to nearby tokens", "target_nearby", None, "nearby_word_mask"),
          ("Control head to the name", "control_language", None, "control_head_word_mask")]


def outcomes(run):
    if run is None:
        return None
    path, key = run
    by = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by[tuple(r[k] for k in key)][r["condition"]] = r
    return list(by.values())


def passing(prompts):
    return [p for p in prompts if p["clean"]["skipped"] == "False" and p["clean"]["passed"] == "True"]


def fails(prompts, cond):
    # of the replies that pass the line check with no intervention, those that fail under cond
    return sum(p[cond]["skipped"] == "False" and p[cond]["passed"] == "False" for p in passing(prompts))


def gains(prompts, cond):
    # replies that fail with no intervention and pass under cond
    return sum(p["clean"]["passed"] == "False" and p["clean"]["skipped"] == "False" and p[cond]["passed"] == "True"
               for p in prompts)


def mechanism():
    zero, mask = [outcomes(r) for r in ZERO_RUNS], [outcomes(r) for r in MASK_RUNS]
    assert len(passing(zero[0])) == len(passing(mask[0]))
    q = list(csv.DictReader(open(QWEN / "liha_l22h6_attention_analysis.csv", encoding="utf-8")))
    g = list(csv.DictReader(open(GEMMA / "attention_by_prompt.csv", encoding="utf-8")))

    def att(rows, *cols):
        return mean(mean(float(r[c]) for c in cols) for r in rows)

    # the Qwen2.5-1.5B runner stores the mean over L22H4, H8 and H9; Gemma-3-1B's layer has three other heads
    attention = [("Head to the name", att(q, "last_prompt_target_lang"), att(g, "last_prompt_h3_target_mass")),
                 ("Head to nearby tokens", att(q, "last_prompt_target_nearby"), att(g, "last_prompt_h3_nearby_mass")),
                 ("Other heads to the name", att(q, "last_prompt_control_lang_mean"),
                  att(g, *(f"last_prompt_h{h}_target_mass" for h in (0, 1, 2))))]

    def counts(spec, runs):
        return [(label, [None if c is None else fails(p, c) for p, c in zip(runs, conds)]) for label, *conds in spec]

    columns = ("Qwen2.5-1.5B cross", "Qwen2.5-1.5B mono", "Gemma-3-1B cross")
    gained = [(f"{col}, {kind} {label.lower()}", n) for kind, spec, runs in (("zeroed:", ZEROED, zero),
                                                                             ("masked:", MASKED, mask))
              for label, *conds in spec for col, p, c in zip(columns, runs, conds) if c and (n := gains(p, c))]
    return {"attention": attention, "passing": [len(passing(p)) for p in zero],
            "scorable": [sum(p["clean"]["skipped"] == "False" for p in r) for r in zero],
            "zeroed": counts(ZEROED, zero), "masked": counts(MASKED, mask), "gains": gained}


def mech_tex(m):
    cell = lambda v: "--" if v is None else str(v)
    group = lambda s: rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{s}}}}} \\"
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Qwen2.5-1.5B} & Gemma-3-1B \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
             r" & Cross & Mono & Cross \\", r"\midrule", group("Attention from the last prompt token")]
    lines += [f"{label} & {qv:.3f} & -- & {gv:.3f} \\\\" for label, qv, gv in m["attention"]]
    lines += [r"\midrule", "Pass without intervention & " + " & ".join(map(str, m["passing"])) + r" \\",
              group("Of these, fail when zeroed")]
    lines += [f"{label} & " + " & ".join(map(cell, v)) + r" \\" for label, v in m["zeroed"]]
    lines.append(group("Of these, fail when masked in generation"))
    lines += [f"{label} & " + " & ".join(map(cell, v)) + r" \\" for label, v in m["masked"]]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The head in Qwen2.5-1.5B (L22H6) and Gemma-3-1B (L11H3) on 96 LCB prompts per column, 24 "
              r"in each of German, Spanish, French and Italian; the control heads are L22H8 and L11H0. Cross prompts "
              r"name the requested language; mono prompts are written in it. Top: mean attention from the last "
              r"prompt token to the language name, to as many nearby tokens, and of three other heads of the layer "
              r"to the name (L22H4, H8 and H9; L11H0, H1 and H2). Bottom: replies that pass LCB's line check without "
              r"intervention, and how many of them fail with a head zeroed on the prompt, during generation after "
              r"the first token, or both, or with its attention to the name or to the nearby tokens masked during "
              r"generation. Qwen2.5-1.5B's cross prompts were drawn among replies that passed in our LCB run, the "
              r"Italian ones among those that had switched with the head removed; the other two columns were drawn "
              r"without regard to earlier replies. One Gemma-3-1B prompt is unscorable.}",
              r"\label{tab:mechanism}", r"\end{table}"]
    return "\n".join(lines)


def mech_preview(m):
    cell = lambda v: "not run" if v is None else str(v)
    lines = ["| | Qwen2.5-1.5B cross | Qwen2.5-1.5B mono | Gemma-3-1B cross |", "|---|---|---|---|"]
    lines += [f"| attention: {label} | {qv:.3f} | not run | {gv:.3f} |" for label, qv, gv in m["attention"]]
    lines.append("| pass without intervention | " + " | ".join(f"{n} of {s}" for n, s in zip(m["passing"],
                                                                                         m["scorable"])) + " |")
    lines += [f"| fail, zeroed: {label} | " + " | ".join(map(cell, v)) + " |" for label, v in m["zeroed"]]
    lines += [f"| fail, masked: {label} | " + " | ".join(map(cell, v)) + " |" for label, v in m["masked"]]
    lines += ["", "Replies that fail without intervention and pass with it, which the table leaves out: "
              + "; ".join(f"{what}, {n}" for what, n in m["gains"]) + "."]
    return "\n".join(lines)


def main():
    rs, m = rows(), mechanism()
    text = ["# Main-text tables", "", "Generated by main_tables.py from results/.", "",
            "## The head in each instruct model", "", preview(rs), "", "```latex", tex(rs), "```", "",
            "## Mechanism in Qwen2.5-1.5B and Gemma-3-1B", "", mech_preview(m), "", "```latex", mech_tex(m), "```", ""]
    Path("tables/main_tables.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
