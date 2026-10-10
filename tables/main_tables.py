"""Main-text table of the head in each instruct model, built from results/ (EXPERIMENTS.md sections 5, 7 and 8).
Run from the repository root: python tables/main_tables.py. Writes tables/main_tables.md with a preview and the
LaTeX to paste."""
import json
from pathlib import Path

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


def table(run):
    return json.load(open(R / run / "summary.json"))["modes"]["head"]["table"]


def layer_of(h):
    return h.split("H")[0]


def flores(run, head):
    t = table(run)
    rest = [v["full"]["c2w"] for h, v in t.items() if layer_of(h) == layer_of(head) and h != head]
    return t[head]["full"]["c2w"], max(rest), t[head]["dnll"]


def lcb(run, head):
    row = json.load(open(R / run / "summary.json"))["rows"][f"{head} zero"]
    return row["monolingual_delta"], row["crosslingual_delta"]


def rows():
    out = []
    for name, run, base, head, lcb_run in MODELS:
        screen = run.endswith("-screen")
        if head is None:
            out.append({"model": name, "note": NOTES[name]})
            continue
        c2w, rest, dnll = flores(run, head)
        r = {"model": name, "head": head, "c2w": c2w, "rest": rest, "dnll": dnll, "screen": screen}
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
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Model & Head & $C\to W$ & Rest of layer & $\Delta$NLL & Base & LCB mono & LCB cross \\", r"\midrule"]
    for r in rs:
        if "note" in r:
            lines.append(rf"{r['model']} & \multicolumn{{7}}{{l}}{{{r['note']}}} \\")
            continue
        s = r"$^{s}$" if r["screen"] else ""
        base = num(r["base"]) + (r"$^{s}$" if r.get("base_screen") else "") if "base" in r else "--"
        lines.append(f"{r['model']} & {r['head']} & {num(r['c2w'])}{s} & {num(r['rest'])}{s} & "
                     f"{num(r['dnll'], signed=True)} & {base} & {delta_tex(r['mono'])} & "
                     f"{delta_tex(r['cross'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The head in each instruct model. $C\to W$: share of the 2,500 FLORES prompts whose "
              r"continuation is in the prompt language at baseline and in another language with the head zeroed; "
              r"rest of layer: the largest value among the other heads of its layer; base: the same head in the base "
              r"model. LCB: paired change in line-level pass rate with the head zeroed, five languages; gray where the "
              r"95\% CI includes zero. $^{s}$: 125-prompt screen. The Llama heads come from a screen of every head on "
              r"crosslingual LCB prompts.}",
              r"\label{tab:models}", r"\end{table*}"]
    return "\n".join(lines)


def preview(rs):
    lines = ["| model | head | c→w | rest of layer | ΔNLL | base | LCB mono | LCB cross |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if "note" in r:
            lines.append(f"| {r['model']} | {r['note']} | | | | | | |")
            continue
        s = " (screen)" if r["screen"] else ""
        base = f"{r['base']:.3f}" + (" (screen)" if r.get("base_screen") else "") if "base" in r else "not run"
        d = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
        lines.append(f"| {r['model']} | {r['head']} | {r['c2w']:.3f}{s} | {r['rest']:.3f} | {r['dnll']:+.3f} | "
                     f"{base} | {d(r['mono'])} | {d(r['cross'])} |")
    return "\n".join(lines)


def main():
    rs = rows()
    text = ["# Main-text tables", "", "Generated by main_tables.py from results/.", "",
            "## The head in each instruct model", "", preview(rs), "", "```latex", tex(rs), "```", ""]
    Path("tables/main_tables.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
