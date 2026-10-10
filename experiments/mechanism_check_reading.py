"""Reading of the two checks written down in experiments/mechanism_checks.md, from the saved outputs. For each model:
the replies that pass the line check without intervention, how many of them fail under each mask and under zeroing,
exact two-sided McNemar tests of the head's mask against the control head's and the nearby mask, and the clean
attention of every head of the layer to the language name. Run from the repository root; writes
results/mechanism-checks/summary.md."""
import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean

R = Path("results")
RUNS = [("Gemma-3-1B", "gemma-l11h3-mechanism-ctrl1"), ("Qwen2.5-1.5B", "qwen-l22h6-mechanism-unselected")]
CONDITIONS = [("head's attention to the name masked", "target_word_mask"),
              ("head's attention to nearby tokens masked", "nearby_word_mask"),
              ("control head's attention to the name masked", "control_head_word_mask"),
              ("head zeroed during generation", "head_zero_decode")]


def mcnemar(b, c):
    # exact two-sided test on the discordant pairs
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def prompts(run):
    by = defaultdict(dict)
    for r in csv.DictReader(open(R / run / "condition_outputs.csv", encoding="utf-8")):
        by[r["prompt_id"]][r["condition"]] = r
    return list(by.values())


def fails(p, cond):
    return p[cond]["skipped"] == "False" and p[cond]["passed"] == "False"


def attention(run):
    rows = list(csv.DictReader(open(R / run / "attention_by_prompt.csv", encoding="utf-8")))
    heads = sorted(int(k[len("last_prompt_h"):-len("_target_mass")]) for k in rows[0]
                   if k.startswith("last_prompt_h") and k.endswith("_target_mass"))
    return {h: mean(float(r[f"last_prompt_h{h}_target_mass"]) for r in rows) for h in heads}


def main():
    lines = ["# Mechanism checks", "", "The reading fixed in experiments/mechanism_checks.md, from the saved outputs "
             "(python experiments/mechanism_check_reading.py). Failures count replies that pass the line check without "
             "intervention and fail with it.", ""]
    for name, run in RUNS:
        meta = json.load(open(R / run / "run_metadata.json"))
        layer, head, ctrl = meta["layer"], meta["head"], meta["control_head"]
        att = attention(run)
        assert ctrl == max((h for h in att if h != head), key=att.get), "control head is not the rule's pick"
        ps = prompts(run)
        ok = [p for p in ps if p["clean"]["skipped"] == "False" and p["clean"]["passed"] == "True"]
        scorable = sum(p["clean"]["skipped"] == "False" for p in ps)
        lines += [f"## {name}, L{layer}H{head} (control head L{layer}H{ctrl})", "",
                  f"{len(ok)} of {scorable} scorable replies pass without intervention ("
                  + ", ".join(f"{l} {n}" for l, n in sorted(Counter(p['clean']['language'] for p in ok).items()))
                  + ").", "", "| condition | fail | by language | fail without intervention, pass with it |",
                  "|---|---|---|---|"]
        for label, cond in CONDITIONS:
            lost = [p for p in ok if fails(p, cond)]
            gained = sum(p["clean"]["skipped"] == "False" and p["clean"]["passed"] == "False"
                         and p[cond]["passed"] == "True" for p in ps)
            by_lang = ", ".join(f"{l} {n}" for l, n in sorted(Counter(p["clean"]["language"] for p in lost).items()))
            lines.append(f"| {label} | {len(lost)} of {len(ok)} | {by_lang} | {gained} |")
        both = sum(fails(p, "target_word_mask") and fails(p, "head_zero_decode") for p in ok)
        lines += ["", f"Failing under both the head's mask and zeroing: {both}.", ""]
        tests = {}
        for other, label in (("control_head_word_mask", "control head's mask"), ("nearby_word_mask", "nearby mask")):
            b = sum(fails(p, "target_word_mask") and not fails(p, other) for p in ok)
            c = sum(fails(p, other) and not fails(p, "target_word_mask") for p in ok)
            tests[other] = (b, c, mcnemar(b, c))
            lines.append(f"- Head's mask against the {label}: {b} vs {c} discordant, "
                         f"exact McNemar p = {tests[other][2]:.4f}")
        specific = all(t[0] > t[1] and t[2] < 0.05 for t in tests.values())
        lines += ["", "Last-prompt-token attention to the language name, mean over the prompts: "
                  + ", ".join(f"L{layer}H{h} {v:.3f}" for h, v in sorted(att.items(), key=lambda x: -x[1])) + ".", "",
                  "Reading: the head's mask fails more replies than the control head's mask and than the nearby mask, "
                  "each with p < 0.05: " + ("yes" if specific else "no") + ".", ""]
    out = R / "mechanism-checks"
    out.mkdir(exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
