# Reading of experiments/mechanism_value_patch.md from the saved outputs.
# usage: python experiments/mechanism_value_patch_reading.py [results root] [out dir]
# The root holds valuepatch-<model>-s1 and -s2 (gemma_l11h3_attention.py outputs) and the earlier runs on the 96
# prompts (gemma-l11h3-mechanism-ctrl1, qwen-l22h6-mechanism-unselected), whose clean replies sample 1 reproduces.
import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean

SWAP = {"de": "fr", "es": "de", "fr": "de", "it": "de"}
EARLIER = {"gemma": "gemma-l11h3-mechanism-ctrl1", "qwen": "qwen-l22h6-mechanism-unselected"}
NAMES = {"gemma": "Gemma-3-1B L11H3", "qwen": "Qwen2.5-1.5B L22H6"}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d):
    by = defaultdict(dict)
    d = Path(d) / "full" if (Path(d) / "full").is_dir() else Path(d)
    for r in csv.DictReader(open(d / "condition_outputs.csv", encoding="utf-8")):
        by[r["prompt_id"]][r["condition"]] = r
    return by


def flag(r, k):
    return str(r[k]).strip().lower() in ("true", "1")


def in_donor(r, lang):
    labels = [x for x in r["labels"].split(",") if x]
    return not flag(r, "skipped") and bool(labels) and all(x == SWAP[lang] for x in labels)


def counts(rows, cond):
    """pass-to-fail and fail-to-pass among prompts scorable in both, newly unscorable among baseline-scorable."""
    pf = fp = new_skip = 0
    for v in rows:
        b, x = v["clean"], v[cond]
        if flag(b, "skipped"):
            continue
        if flag(x, "skipped"):
            new_skip += 1
            continue
        pf += flag(b, "passed") and not flag(x, "passed")
        fp += (not flag(b, "passed")) and flag(x, "passed")
    return pf, fp, new_skip


def fails(rows, cond):
    return {i for i, v in enumerate(rows) if not flag(v["clean"], "skipped") and flag(v["clean"], "passed")
            and not flag(v[cond], "skipped") and not flag(v[cond], "passed")}


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "mechanism-value-patch"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Value patching and the larger mechanism sample", "",
             "Reading fixed in experiments/mechanism_value_patch.md. Counts over baseline-passing replies unless said "
             "otherwise; exact two-sided McNemar tests, uncorrected. Sample 1 = the 96 earlier prompts (its nearby "
             "and control-head masks come from the earlier run, whose clean replies sample 1 reproduces exactly); "
             "sample 2 = 384 new prompts.", ""]
    for m in ("qwen", "gemma"):
        s1, s2 = load(root / f"valuepatch-{m}-s1"), load(root / f"valuepatch-{m}-s2")
        early = load(root / EARLIER[m])
        for pid, v in s1.items():  # bring the earlier masks onto sample 1
            assert v["clean"]["reply"] == early[pid]["clean"]["reply"], (m, pid)
            for c in ("target_word_mask", "nearby_word_mask", "control_head_word_mask", "head_zero_decode"):
                v[c] = early[pid][c]
        rows1, rows2 = list(s1.values()), list(s2.values())
        pooled = rows1 + rows2
        lines += [f"## {NAMES[m]}", ""]
        for label, rows in (("sample 1", rows1), ("sample 2", rows2), ("pooled", pooled)):
            sc = [v for v in rows if not flag(v["clean"], "skipped")]
            ps = [v for v in sc if flag(v["clean"], "passed")]
            lines.append(f"{label}: {len(ps)} of {len(sc)} scorable replies pass at baseline ({len(rows)} prompts)")
        lines += ["", "| condition | sample | pass to fail | fail to pass | newly unscorable | all lines in donor language "
                  "(of scorable) | mean output change |", "|---|---|---|---|---|---|---|"]
        conds = ("target_word_mask", "nearby_word_mask", "control_head_word_mask", "head_zero_decode",
                 "value_patch_same", "value_patch_other")
        for cond in conds:
            for label, rows in (("1", rows1), ("2", rows2), ("pooled", pooled)):
                pf, fp, ns = counts(rows, cond)
                sc = [v for v in rows if not flag(v[cond], "skipped")]
                donor = sum(in_donor(v[cond], v["clean"]["language"]) for v in sc)
                norms = [float(v[cond]["delta_norm_mean"]) for v in rows if v[cond].get("delta_norm_mean")]
                lines.append(f"| {cond} | {label} | {pf} | {fp} | {ns} | {donor} / {len(sc)} | "
                             f"{mean(norms):.3f} |" if norms else
                             f"| {cond} | {label} | {pf} | {fp} | {ns} | {donor} / {len(sc)} | |")
        base_donor = sum(in_donor(v["clean"], v["clean"]["language"]) for v in pooled
                         if not flag(v["clean"], "skipped"))
        lines += ["", f"Baseline replies already all in the donor language: {base_donor}", ""]

        def compare(rows, a, b):
            fa, fb = fails(rows, a), fails(rows, b)
            return len(fa - fb), len(fb - fa), mcnemar(len(fa - fb), len(fb - fa))

        def compare_donor(rows, a, b):
            both = [v for v in rows if not flag(v[a], "skipped") and not flag(v[b], "skipped")]
            da = {i for i, v in enumerate(both) if in_donor(v[a], v["clean"]["language"])}
            db = {i for i, v in enumerate(both) if in_donor(v[b], v["clean"]["language"])}
            return len(da - db), len(db - da), mcnemar(len(da - db), len(db - da)), len(both)

        g = compare(pooled, "value_patch_same", "nearby_word_mask")
        c = compare(pooled, "value_patch_other", "value_patch_same")
        d = compare_donor(pooled, "value_patch_other", "value_patch_same")
        r1 = compare(rows2, "target_word_mask", "nearby_word_mask")
        r2 = compare(rows2, "target_word_mask", "control_head_word_mask")
        general = g[0] > g[1] and g[2] < 0.05
        lines += [
            "Readings:",
            f"- Guard, pooled: value_patch_same against nearby mask, {g[0]} vs {g[1]} discordant, p = {g[2]:.4g}; "
            f"read as a general perturbation: {'yes' if general else 'no'}",
            f"- Content, pooled: value_patch_other against value_patch_same, {c[0]} vs {c[1]} discordant failures, "
            f"p = {c[2]:.4g}",
            f"- Direction, pooled: replies all in the donor language, other vs same, {d[0]} vs {d[1]} discordant of "
            f"{d[3]} scorable in both, p = {d[2]:.4g}",
            f"- Replication, sample 2: name mask against nearby mask {r1[0]} vs {r1[1]}, p = {r1[2]:.4g}; against "
            f"control head's mask {r2[0]} vs {r2[1]}, p = {r2[2]:.4g}",
            ""]
        # by language, pooled
        lines += ["Pass to fail by language, pooled (baseline-passing n in brackets):", "",
                  "| condition | de | es | fr | it |", "|---|---|---|---|---|"]
        for cond in conds:
            cells = []
            for lang in ("de", "es", "fr", "it"):
                rows = [v for v in pooled if v["clean"]["language"] == lang]
                n = sum(not flag(v["clean"], "skipped") and flag(v["clean"], "passed") for v in rows)
                cells.append(f"{counts(rows, cond)[0]} ({n})")
            lines.append(f"| {cond} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
