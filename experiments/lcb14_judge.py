"""Per-language reading of the 14-language LCB runs, as fixed in lcb14_plan.md: the head's paired LPR change under
zero ablation with a bootstrap 95% CI, against the three controls' changes. A language is affected if the head's CI
lies below zero and its change is below every control's change. The Qwen2.5-1.5B pattern is shared if, on
monolingual prompts, Hindi is affected and Chinese, Japanese and Russian are not."""
import argparse
import gzip
import json
from pathlib import Path

from lcb import paired_ci

TASKS = ("monolingual", "crosslingual")


def load(d):
    rows, p = {}, Path(d) / "samples.jsonl"
    f = open(p, encoding="utf-8") if p.exists() else gzip.open(f"{p}.gz", "rt", encoding="utf-8")
    for line in f:
        r = json.loads(line)
        rows.setdefault(r["cond"], []).append(r)
    return rows


def change(base, cond, idx):
    both = [i for i in idx if not base[i]["skipped"] and not cond[i]["skipped"]]
    if not both:
        return None, 0
    return paired_ci([base[i]["pass"] for i in both], [cond[i]["pass"] for i in both]), len(both)


def judge(d):
    rows = load(d)
    base = rows["base"]
    key = [(r["task"], r["language"], r["prompt"]) for r in base]
    assert all([(r["task"], r["language"], r["prompt"]) for r in v] == key for v in rows.values())
    head = next(c for c in rows if c.endswith(" zero") and "(control)" not in c)
    controls = [c for c in rows if c.endswith("zero (control)")]
    langs = sorted({r["language"] for r in base} - {"en"})
    out = {"dir": str(d), "head": head.split()[0], "controls": [c.split()[0] for c in controls], "tasks": {}}
    for t in TASKS:
        res = {}
        for l in langs:
            idx = [i for i, r in enumerate(base) if r["task"] == t and r["language"] == l]
            if not idx:
                continue
            (m, lo, hi), n = change(base, rows[head], idx)
            ctrl = [change(base, rows[c], idx)[0][0] for c in controls]
            res[l] = {"n": n, "delta": m, "ci": [lo, hi], "controls": ctrl,
                      "affected": hi < 0 and all(m < c for c in ctrl)}
        out["tasks"][t] = res
    mono = out["tasks"]["monolingual"]
    out["pattern_shared"] = mono["hi"]["affected"] and not any(mono[l]["affected"] for l in ("zh", "ja", "ru"))
    return out


def report(res):
    lines = [f"# {Path(res['dir']).name}: per-language reading (lcb14_plan.md)", "",
             f"Head {res['head']}, zero ablation; controls {', '.join(res['controls'])}. Δ = paired LPR change "
             "against base on prompts both runs score, bootstrap 95% CI. Affected = CI below zero and Δ below every "
             "control's Δ.", ""]
    for t, langs in res["tasks"].items():
        lines += [f"## {t}", "", "| language | n | Δ head | 95% CI | Δ controls | affected |", "|---|---|---|---|---|---|"]
        for l, r in langs.items():
            lines.append(f"| {l} | {r['n']} | {r['delta']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
                         + ", ".join(f"{c:+.3f}" for c in r["controls"]) + f" | {'yes' if r['affected'] else ''} |")
        lines.append("")
    lines.append(f"Qwen2.5-1.5B pattern (monolingual: hi affected; zh, ja, ru not): "
                 f"{'shared' if res['pattern_shared'] else 'not shared'}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    a = p.parse_args()
    for d in a.dirs:
        res = judge(d)
        json.dump(res, open(Path(d) / "judge.json", "w"), indent=1)
        (Path(d) / "judge.md").write_text(report(res), encoding="utf-8")
        print(report(res))


if __name__ == "__main__":
    main()
