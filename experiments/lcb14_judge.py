"""Per-language reading of the 14-language LCB runs, as fixed in lcb14_plan.md: the head's paired LPR change under
zero ablation with a bootstrap 95% CI, against the three controls' changes. A language is affected if the head's CI
lies below zero and its change is below every control's change. The Qwen2.5-1.5B pattern is shared if, on
monolingual prompts, Hindi is affected and Chinese, Japanese and Russian are not. With --skipped-fail, a check outside
the plan: a reply the line check cannot score under an intervention counts as a failure instead of being left out."""
import argparse
import gzip
import json
from pathlib import Path

from lcb import paired_ci

TASKS = ("monolingual", "crosslingual")


def load(d):
    rows, p = {}, Path(d) / "samples.jsonl"
    with open(p, encoding="utf-8") if p.exists() else gzip.open(f"{p}.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.setdefault(r["cond"], []).append(r)
    return rows


def change(base, cond, idx, skipped_fail=False):
    # prompts both runs score; with skipped_fail, every prompt base scores, an unscorable reply under cond failing
    keep = [i for i in idx if not base[i]["skipped"] and (skipped_fail or not cond[i]["skipped"])]
    if not keep:  # nothing to compare: no change to read, never counted as affected
        return (float("nan"),) * 3, 0
    after = [0 if cond[i]["skipped"] else cond[i]["pass"] for i in keep]
    return paired_ci([base[i]["pass"] for i in keep], after), len(keep)


def judge(d, skipped_fail=False):
    rows = load(d)
    base = rows["base"]
    key = [(r["task"], r["language"], r["prompt"]) for r in base]
    assert all([(r["task"], r["language"], r["prompt"]) for r in v] == key for v in rows.values())
    head = next(c for c in rows if c.endswith(" zero") and "(control)" not in c)
    controls = [c for c in rows if c.endswith("zero (control)")]
    langs = sorted({r["language"] for r in base} - {"en"})
    out = {"dir": str(d), "head": head.split()[0], "controls": [c.split()[0] for c in controls],
           "skipped_fail": skipped_fail, "tasks": {}}
    for t in TASKS:
        res = {}
        for l in langs:
            idx = [i for i, r in enumerate(base) if r["task"] == t and r["language"] == l]
            if not idx:
                continue
            (m, lo, hi), n = change(base, rows[head], idx, skipped_fail)
            ctrl = [change(base, rows[c], idx, skipped_fail)[0][0] for c in controls]
            res[l] = {"n": n, "delta": m, "ci": [lo, hi], "controls": ctrl,
                      "affected": hi < 0 and all(m < c for c in ctrl)}
        out["tasks"][t] = res
    mono = out["tasks"]["monolingual"]
    out["pattern_shared"] = mono["hi"]["affected"] and not any(mono[l]["affected"] for l in ("zh", "ja", "ru"))
    return out


def report(res):
    scored = ("on prompts base scores, with a reply the intervention leaves unscorable counted as a failure (outside "
              "the plan)" if res["skipped_fail"] else "on prompts both runs score")
    lines = [f"# {Path(res['dir']).name}: per-language reading (lcb14_plan.md)", "",
             f"Head {res['head']}, zero ablation; controls {', '.join(res['controls'])}. Δ = paired LPR change "
             f"against base {scored}, bootstrap 95% CI. Affected = CI below zero and Δ below every control's Δ.", ""]
    for t, langs in res["tasks"].items():
        lines += [f"## {t}", "", "| language | n | Δ head | 95% CI | Δ controls | affected |",
                  "|---|---|---|---|---|---|"]
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
    p.add_argument("--skipped-fail", action="store_true", help="writes judge_skipped_fail.json and .md")
    a = p.parse_args()
    name = "judge_skipped_fail" if a.skipped_fail else "judge"
    for d in a.dirs:
        res = judge(d, a.skipped_fail)
        json.dump(res, open(Path(d) / f"{name}.json", "w"), indent=1)
        (Path(d) / f"{name}.md").write_text(report(res), encoding="utf-8")
        print(report(res))


if __name__ == "__main__":
    main()
