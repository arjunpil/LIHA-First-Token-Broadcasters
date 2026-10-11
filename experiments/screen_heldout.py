# LCB changes of a run whose head was picked by an LCB screen (sections 8 and 9e), without the screen's prompts.
# The screen's prompts come from its saved replies (out/<model>-lcbscreen/samples.jsonl, not committed) or from the
# list this script saves. The full-run numbers are checked against the run's summary.json first.
# usage: python screen_heldout.py <screen dir or prompts.json> <full run dir> <head> <out prefix>
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from lcb import TASKS, lpr, paired_ci


def key(r):
    return r["task"], r["source"], r["language"], r["prompt"]


def read(path):
    gz = path.with_suffix(".jsonl.gz")
    f = open(path, encoding="utf-8") if path.exists() else gzip.open(gz, "rt", encoding="utf-8")
    with f:
        yield from (json.loads(line) for line in f)


def screen_prompts(src):
    if src.suffix == ".json":
        return [tuple(k) for k in json.load(open(src, encoding="utf-8"))]
    keys = []
    for r in read(src / "samples.jsonl"):
        if r["cond"] != "base":
            break
        keys.append(key(r))
    return keys


def main():
    src, run, head, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    screen = screen_prompts(src)
    in_screen = set(screen)
    conds = {}
    for r in read(run / "samples.jsonl"):
        conds.setdefault(r["cond"], []).append(r)
    base = conds["base"]
    summary = json.load(open(run / "summary.json"))
    langs = summary["args"]["langs"].split(",")
    # grouped by language in the run's order, as lcb.py does, so the bootstrap draws match
    non_en = {t: [i for l in langs if l != "en" for i, r in enumerate(base) if r["task"] == t and r["language"] == l]
              for t in TASKS}
    keep = {(t, part): [i for i in non_en[t] if part == "all" or key(base[i]) not in in_screen]
            for t in TASKS for part in ("all", "held_out")}

    out = {"head": head, "run": run.name, "screen_prompts": len(screen),
           "found_in_run": sum(key(r) in in_screen for r in base), "rows": {}}
    for cond, rows in conds.items():
        assert list(map(key, rows)) == list(map(key, base))
        row = {}
        for (t, part), idx in keep.items():
            both = [i for i in idx if not rows[i]["skipped"] and not base[i]["skipped"]]
            d = None if cond == "base" else paired_ci([base[i]["pass"] for i in both], [rows[i]["pass"] for i in both])
            row[f"{t}/{part}"] = {"prompts": len(idx), "pairs": len(both), "lpr": lpr(rows, rows, idx), "delta": d}
        s = summary["rows"][cond]
        for t in TASKS:
            assert abs(s[t] - row[f"{t}/all"]["lpr"]) < 1e-9, (cond, t)
            assert cond == "base" or np.allclose(s[f"{t}_delta"], row[f"{t}/all"]["delta"]), (cond, t)
        out["rows"][cond] = row
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{prefix}.json", "w"), indent=1)
    if src.suffix != ".json":
        json.dump(screen, open(f"{prefix}-screen-prompts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    for cond, row in out["rows"].items():
        if cond != "base":
            a, h = row["crosslingual/all"]["delta"], row["crosslingual/held_out"]["delta"]
            print(f"{cond}: cross {a[0]:+.3f} -> {h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}]")


if __name__ == "__main__":
    main()
