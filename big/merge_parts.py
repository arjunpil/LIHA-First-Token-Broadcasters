"""Merge per-worker sweep outputs (out/<key>-part*/) into out/<key>/ so detect.py and analyze.py can run on it.

Usage: python big/merge_parts.py <key>
"""
import json
import shutil
import sys
from pathlib import Path

key = sys.argv[1]
parts = sorted(Path("out").glob(f"{key}-part*"))
if not parts:
    sys.exit(f"no out/{key}-part* folders")
dest = Path("out") / key
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(parts[0] / "prompts.csv", dest / "prompts.csv")

base, rows, seen = None, [], set()
for p in parts:
    if (p / "prompts.csv").read_bytes() != (dest / "prompts.csv").read_bytes():
        sys.exit(f"{p} used a different prompt set")
    for line in open(p / "gens.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["cond"] == "base":
            if base is None:
                base = r
            elif r["texts"] != base["texts"]:
                print(f"warning: base generations differ in {p} (batching/precision); keeping the first")
            continue
        if r["cond"] not in seen:
            seen.add(r["cond"])
            rows.append(r)

with open(dest / "gens.jsonl", "w", encoding="utf-8") as f:
    for r in [base] + rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"merged {len(parts)} parts -> {dest}/gens.jsonl ({len(rows)} head conditions)")
