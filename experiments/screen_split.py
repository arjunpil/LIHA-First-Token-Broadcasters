# Effects of the reported heads on the prompts their screen used and on the rest (section 9g).
# FLORES: the seven heads' c->w on the 125 screen prompts (first 25 per language) and on the other 2,375, from the
# 2,500-prompt runs. LCB: the screen-picked heads' crosslingual change on the 100 screen prompts and on the rest,
# from results/lcb-screen-heldout (the screen-only value follows from the all and held-out pair means).
# usage, from the repo root: python experiments/screen_split.py
import csv
import json
from pathlib import Path

FLORES = [("Qwen2.5-1.5B", "qwen-instruct-full", "L22H6"), ("Qwen2.5-3B", "qwen2.5-3b-instruct", "L27H13"),
          ("Qwen2.5-7B", "qwen2.5-7b-instruct", "L19H1"), ("Qwen3-1.7B", "qwen3-1.7b-instruct", "L18H12"),
          ("Gemma-3-1B", "gemma3-1b-instruct", "L11H3"), ("Gemma-3-4B", "gemma3-4b-instruct", "L24H0"),
          ("OLMo-2-1B", "olmo2-1b-instruct", "L12H8")]


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


expected = [r["language"] for r in csv.DictReader(open("prompts/prompts_european.csv", encoding="utf-8"))]
seen, screen = {}, set()
for i, l in enumerate(expected):
    seen[l] = seen.get(l, 0) + 1
    if seen[l] <= 25:
        screen.add(i)
rest = [i for i in range(len(expected)) if i not in screen]

print("FLORES c->w | screen (125) | rest (2,375) | all (2,500)")
for name, run, head in FLORES:
    labels = json.load(open(f"results/{run}/labels.json"))
    base, abl = labels["base"]["labels"], labels[f"head:{head}"]["labels"]

    def c2w(idx):
        return sum(same(base[i], expected[i]) and not same(abl[i], expected[i]) for i in idx) / len(idx)

    print(f"{name} {head} | {c2w(sorted(screen)):.3f} | {c2w(rest):.3f} | {c2w(range(len(expected))):.3f}")

print("\nLCB crosslingual change | screen | rest | all")
for f in sorted(Path("results/lcb-screen-heldout").glob("*.json")):
    if f.name.endswith("screen-prompts.json"):
        continue
    s = json.load(open(f))
    for cond, row in s["rows"].items():
        if cond == "base" or "control" in cond or not cond.endswith("zero"):
            continue
        a, h = row["crosslingual/all"], row["crosslingual/held_out"]
        n_s = a["pairs"] - h["pairs"]
        d_s = (a["pairs"] * a["delta"][0] - h["pairs"] * h["delta"][0]) / n_s
        print(f"{f.stem} {cond} | {d_s:+.3f} ({n_s} pairs) | {h['delta'][0]:+.3f} | {a['delta'][0]:+.3f}")
