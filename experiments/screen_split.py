# Checks on how the reported heads were picked (section 9g).
# 1. FLORES: the seven heads' c->w on the 125 screen prompts (first 25 per language) and on the other 2,375, from the
#    2,500-prompt runs.
# 2. FLORES: the strongest head outside the layers rerun on 2,500 prompts that meets both conditions of the head rule
#    on the screen (dNLL <= 1, at least 0.9 of the 25 English continuations kept in English).
# 3. LCB: the screen-picked heads' crosslingual change on the 100 screen prompts and on the rest, from
#    results/lcb-screen-heldout (the screen-only value follows from the all and held-out pair means).
# 4. LCB: on the screens of the three models without a FLORES head, the share of replies that pass at baseline and
#    fail with the head removed, for the picked head and the other heads of its layer (results/lcb-screen-directions).
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

print("\nFLORES, outside the rerun layers: strongest head meeting both conditions on the screen")
for name, run, head in FLORES[1:] + [("OLMo-3-7B", "olmo3-7b-instruct", "L14H25")]:
    table = json.load(open(f"results/{run}-screen/summary.json"))["modes"]["head"]["table"]
    labels = json.load(open(f"results/{run}-screen/labels.json"))
    rerun = {k.split(":")[1].split("H")[0] for k in json.load(open(f"results/{run}/labels.json")) if k.startswith("head:")}
    en = [j for j, i in enumerate(sorted(screen)) if expected[i] == "en"]

    def eligible(h):
        kept = sum(same(labels[f"head:{h}"]["labels"][j], "en") for j in en) / len(en)
        return table[h]["dnll"] <= 1 and kept >= 0.9

    out = [h for h in table if h.split("H")[0] not in rerun and eligible(h)]
    best = max(out, key=lambda h: table[h]["full"]["c2w"])
    print(f"{name}: rerun layers {', '.join(sorted(rerun, key=lambda x: int(x[1:])))}; outside them {best} "
          f"{table[best]['full']['c2w']:.3f}, against {head} {table[head]['full']['c2w']:.3f} on the screen")

print("\nLCB screen, pass at baseline and fail with the head removed | picked head | other heads of its layer, max")
for model, head in (("llama3.2-1b-instruct", "L8H25"), ("llama3.2-3b-instruct", "L13H19"), ("qwen3-4b-instruct", "L24H27")):
    heads = json.load(open(f"results/lcb-screen-directions/{model}.json"))["heads"]
    layer = [h for h in heads if h.split("H")[0] == head.split("H")[0] and h != head]
    other = max(layer, key=lambda h: heads[h]["pass_to_fail"])
    print(f"{model} | {head} {heads[head]['pass_to_fail']:.3f} (reverse {heads[head]['fail_to_pass']:.3f}) | "
          f"{other} {heads[other]['pass_to_fail']:.3f} of {len(layer)}")
