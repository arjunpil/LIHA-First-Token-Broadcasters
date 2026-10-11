# reading of an LCB head screen (experiments/lcb_screen_flores.md). Heads that fail the head rule on the 125-prompt
# FLORES screen (dNLL above 1, or under 0.9 of English continuations kept in English) are left out. Prints the head
# that gets a full LCB run: the strongest one, if its drop beats the FLORES head's by more than 0.025 and is at least
# 0.1. Prints nothing otherwise. Writes the reading to <screen>/reading.md.
# usage: python pick_lcbscreen.py out/<model>-lcbscreen out/<flores screen> <flores head>
import csv
import json
import math
import sys
from pathlib import Path

TIE, MIN_DROP = 0.025, 0.1


def same(label, expected):
    return label == expected or label.startswith(expected + "-")


screen, flores, fh = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
s = json.load(open(screen / "screen.json"))
table = json.load(open(flores / "summary.json"))["modes"]["head"]["table"]
labels = json.load(open(flores / "labels.json"))
expected = [r["language"] for r in csv.DictReader(open(flores / "prompts.csv", encoding="utf-8"))]
en = [i for i, e in enumerate(expected) if e == "en"]
assert len(labels["base"]["labels"]) == len(expected) == 125 and len(en) == 25

bad = {}
for h, row in table.items():
    kept = sum(same(labels[f"head:{h}"]["labels"][i], "en") for i in en) / len(en)
    if row["dnll"] > 1 or kept < 0.9:
        bad[h] = (row["dnll"], kept)
assert set(s["heads"]) == set(table), "screen and FLORES screen cover different heads"
drop = {h: s["base"] - r for h, r in s["heads"].items() if h not in bad and not math.isnan(r)}
order = sorted(drop, key=lambda h: -drop[h])
top = order[0]
rank = 1 + sum(drop[h] > drop[fh] + 1e-12 for h in drop)
tied = [h for h in order if h != fh and abs(drop[h] - drop[fh]) <= TIE and drop[h] >= drop[fh]]
pick = top if drop[top] - drop[fh] > TIE and drop[top] >= MIN_DROP else ""

lines = [f"# Reading of {screen.name}", "",
         f"Baseline pass rate {s['base']:.3f}. Left out by the head rule: "
         + (", ".join(f"{h} (dNLL {d:+.2f}, English kept {k:.2f})" for h, (d, k) in sorted(bad.items()))
            or "none") + ".",
         "", f"FLORES head {fh}: drop {drop[fh]:+.3f}, rank {rank} of {len(drop)}"
         + (f"; within {TIE} above it: {', '.join(tied)}" if tied else "") + ".",
         f"Strongest head {top}: drop {drop[top]:+.3f}.",
         f"Full LCB run: {pick or 'none'}.", "", "| head | drop |", "|---|---|"]
lines += [f"| {h} | {drop[h]:+.3f} |" for h in order[:10]]
(screen / "reading.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:6]), file=sys.stderr)
print(pick)
