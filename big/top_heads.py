"""Print the top correct->wrong heads from out/<key>/summary.json (same rule as followup.py: dNLL <= max).

Usage: python big/top_heads.py <key> [top=3] [max_dnll=0.1]
Prints two lines: the heads (L22H6,L17H7,...) and their layers (22,17,...).
"""
import json
import sys

key = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_dnll = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
t = json.load(open(f"out/{key}/summary.json"))["modes"]["head"]["table"]
ok = [h for h in t if t[h]["dnll"] <= max_dnll]
heads = sorted(ok, key=lambda h: -t[h]["full"]["c2w"])[:top]
print(",".join(heads))
print(",".join(sorted({h[1:].split("H")[0] for h in heads}, key=int)))
