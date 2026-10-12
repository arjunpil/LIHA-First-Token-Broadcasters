# Directions of change on an LCB head screen (section 9g): for each head, the share of prompts scored with and
# without the head that pass at baseline and fail with it removed, and the reverse. The screen's replies are in
# out/<model>-lcbscreen/samples.jsonl (not committed); the pooled pass rates are checked against its screen.json.
# usage: python experiments/lcb_screen_directions.py <screen dir with samples.jsonl> <out json>
import json
import sys
from pathlib import Path


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    conds = {}
    for line in open(src / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        conds.setdefault(r["cond"], []).append(r)
    base = conds.pop("base")
    screen = json.load(open(src / "screen.json"))
    heads = {}
    for head, rows in conds.items():
        assert [r["prompt"] for r in rows] == [r["prompt"] for r in base]
        scored = [r["pass"] for r in rows if not r["skipped"]]
        assert abs(sum(scored) / len(scored) - screen["heads"][head]) < 1e-9, head
        pairs = [(b["pass"], r["pass"]) for b, r in zip(base, rows) if not b["skipped"] and not r["skipped"]]
        heads[head] = {"pairs": len(pairs),
                       "pass_to_fail": sum(b and not a for b, a in pairs) / len(pairs),
                       "fail_to_pass": sum(a and not b for b, a in pairs) / len(pairs)}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"screen": src.name, "base": screen["base"], "heads": heads}, open(out, "w"), indent=0)


if __name__ == "__main__":
    main()
