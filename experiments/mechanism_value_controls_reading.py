# Reading of experiments/mechanism_value_controls.md.
# usage: python experiments/mechanism_value_controls_reading.py [new runs root] [results root] [out dir]
# The new runs root holds valuecontrols-<model>-s1 and -s2; the results root holds the earlier runs read by
# mechanism_value_patch_reading.py, whose clean replies the new runs must reproduce.
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))
from mechanism_value_patch_reading import EARLIER, NAMES, SWAP, counts, fails, flag, load, mcnemar  # noqa: E402

SWAP2 = {"de": "es", "es": "it", "fr": "it", "it": "fr"}
RANDOM = ("value_patch_random1", "value_patch_random2", "value_patch_random3")
OLD = ("value_patch_same", "value_patch_other", "target_word_mask")


def in_lang(r, target):
    labels = [x for x in r["labels"].split(",") if x]
    return not flag(r, "skipped") and bool(labels) and all(x == target for x in labels)


def moved(rows, cond, swap):
    return {i for i, v in enumerate(rows) if in_lang(v[cond], swap[v["clean"]["language"]])}


def merged(new_root, old_root, m):
    rows = []
    for s in ("s1", "s2"):
        new = load(Path(new_root) / f"valuecontrols-{m}-{s}")
        old = load(Path(old_root) / f"valuepatch-{m}-{s}")
        masks = load(Path(old_root) / EARLIER[m]) if s == "s1" else old
        assert set(new) == set(old), (m, s)
        for pid, v in new.items():
            assert v["clean"]["reply"] == old[pid]["clean"]["reply"], (m, s, pid)
            v["value_patch_same"], v["value_patch_other"] = old[pid]["value_patch_same"], old[pid]["value_patch_other"]
            v["target_word_mask"] = masks[pid]["target_word_mask"]
            if "control_head_word_mask" in v:
                v["l11h2_word_mask"] = v.pop("control_head_word_mask")
            rows.append(v)
    return rows


def main():
    new_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    old_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else old_root / "mechanism-value-controls"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Controls for the value patch", "",
             "Reading fixed in experiments/mechanism_value_controls.md (79dc4e6). Samples 1 and 2 pooled; counts over "
             "baseline-passing replies unless said otherwise; exact two-sided McNemar tests, uncorrected. The clean "
             "replies of the new runs reproduce the earlier ones exactly (checked).", ""]
    for m in ("qwen", "gemma"):
        rows = merged(new_root, old_root, m)
        conds = [*OLD, "value_patch_other2", *RANDOM] + (["l11h2_word_mask"] if "l11h2_word_mask" in rows[0] else [])
        sc = [v for v in rows if not flag(v["clean"], "skipped")]
        lines += [f"## {NAMES[m]}", "",
                  f"{sum(flag(v['clean'], 'passed') for v in sc)} of {len(sc)} scorable replies pass at baseline "
                  f"({len(rows)} prompts)", "",
                  "| condition | pass to fail | fail to pass | newly unscorable | all lines in first donor language "
                  "| all lines in second donor language | mean output change |", "|---|---|---|---|---|---|---|"]
        for c in conds:
            pf, fp, ns = counts(rows, c)
            n = sum(not flag(v[c], "skipped") for v in rows)
            norms = [float(v[c]["delta_norm_mean"]) for v in rows if v[c].get("delta_norm_mean")]
            lines.append(f"| {c} | {pf} | {fp} | {ns} | {len(moved(rows, c, SWAP))} / {n} | "
                         f"{len(moved(rows, c, SWAP2))} / {n} | {f'{mean(norms):.3f}' if norms else ''} |")

        def cmp(a, b, sets):
            sa, sb = sets(a), sets(b)
            return len(sa - sb), len(sb - sa), mcnemar(len(sa - sb), len(sb - sa))

        def both(rows_, a, b, swap):
            keep = [v for v in rows_ if not flag(v[a], "skipped") and not flag(v[b], "skipped")]
            return lambda c: moved(keep, c, swap)

        lines += ["", "Readings:"]
        size = [cmp("value_patch_other", r, lambda c: fails(rows, c)) for r in RANDOM]
        lines.append("- Size, other-language against random-direction failures, per seed: " + "; ".join(
            f"{a} vs {b}, p = {p:.3g}" for a, b, p in size)
            + f". Read as depending on direction: {'yes' if all(a > b and p < 0.05 for a, b, p in size) else 'no'}")
        dirn = [cmp("value_patch_other", r, both(rows, "value_patch_other", r, SWAP)) for r in RANDOM]
        lines.append("- Direction, replies all in the first donor language, other-language against random, per "
                     "seed: " + "; ".join(f"{a} vs {b}, p = {p:.3g}" for a, b, p in dirn)
                     + f". Read as specific to its values: {'yes' if all(a > b and p < 0.05 for a, b, p in dirn) else 'no'}")
        a, b, p = cmp("value_patch_other2", "value_patch_same", both(rows, "value_patch_other2", "value_patch_same", SWAP2))
        lines.append(f"- Second language, replies all in the second donor language, other2 against same: {a} vs {b}, "
                     f"p = {p:.3g}. Shown: {'yes' if a > b and p < 0.05 else 'no'}")
        if "l11h2_word_mask" in conds:
            a, b, p = cmp("target_word_mask", "l11h2_word_mask", lambda c: fails(rows, c))
            lines.append(f"- L11H2, L11H3 name mask against L11H2 name mask failures: {a} vs {b}, p = {p:.3g}. "
                         f"Shown: {'yes' if a > b and p < 0.05 else 'no'}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
