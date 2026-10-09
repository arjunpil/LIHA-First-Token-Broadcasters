"""Word-level pass rate (WPR) of LCB for saved replies, as in the benchmark's compute_metrics.py: among the replies
whose lines are all in the expected language, the share with no English dictionary word (lowercase, longer than
three letters). The benchmark reports it for ar, hi, ja, ko, ru and zh only."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lcb import PUNCT, words

WPR_LANGS = ("ar", "hi", "ja", "ko", "ru", "zh")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out/qwen-instruct-lcb-all")
    p.add_argument("--words", default="lcb_words")
    a = p.parse_args()
    d = Path(a.dir)
    en_words = {w.strip() for w in open(a.words, encoding="utf-8")}
    en_words = {w for w in en_words if w.islower() and len(w) > 3}
    # per condition, task, language and source: [replies scored without line errors, of them with an English word]
    cnt = defaultdict(lambda: [0, 0])
    for line in open(d / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        lang = r["language"]
        if lang not in WPR_LANGS or r["skipped"] or not r["pass"]:
            continue
        text = r["text"].split("\nQ:")[0].strip().translate(PUNCT).replace("—", " ").replace("،", "")
        toks = []
        for ln in text.split("\n"):
            t = words(ln, lang)
            if len(t) >= 5:
                toks += t
        c = cnt[(r["cond"], r["task"], lang, r["source"])]
        c[0] += 1
        c[1] += any(t.strip() in en_words for t in toks)
    res = defaultdict(dict)
    for (cond, task, lang, src), (n, bad) in cnt.items():
        res[(cond, task)].setdefault(lang, []).append(1 - bad / max(1, n))
    conds = list(dict.fromkeys(k[0] for k in cnt))
    out, lines = {}, [f"# WPR for {d.name}", "",
                      "Share of the replies without line errors that contain no English word, averaged over sources "
                      "per language, then over languages.", "",
                      "| condition | task | " + " | ".join(WPR_LANGS) + " | mean |", "|---|---|" + "---|" * 7]
    for cond in conds:
        for task in ("monolingual", "crosslingual"):
            per = {l: float(np.mean(v)) for l, v in res[(cond, task)].items()}
            if not per:
                continue
            mean = float(np.mean(list(per.values())))
            out[f"{cond}/{task}"] = {"per_language": per, "mean": mean}
            lines.append(f"| {cond} | {task} | " + " | ".join(f"{per[l]:.2f}" if l in per else "" for l in WPR_LANGS)
                         + f" | {mean:.3f} |")
    json.dump(out, open(d / "wpr.json", "w"), indent=1)
    (d / "wpr.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
