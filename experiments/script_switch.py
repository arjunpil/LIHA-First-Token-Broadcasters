"""Script switches in the saved 14-language LCB runs (nothing is regenerated).

LPR only scores lines of 5+ space-separated words, so Chinese/Japanese lines (no spaces) are never scored.
A reply that switches into Han/kana script is then either skipped (no scorable line left) or scored only on
the lines that kept the expected script, so it can still pass. This counts, per language and task:
  - switched: replies with more Han/kana characters than characters of the expected script
  - how the switched replies were scored: skipped / passing / failing
  - mostly Han/kana: switched replies where Han/kana is the largest script in the reply (mixed-script
    replies, e.g. a Chinese first line followed by English, are left out here)
  - pass rates: as LPR treats skipped replies (left out), and with skipped replies counted as failures
zh and ja are left out (Han/kana is their own script). Rows are shown when the ablated condition has at
least MIN_ROW switched replies, plus all Korean rows.

Usage: python experiments/script_switch.py qwen-instruct-lcb-all qwen2.5-3b-instruct-lcb-all ...
Writes results/script-switch/summary.md
"""
import gzip
import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[一-鿿぀-ヿ]")
SCRIPTS = {
    "hangul": re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]"),
    "arabic": re.compile(r"[؀-ۿ]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "latin": re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]"),
}
OWN = {"ko": "hangul", "ar": "arabic", "ru": "cyrillic", "hi": "devanagari"}  # all others: latin
MIN_ROW = 3


def switched(text, lang):
    return len(CJK.findall(text)) > len(SCRIPTS[OWN.get(lang, "latin")].findall(text))


def mostly_cjk(text):
    n = len(CJK.findall(text))
    return n > 0 and all(n > len(p.findall(text)) for p in SCRIPTS.values())


def stats(rows):
    n = len(rows)
    scored = [r for r in rows if not r.get("skipped")]
    passed = sum(bool(r.get("pass")) for r in scored)
    sw = [r for r in rows if switched(r["text"], r["language"])]
    sw_skip = sum(bool(r.get("skipped")) for r in sw)
    sw_pass = sum(bool(r.get("pass")) for r in sw if not r.get("skipped"))
    return {"n": n, "skipped": n - len(scored), "sw": len(sw), "sw_skip": sw_skip, "sw_pass": sw_pass,
            "sw_fail": len(sw) - sw_skip - sw_pass, "mostly": sum(mostly_cjk(r["text"]) for r in sw),
            "lpr": passed / len(scored) if scored else float("nan"), "lpr_skip_fail": passed / n if n else float("nan")}


def main():
    lines = ["# Script switches in the 14-language LCB runs", "",
             "From the saved replies; nothing regenerated. Switched: more Han/kana characters than characters of the expected script. "
             "Mostly Han/kana: Han/kana is the largest script in the reply. "
             "Pass rates are simple shares (not averaged over sources like LPR): the first leaves skipped replies out as LPR does, the second counts them as failures. "
             f"Rows: all Korean rows, and other languages where the ablated condition has at least {MIN_ROW} switched replies (zh, ja left out).", "",
             "| model | head | lang | task | condition | skipped | switched | switched: skipped / passing / failing | mostly Han/kana | pass rate (skipped left out) | pass rate (skipped = fail) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in sys.argv[1:]:
        rows = [json.loads(l) for l in gzip.open(Path("results") / run / "samples.jsonl.gz", "rt", encoding="utf-8")]
        head = next(c for c in sorted({r["cond"] for r in rows}) if c.endswith(" zero") and "control" not in c)
        langs = sorted({r["language"] for r in rows} - {"zh", "ja"}, key=lambda l: (l != "ko", l))
        for lang in langs:
            for task in ("monolingual", "crosslingual"):
                pick = lambda cond: [r for r in rows if r["cond"] == cond and r["language"] == lang and r["task"] == task]
                b, z = stats(pick("base")), stats(pick(head))
                if not z["n"] or (lang != "ko" and z["sw"] < MIN_ROW):
                    continue
                for cond, s in (("base", b), (head, z)):
                    lines.append(f"| {run.removesuffix('-lcb-all')} | {head.split()[0]} | {lang} | {task} | {cond} | "
                                 f"{s['skipped']}/{s['n']} | {s['sw']}/{s['n']} | {s['sw_skip']} / {s['sw_pass']} / {s['sw_fail']} | "
                                 f"{s['mostly']} | {s['lpr']:.2f} | {s['lpr_skip_fail']:.2f} |")
    out = Path("results/script-switch")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
