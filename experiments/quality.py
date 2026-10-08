import argparse
import csv
import json
from collections import Counter

from analyze import same


def repetition(text):
    words = text.split()
    return 1 - len(set(words)) / len(words) if words else 0.0


def ngrams(text, n=4):
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def copied(prompt, text):
    grams = ngrams(text)
    return len(grams & ngrams(prompt)) / len(grams) if grams else 0.0


def kind(prompt, text):
    if repetition(text) >= 0.5:
        return "repetition"
    return "prompt copy" if copied(prompt, text) >= 0.5 else "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="name,gens.jsonl,labels.json,head")
    p.add_argument("--prompts", default="prompts/prompts_european.csv")
    p.add_argument("--out", default="results/quality.md")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.prompts, encoding="utf-8")))
    kinds = ["repetition", "prompt copy", "other"]
    lines = ["# What the outputs that stay in the prompt language look like", "",
             "Baseline outputs of non-English prompts that langdetect puts in the prompt language, split into "
             "repetition (half or more of the words repeated), prompt copy (half or more of the 4-grams taken from the "
             "prompt) and other. The last columns give the share of each kind the model's top head sends to English "
             "when removed.", "",
             "| model | outputs | repetition | prompt copy | other | head | to English: repetition / copy / other |",
             "|---|---|---|---|---|---|---|"]
    for spec in a.runs:
        name, gens, labels, head = spec.split(",")
        lab = json.load(open(labels))
        texts = {}
        for line in open(gens, encoding="utf-8"):
            r = json.loads(line)
            if r["cond"] in ("base", f"head:{head}"):
                texts[r["cond"]] = r["texts"]
            if len(texts) == 2:
                break
        base, abl = lab["base"]["labels"], lab[f"head:{head}"]["labels"]
        kept = [i for i, r in enumerate(rows) if r["language"] != "en" and same(base[i], r["language"])
                and texts["base"][i].strip()]
        k = {i: kind(rows[i]["prompt"], texts["base"][i]) for i in kept}
        n = Counter(k.values())
        flip = Counter(k[i] for i in kept if abl[i] == "en")
        lines.append(f"| {name} | {len(kept)} | " + " | ".join(f"{n[x] / len(kept):.0%}" for x in kinds) +
                     f" | {head} | " + " / ".join(f"{flip[x] / n[x]:.0%}" if n[x] else "-" for x in kinds) + " |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
