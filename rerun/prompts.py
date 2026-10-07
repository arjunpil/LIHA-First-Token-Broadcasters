import argparse
import csv

from datasets import load_dataset

CODES = {"en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
         "zh": "zho_Hans", "ru": "rus_Cyrl"}
EUROPEAN = ["en", "fr", "de", "es", "it"]
ORIGINAL = {
    "en": ["The weather today is very", "I would like to tell you about", "Scientists have discovered that",
           "The most important thing in life is", "Once upon a time there was a"],
    "fr": ["Le temps aujourd'hui est très", "Je voudrais vous parler de", "Les scientifiques ont découvert que",
           "La chose la plus importante dans la vie est", "Il était une fois un"],
    "de": ["Das Wetter heute ist sehr", "Ich möchte Ihnen über", "Wissenschaftler haben entdeckt, dass",
           "Das Wichtigste im Leben ist", "Es war einmal ein"],
    "es": ["El tiempo hoy es muy", "Me gustaría hablarle sobre", "Los científicos han descubierto que",
           "Lo más importante en la vida es", "Había una vez un"],
    "it": ["Il tempo oggi è molto", "Vorrei parlarvi di", "Gli scienziati hanno scoperto che",
           "La cosa più importante nella vita è", "C'era una volta un"],
}


def good(t):
    t = t.strip()
    return 20 <= len(t) <= 300 and sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.5


def flores(split):
    ds = load_dataset("mteb/flores", split=split)
    return {lang: list(ds[code]) for lang, code in CODES.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=500)
    p.add_argument("--out", default="prompts_european.csv")
    a = p.parse_args()
    raw = flores("devtest")
    idx = [i for i in range(min(map(len, raw.values()))) if all(good(raw[l][i]) for l in raw)]
    rows = []
    for lang in EUROPEAN:
        seen = {x.strip().lower() for x in ORIGINAL[lang]}
        rows += [(x, lang, "original") for x in ORIGINAL[lang]]
        extra = [raw[lang][i].strip() for i in idx if raw[lang][i].strip().lower() not in seen]
        rows += [(x, lang, "flores200") for x in extra[:a.per_lang - len(ORIGINAL[lang])]]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "language", "source"])
        w.writerows(rows)
    print(f"{len(rows)} prompts from {len(idx)} parallel flores sentences")


if __name__ == "__main__":
    main()
