import argparse
import csv
import os
from collections import Counter
from urllib.request import urlretrieve

from datasets import load_dataset

CODES = {"en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "it": "ita_Latn",
         "zh": "zho_Hans", "ru": "rus_Cyrl"}
EUROPEAN = ["en", "fr", "de", "es", "it"]
EXTENDED = ["zh", "ru"]
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
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
    "zh": ["今天的天气非常", "我想告诉你关于", "科学家们发现了", "生活中最重要的事是", "从前有一个"],
    "ru": ["Сегодня погода очень", "Я хотел бы рассказать вам о", "Учёные обнаружили, что",
           "Самое важное в жизни — это", "Однажды жил-был"],
}


def good(t):
    t = t.strip()
    return 20 <= len(t) <= 300 and sum(c.isalpha() for c in t) / max(len(t), 1) >= 0.5


def flores(split):
    ds = load_dataset("mteb/flores", split=split)
    return {lang: list(ds[code]) for lang, code in CODES.items()}


def majority(text, ft):
    from langdetect import DetectorFactory, LangDetectException, detect
    import langid
    DetectorFactory.seed = 0
    votes = []
    try:
        votes.append(detect(text))
    except LangDetectException:
        votes.append("unknown")
    votes.append(langid.classify(text)[0])
    label = ft.f.predict(text.replace("\n", " "), 1, 0.0, "strict")[0][1].replace("__label__", "")
    votes.append("zh" if label in ("zh", "cmn", "yue", "zho") else label)
    valid = [v for v in votes if v != "unknown"]
    if not valid:
        return "unknown"
    top, count = Counter(valid).most_common(1)[0]
    return top if len(valid) >= 2 and count >= 2 else valid[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="european", choices=["european", "extended"])
    p.add_argument("--per-lang", type=int, default=None)
    p.add_argument("--lid", default="lid.176.bin")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    langs = EUROPEAN if a.set == "european" else EXTENDED
    per_lang = a.per_lang or (500 if a.set == "european" else 100)
    raw = flores("devtest")
    idx = [i for i in range(min(map(len, raw.values()))) if all(good(raw[l][i]) for l in raw)]
    if a.set == "extended":
        import fasttext
        if not os.path.exists(a.lid):
            urlretrieve(LID_URL, a.lid)
        ft = fasttext.load_model(a.lid)
    rows = []
    for lang in langs:
        seen = {x.strip().lower() for x in ORIGINAL[lang]}
        rows += [(x, lang, "original") for x in ORIGINAL[lang]]
        extra = [raw[lang][i].strip() for i in idx if raw[lang][i].strip().lower() not in seen]
        extra = extra[:per_lang - len(ORIGINAL[lang])]
        if lang == "ru":
            extra = [x for x in extra if majority(x, ft) == "ru"]
        rows += [(x, lang, "flores200") for x in extra]
    with open(a.out or f"prompts_{a.set}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "language", "source"])
        w.writerows(rows)
    print(f"{len(rows)} prompts from {len(idx)} parallel flores sentences")


if __name__ == "__main__":
    main()
