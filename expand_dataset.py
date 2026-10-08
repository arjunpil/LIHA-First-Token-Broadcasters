"""
expand_dataset.py v2 — 500 European / 100 zh+ru prompts, majority-vote detection.

Install first:
  py -3.12 -m pip install datasets langid fasttext

Run:
  py -3.12 expand_dataset.py
"""

import os, sys, urllib.request
import pandas as pd
from collections import Counter
from langdetect import detect, LangDetectException, DetectorFactory
DetectorFactory.seed = 0

DATA_DIR        = "data"
EUROPEAN_TARGET = 500
EXTENDED_TARGET = 100
MIN_CHARS       = 20
MAX_CHARS       = 300
os.makedirs(DATA_DIR, exist_ok=True)

FLORES_CODES = {
    "en":"eng_Latn","fr":"fra_Latn","de":"deu_Latn",
    "es":"spa_Latn","it":"ita_Latn","zh":"zho_Hans","ru":"rus_Cyrl",
}
EUROPEAN_LANGS = ["en","fr","de","es","it"]
EXTENDED_LANGS = ["zh","ru"]
ALL_LANGS      = EUROPEAN_LANGS + EXTENDED_LANGS

ORIGINAL_PROMPTS = {
    "en":["The weather today is very","I would like to tell you about",
          "Scientists have discovered that","The most important thing in life is",
          "Once upon a time there was a"],
    "fr":["Le temps aujourd'hui est très","Je voudrais vous parler de",
          "Les scientifiques ont découvert que",
          "La chose la plus importante dans la vie est","Il était une fois un"],
    "de":["Das Wetter heute ist sehr","Ich möchte Ihnen über",
          "Wissenschaftler haben entdeckt, dass","Das Wichtigste im Leben ist",
          "Es war einmal ein"],
    "es":["El tiempo hoy es muy","Me gustaría hablarle sobre",
          "Los científicos han descubierto que","Lo más importante en la vida es",
          "Había una vez un"],
    "it":["Il tempo oggi è molto","Vorrei parlarvi di",
          "Gli scienziati hanno scoperto che",
          "La cosa più importante nella vita è","C'era una volta un"],
    "zh":["今天的天气非常","我想告诉你关于","科学家们发现了",
          "生活中最重要的事是","从前有一个"],
    "ru":["Сегодня погода очень","Я хотел бы рассказать вам о",
          "Учёные обнаружили, что","Самое важное в жизни — это",
          "Однажды жил-был"],
}

# ── Majority-vote detection ───────────────────────────────────────────────────
_ft_model = None

def load_fasttext_model():
    global _ft_model
    if _ft_model is not None:
        return _ft_model

    # Try fasttext or fasttext-wheel (same API)
    try:
        import fasttext
        path = os.path.join(DATA_DIR, "lid.176.bin")
        if not os.path.exists(path):
            print("  Downloading fastText lid model (~130MB)...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                path
            )
        _ft_model = fasttext.load_model(path)
        print("  fasttext loaded OK")
        return _ft_model
    except ImportError:
        pass
    except Exception as e:
        print(f"  fasttext load failed: {e}")

    # Try fasttext-langdetect (different API, wraps fasttext)
    try:
        from fastlangid.langid import LID
        _ft_model = LID()
        print("  fasttext-langdetect loaded OK")
        return _ft_model
    except ImportError:
        pass

    print("  fasttext unavailable — using 2-way vote (langdetect + langid)")
    return None

def detect_majority(text, expected_lang):
    votes = []
    # langdetect
    try:
        votes.append(detect(text))
    except LangDetectException:
        votes.append("unknown")
    # langid
    try:
        import langid
        lang, _ = langid.classify(text)
        votes.append(lang)
    except ImportError:
        votes.append("unknown")
    # fasttext (handles both fasttext and fasttext-langdetect APIs)
    ft = load_fasttext_model()
    if ft is not None:
        try:
            # Standard fasttext API
            if hasattr(ft, "predict"):
                pred = ft.predict(text.replace("\n"," "), k=1)
                if isinstance(pred[0][0], str):
                    ft_lang = pred[0][0].replace("__label__","")
                else:
                    ft_lang = pred[0][0]
            # fasttext-langdetect API
            elif hasattr(ft, "predict_lang"):
                ft_lang = ft.predict_lang(text)
            else:
                ft_lang = "unknown"
            if ft_lang in ("zh","cmn","yue","zho"):
                ft_lang = "zh"
            votes.append(ft_lang)
        except Exception:
            votes.append("unknown")

    valid = [v for v in votes if v != "unknown"]
    if not valid:
        return "unknown", "all_failed"
    c = Counter(valid)
    top_lang, top_count = c.most_common(1)[0]
    if len(valid) >= 2 and top_count >= 2:
        return top_lang, f"majority_{top_count}/{len(valid)}"
    return valid[0], "single"

# ── Flores loading ────────────────────────────────────────────────────────────
def load_all_flores():
    """Load all languages at once from mteb/flores (Parquet, no script needed).

    Returns dict: {lang_short: [sentence, ...]}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("  Loading mteb/flores (devtest split)...")
    ds = load_dataset("mteb/flores", split="devtest")
    print(f"  Got {len(ds)} rows")

    raw = {}
    for lang, flores_code in FLORES_CODES.items():
        if flores_code in ds.column_names:
            sents = [row[flores_code] for row in ds]
            print(f"  {lang} ({flores_code}): {len(sents)} sentences")
            raw[lang] = sents
        else:
            print(f"  WARNING: column {flores_code} not found!")
            raw[lang] = []
    return raw

def is_good(text):
    t = text.strip()
    if len(t) < MIN_CHARS or len(t) > MAX_CHARS:
        return False
    return sum(c.isalpha() for c in t) / max(len(t),1) >= 0.5

def find_parallel(flores_raw):
    print("\nFinding parallel indices...")
    min_len = min(len(s) for s in flores_raw.values())
    idx = [i for i in range(min_len)
           if all(is_good(flores_raw[l][i]) for l in flores_raw)]
    print(f"  {len(idx)} valid parallel indices")
    return {l: [flores_raw[l][i] for i in idx] for l in flores_raw}

def build_dataset(lang, sentences, target):
    orig = ORIGINAL_PROMPTS.get(lang, [])
    orig_set = {p.strip().lower() for p in orig}
    rows = [{"prompt":p,"language":lang,"source":"original"} for p in orig]
    added = 0
    for s in sentences:
        if added >= target - len(orig):
            break
        s = s.strip()
        if s.lower() not in orig_set:
            rows.append({"prompt":s,"language":lang,"source":"flores200"})
            added += 1
    df = pd.DataFrame(rows)
    print(f"  {lang}: {len(orig)} orig + {added} flores = {len(df)} total")
    if len(df) < target:
        print(f"  WARNING: only {len(df)} (target {target})")
    return df

def verify(df, lang):
    detected, methods, verified = [], [], []
    for p in df["prompt"]:
        d, m = detect_majority(str(p), lang)
        detected.append(d); methods.append(m); verified.append(d==lang)
    df = df.copy()
    df["detected"] = detected
    df["detection_method"] = methods
    df["verified"] = verified
    flores_rows = df[df["source"]=="flores200"]
    if len(flores_rows):
        rate = flores_rows["verified"].mean()
        bad  = (~flores_rows["verified"]).sum()
        print(f"  {lang}: {rate:.1%} pass ({bad} mismatches)")
        if bad > 0 and lang in EXTENDED_LANGS:
            if lang == "zh":
                print(f"  Keeping {bad} zh mismatches (detector unreliable for Chinese)")
            else:
                df = df.drop(flores_rows[~flores_rows["verified"]].index).reset_index(drop=True)
                print(f"  Removed {bad} misdetected rows")
    return df

def main():
    print("="*60)
    print("expand_dataset.py v2 — 500 EU / 100 zh+ru / majority-vote")
    print("="*60)

    print("\n[1] Loading Flores-200...")
    raw = load_all_flores()
    missing = [l for l in ALL_LANGS if not raw.get(l)]
    if missing:
        raise RuntimeError(f"Failed: {missing}")

    print("\n[2] Parallel alignment...")
    parallel = find_parallel(raw)

    print("\n[3] Building datasets...")
    all_dfs = []
    for lang in EUROPEAN_LANGS:
        print(f"\n  [{lang.upper()}] target={EUROPEAN_TARGET}")
        df = build_dataset(lang, parallel[lang], EUROPEAN_TARGET)
        df = verify(df, lang)
        p = os.path.join(DATA_DIR, f"prompts_{lang}.csv")
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"  Saved → {p}")
        all_dfs.append(df)

    for lang in EXTENDED_LANGS:
        print(f"\n  [{lang.upper()}] target={EXTENDED_TARGET}")
        df = build_dataset(lang, parallel[lang], EXTENDED_TARGET)
        df = verify(df, lang)
        p = os.path.join(DATA_DIR, f"prompts_{lang}.csv")
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"  Saved → {p}")
        all_dfs.append(df)

    print("\n[4] Writing combined file...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR,"all_prompts.csv"),
                    index=False, encoding="utf-8")

    print("\n"+"="*60)
    print("SUMMARY")
    print("="*60)
    print(combined.groupby(["language","source"]).size().unstack(fill_value=0).to_string())
    print(f"\nTotal: {len(combined)} prompts")
    print("\nVerification (flores rows):")
    for l in ALL_LANGS:
        sub = combined[(combined["language"]==l)&(combined["source"]=="flores200")]
        if len(sub):
            print(f"  {l}: {sub['verified'].mean():.1%} ({sub['verified'].sum()}/{len(sub)})")

    print("""
Next steps:
  py -3.12 experiment.py
  py -3.12 multi_ablation.py
  py -3.12 per_language.py
  py -3.12 confidence_intervals.py
  py -3.12 random_baseline.py
  py -3.12 extended_languages.py
  py -3.12 bloom_experiment.py
  py -3.12 probing_experiment.py
""")

if __name__ == "__main__":
    main()