import json
import sys
from multiprocessing import Pool

from langdetect import DetectorFactory, LangDetectException, detect


def init():
    DetectorFactory.seed = 0


def label(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def main(model="gpt2"):
    path, out = f"out/{model}/gens.jsonl", f"out/{model}/labels.json"
    runs = [json.loads(line) for line in open(path, encoding="utf-8")]
    uniq = sorted({t for r in runs for t in r["texts"]})
    with Pool(4, initializer=init) as pool:
        lab = dict(zip(uniq, pool.map(label, uniq, chunksize=200)))
    json.dump({r["cond"]: {"labels": [lab[t] for t in r["texts"]], "nll": r["nll"]} for r in runs},
              open(out, "w"))
    print(len(runs), "conditions,", len(uniq), "unique texts")


if __name__ == "__main__":
    main(*sys.argv[1:])
