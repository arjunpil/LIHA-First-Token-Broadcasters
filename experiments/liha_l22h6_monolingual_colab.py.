# Qwen2.5-1.5B-Instruct L22H6: MONOLINGUAL phase intervention.
# Reproduction: run from a repository checkout with an NVIDIA GPU.
# The frozen manifest is read from results/qwen-l22h6-mechanism/.
# In a Colab notebook, the repository must be available in the current directory;
# an existing FP32/eager `model` and `tok` pair may be reused.
# Frozen sample: 24 prompts each for it/fr/de/es from LCB baseline records.
# Each prompt is run with five matched conditions and 100 greedy tokens.

import json
import os
import re
import string
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
# Files are addressed relative to the repository, not a public user account.
# Set LIHA_REPO_ROOT to the checkout directory when running from another path.
ROOT = Path(os.environ.get("LIHA_REPO_ROOT", "")).expanduser() if os.environ.get("LIHA_REPO_ROOT") else (Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd())
MANIFEST_PATH = ROOT / "results/qwen-l22h6-mechanism/monolingual_prompt_manifest.csv"
OUT = Path(os.environ.get("LIHA_MONOLINGUAL_OUT", str(ROOT / "out" / "qwen-l22h6-monolingual")))
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "monolingual_phase_outputs.csv"
SEED = 20261008
PER_LANGUAGE = 24
MAX_NEW_TOKENS = 100
LANGUAGES = ("it", "fr", "de", "es")
CONDITIONS = ("clean", "L22H6_prefill", "L22H6_decode", "L22H6_all", "L22H8_control")
PUNCT = str.maketrans("", "", string.punctuation)

assert torch.cuda.is_available(), "Use Runtime > Change runtime type > GPU in Colab."

# Rerun the exact prompts from the archived manifest. The original sampling
# procedure and source provenance are recorded in MONOLINGUAL.md; do not
# resample a mutable baseline data file when reproducing the saved results.
if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        f"Missing frozen prompt manifest: {MANIFEST_PATH}. "
        "Run this script from a checkout containing results/qwen-l22h6-mechanism/. "
        "For notebooks, set LIHA_REPO_ROOT to the checkout directory."
    )
manifest = pd.read_csv(MANIFEST_PATH)
required_columns = {"prompt_id", "language", "source", "prompt", "upstream_baseline_pass", "upstream_baseline_skipped"}
assert required_columns.issubset(manifest.columns)
assert len(manifest) == 96 and manifest.prompt.nunique() == 96
assert manifest.prompt_id.is_unique and set(manifest.prompt_id.astype(int)) == set(range(96))
assert set(manifest.language) == set(LANGUAGES)
assert all((manifest.language == lang).sum() == PER_LANGUAGE for lang in LANGUAGES)
manifest.to_csv(OUT / "monolingual_prompt_manifest.csv", index=False)
print("Loaded 96 frozen monolingual LCB prompts from the repository manifest.", flush=True)
print("Source counts by language:\n", pd.crosstab(manifest.language, manifest.source))

names_pattern = re.compile(r"\b(Italian|French|German|Spanish)\b", re.I)
mentions = int(manifest.prompt.map(lambda p: bool(names_pattern.search(p))).sum())
print(f"Prompts mentioning any of the four language names: {mentions}/96")
if mentions:
    print("Inspect language-name mentions before claiming this set has no language words.")

# Reuse the existing FP32 Qwen model and tokenizer where possible.
existing_model = globals().get("model")
existing_tok = globals().get("tok")
if existing_model is None or existing_tok is None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Qwen2.5-1.5B-Instruct in float32/eager mode...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to("cuda")
else:
    model, tok = existing_model, existing_tok
    observed_name = getattr(model.config, "_name_or_path", "")
    if MODEL_ID not in observed_name:
        raise RuntimeError("Loaded model is not Qwen2.5-1.5B-Instruct: " + observed_name)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Reuse requires a model loaded with attn_implementation='eager'. Reload the model with eager attention.")

model = model.float().to("cuda").eval()
assert next(model.parameters()).dtype == torch.float32
assert model.config.num_hidden_layers > 22
proj = model.model.layers[22].self_attn.o_proj
num_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // num_heads

# Reuse the loaded fastText model, or a local lid.176.bin if available.
lid = globals().get("lid")
if lid is None or not hasattr(lid, "f"):
    try:
        import fasttext
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "fasttext-community==0.11.8"])
        import fasttext
    candidates = [
        OUT / "lid.176.bin",
        ROOT / "lid.176.bin",
        ROOT / "out/qwen-l22h6-mechanism/lid.176.bin",
        Path("/content/lid.176.bin"),
    ]
    lid_file = next((p for p in candidates if p.is_file() and p.stat().st_size > 1000000), None)
    if lid_file is None:
        lid_file = candidates[1]
        print("Downloading fastText lid.176.bin (one time)...", flush=True)
        urlretrieve(LID_URL, str(lid_file))
    lid = fasttext.load_model(str(lid_file))


def check_language(text, language):
    cleaned = (text.split("\nQ:")[0].strip().translate(PUNCT)
               .replace("—", " ").replace("،", ""))
    lines = [line for line in cleaned.split("\n") if len(line.split()) >= 5]
    if not lines:
        return {"skipped": True, "passed": False, "detected_languages": ""}
    labels = []
    for line in lines:
        (prob, label), = lid.f.predict(line + "\n", 1, 0.0, "strict")
        labels.append(label[9:] if prob > 0.3 else "unknown")
    return {"skipped": False, "passed": all(x == language for x in labels),
            "detected_languages": ",".join(labels)}


@contextmanager
def ablate_phase(head, phase):
    counts = {"prefill_calls": 0, "decode_calls": 0, "edited_calls": 0}
    if head is None:
        yield counts
        return
    span = slice(head * head_dim, (head + 1) * head_dim)
    def hook(module, args):
        x = args[0]
        which = "decode" if x.shape[1] == 1 else "prefill"
        counts[which + "_calls"] += 1
        if phase != "all" and phase != which:
            return None
        y = x.clone()
        y[..., span] = 0
        counts["edited_calls"] += 1
        return (y,) + args[1:]
    handle = proj.register_forward_pre_hook(hook)
    try:
        yield counts
    finally:
        handle.remove()


@torch.inference_mode()
def run_one(prompt, condition):
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
        strftime_now=lambda fmt: "2026-10-08",
    )
    ids = tok(formatted, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], dtype=torch.long, device="cuda")
    if condition == "clean":
        head, phase = None, "all"
    elif condition == "L22H8_control":
        head, phase = 8, "all"
    else:
        head, phase = 6, condition.removeprefix("L22H6_")
    with ablate_phase(head, phase) as counts:
        out = model.generate(
            input_ids=x, attention_mask=torch.ones_like(x),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1, use_cache=True,
            pad_token_id=tok.eos_token_id,
            temperature=None, top_p=None, top_k=None,
        )
    generated = out[0, len(ids):].tolist()
    if not generated:
        raise RuntimeError("Empty generation")
    if head is not None and counts["prefill_calls"] != 1:
        raise RuntimeError("Unexpected prefill call count: " + repr(counts))
    return tok.decode(generated, skip_special_tokens=True), generated, counts


# Resume a partially completed run if present, without mixing different prompt sets.
rows = []
if CSV_PATH.is_file():
    previous = pd.read_csv(CSV_PATH).to_dict("records")
    for row in previous:
        match = manifest.loc[manifest.prompt_id == int(row["prompt_id"])].iloc[0]
        assert row["prompt"] == match.prompt and row["language"] == match.language
        assert row["condition"] in CONDITIONS
    rows = previous
    print("Resuming from", len(rows), "saved condition rows.")
done = {(int(r["prompt_id"]), r["condition"]) for r in rows}
assert len(done) == len(rows), "Duplicate checkpoint rows"

for i, item in manifest.iterrows():
    for condition in CONDITIONS:
        if (int(item.prompt_id), condition) in done:
            continue
        reply, generated, counts = run_one(item.prompt, condition)
        rows.append({
            "prompt_id": int(item.prompt_id),
            "task": "monolingual", "language": item.language,
            "source": item.source, "prompt": item.prompt,
            "condition": condition, "reply": reply,
            "generated_tokens": len(generated),
            "first_token_id": int(generated[0]),
            **check_language(reply, item.language),
            **counts,
        })
    subset = {r["condition"]: r for r in rows if int(r["prompt_id"]) == int(item.prompt_id)}
    if len(subset) == 5:
        if int(subset["clean"]["first_token_id"]) != int(subset["L22H6_decode"]["first_token_id"]):
            raise RuntimeError("Decode-only changed the first token on prompt " + str(item.prompt_id))
        if int(subset["L22H6_decode"]["edited_calls"]) == 0:
            print("WARNING: decode-only intervention not executed for prompt", item.prompt_id)
    if (i + 1) % 8 == 0:
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
        print(f"Completed {i+1}/96 monolingual prompts; checkpoint saved.", flush=True)

result = pd.DataFrame(rows)
assert len(result) == 96 * len(CONDITIONS)
assert not result.duplicated(["prompt_id", "condition"]).any()
result.to_csv(CSV_PATH, index=False)

print("\n=== MONOLINGUAL L22H6 PHASE SUMMARY ===")
summary = (result.groupby(["language", "condition"])
           .agg(n=("passed", "size"), unscorable=("skipped", "sum"),
                passes=("passed", "sum")))
summary["pass_rate_scorable"] = summary.passes / (summary.n - summary.unscorable).replace(0, np.nan)
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
summary.to_csv(OUT / "monolingual_phase_summary.csv")

print("\n=== MONOLINGUAL PAIRED COMPARISONS (CLEAN-CORRECT, ALL CONDITIONS SCORABLE) ===")
for language in list(LANGUAGES) + ["ALL"]:
    part = result if language == "ALL" else result[result.language == language]
    pass_wide = part.pivot(index="prompt_id", columns="condition", values="passed").astype(bool)
    skip_wide = part.pivot(index="prompt_id", columns="condition", values="skipped").astype(bool)
    valid = pass_wide.loc[(~skip_wide.any(axis=1)) & pass_wide["clean"]]
    pre = valid["L22H6_prefill"]
    dec = valid["L22H6_decode"]
    pre_only = int((pre & ~dec).sum())
    dec_only = int((dec & ~pre).sum())
    discordant = pre_only + dec_only
    p = binomtest(pre_only, discordant, p=0.5).pvalue if discordant else 1.0
    print(f"{language}: n={len(valid)}, prefill={int(pre.sum())}/{len(valid)}, "
          f"decode={int(dec.sum())}/{len(valid)}, full={int(valid['L22H6_all'].sum())}/{len(valid)}, "
          f"control={int(valid['L22H8_control'].sum())}/{len(valid)}, "
          f"prefill-only/ decode-only pass={pre_only}/{dec_only}, paired p={p:.6g}")

metadata = {
    "model": MODEL_ID, "model_commit": getattr(model.config, "_commit_hash", None),
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__, "dtype": str(next(model.parameters()).dtype),
    "attention": "eager", "sample_seed": SEED, "prompts_per_language": PER_LANGUAGE,
    "task": "LCB monolingual",
    "data_provenance": "results/qwen-instruct-lcb/samples.jsonl.gz (original sample; unpinned main branch)",
    "reproduction_prompts": "results/qwen-l22h6-mechanism/monolingual_prompt_manifest.csv",
    "intervention": "L22H6 o_proj input zero; separate prefill and cached decode phases",
    "max_new_tokens": MAX_NEW_TOKENS, "selection": "source-balanced sample of saved LCB base prompts, regardless of baseline correctness",
    "important_caveat": "Exploratory; FP32 single-prompt replay versus upstream FP16 batched benchmark; report both unconditional and clean-correct rates",
}
(OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
archive = OUT.parent / "LIHA_L22H6_monolingual_phase_results.zip"
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "lid.176.bin":
            z.write(p, arcname=p.name)
print("\n=== FINISHED ===")
print("Saved:", archive)
try:
    from google.colab import files
except ImportError:
    print("Results ZIP saved locally; download it manually if needed.")
else:
    files.download(str(archive))
