#!/usr/bin/env python3
"""Exploratory Gemma-3-1B-Instruct L11H3 language-token attention and decode-edge check.

Run in an NVIDIA GPU environment. Study crosslingual LCB prompts sampled independently
of model outcomes. The analysis tests attention to requested-language tokens and
then masks that attention during cached decoding; it does NOT test language steering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import string
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import numpy as np
import torch

MODEL = "google/gemma-3-1b-it"
LAYER, HEAD, CONTROL_HEAD = 11, 3, 0
LANGUAGES = {"de": "German", "es": "Spanish", "fr": "French", "it": "Italian"}
LCB_URL = "https://raw.githubusercontent.com/for-ai/language-confusion/HEAD/test_sets.zip"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
SEED = 20261009
PUNCT = str.maketrans("", "", string.punctuation)
MODES = ("clean", "target_word_mask", "nearby_word_mask", "control_head_word_mask", "head_zero_decode")
ATTENTION_GROUPS = ("last_prompt", "early_reply", "later_reply")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("gemma_l11h3_results"))
    p.add_argument("--per-language", type=int, default=24)
    p.add_argument("--smoke", action="store_true", help="Use two prompts per language; write to a separate smoke folder")
    p.add_argument("--attention-only", action="store_true", help="Measure clean attention only (no causal interventions)")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--model-revision", default=None, help="Optional immutable Hugging Face model commit hash")
    p.add_argument("--manifest", type=Path, default=None, help="Rerun an existing frozen prompt_manifest.csv")
    p.add_argument("--seed", type=int, default=SEED)
    # defaults reproduce the Gemma-3-1B run; the same design runs on another model or head with these
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--head", type=int, default=HEAD)
    p.add_argument("--control-head", type=int, default=CONTROL_HEAD)
    p.add_argument("--lid", type=Path, default=None, help="Local fastText lid.176.bin instead of downloading it")
    return p.parse_args()


def ensure_lcb(out: Path):
    test_sets = out / "lcb" / "test_sets"
    zip_path = out / "lcb" / "test_sets.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_sets.exists():
        if not zip_path.is_file():
            print("Downloading original public LCB test sets...", flush=True)
            with urlopen(LCB_URL, timeout=90) as resp:
                zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path) as z:
            if any(Path(x).is_absolute() or ".." in Path(x).parts for x in z.namelist()):
                raise ValueError("Unsafe archive member in LCB download")
            z.extractall(test_sets.parent)
    if not (test_sets / "crosslingual").is_dir():
        raise FileNotFoundError("Expected LCB crosslingual directory at " + str(test_sets))
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.exists() else None
    return test_sets, sha


def eligible_language_name(prompt, lang):
    return bool(re.search(r"\b" + re.escape(LANGUAGES[lang]) + r"\b", prompt, flags=re.I))


def build_manifest(out: Path, per_language: int, seed: int):
    tests, corpus_hash = ensure_lcb(out.parent)
    rng = random.Random(seed)
    selected = []
    pool_counts = {}
    for lang, target_name in LANGUAGES.items():
        by_source = {}
        for f in sorted((tests / "crosslingual").glob("*/*.csv")):
            if f.stem != lang:
                continue
            rows = list(csv.DictReader(f.open(encoding="utf-8", newline="")))
            cand = [dict(language=lang, source=f.parent.name, prompt=r["prompt"])
                    for r in rows if eligible_language_name(r["prompt"], lang)]
            rng.shuffle(cand)
            if cand:
                by_source[f.parent.name] = cand
        pool_counts[lang] = {k: len(v) for k, v in by_source.items()}
        if sum(map(len, by_source.values())) < per_language:
            raise RuntimeError(f"Only {pool_counts[lang]} eligible LCB prompts for {lang}; reduce --per-language")
        # Round-robin over LCB sources, without using model outputs to select examples.
        keys = sorted(by_source)
        rng.shuffle(keys)
        picks = []
        while len(picks) < per_language:
            added = False
            for source in keys:
                if by_source[source] and len(picks) < per_language:
                    picks.append(by_source[source].pop())
                    added = True
            if not added:
                break
        selected.extend(picks)
    rng.shuffle(selected)
    for idx, item in enumerate(selected):
        item["prompt_id"] = idx
    assert len(selected) == 4 * per_language
    save_csv(out / "prompt_manifest.csv", selected, ["prompt_id", "language", "source", "prompt"])
    return selected, corpus_hash, pool_counts


def load_manifest(path):
    with path.open(encoding="utf-8", newline="") as f:
        selected = list(csv.DictReader(f))
    for i, item in enumerate(selected):
        item["prompt_id"] = int(item["prompt_id"])
        if not eligible_language_name(item["prompt"], item["language"]):
            raise ValueError(f"Missing explicit language name in manifest row {i}")
    if len({r["prompt_id"] for r in selected}) != len(selected):
        raise ValueError("Nonunique prompt IDs in manifest")
    return selected


def save_csv(path: Path, rows: list[dict], names: list[str] | None = None):
    if not rows:
        return
    if names is None:
        names = list(dict.fromkeys(k for x in rows for k in x))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def language_score(text, lang, lid):
    text = text.split("\nQ:")[0].strip().translate(PUNCT).replace("—", " ").replace("،", "")
    lines = [line for line in text.split("\n") if len(line.split()) >= 5]
    if not lines:
        return {"skipped": True, "passed": False, "labels": ""}
    labels = []
    for line in lines:
        (prob, label), = lid.f.predict(line + "\n", 1, 0.0, "strict")
        labels.append(label[9:] if prob > 0.3 else "unknown")
    return {"skipped": False, "passed": all(x == lang for x in labels), "labels": ",".join(labels)}


class GemmaExperiment:
    def __init__(self, args, out):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. In Colab, select Runtime > Change runtime type > T4/L4 GPU.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import fasttext

        self.a = args
        self.out = out
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=args.model_revision, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=args.model_revision, dtype=torch.float32, attn_implementation="eager"
        ).to("cuda").eval()
        inner = getattr(self.model.model, "language_model", self.model.model)
        self.attn = inner.layers[LAYER].self_attn
        self.proj = self.attn.o_proj
        cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.head_count = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        assert HEAD < self.head_count and CONTROL_HEAD < self.head_count
        assert getattr(cfg, "_attn_implementation", "eager") == "eager", "Require eager attention"
        self.rev = getattr(self.model.config, "_commit_hash", None) or self.tok.init_kwargs.get("_commit_hash")
        try:
            from huggingface_hub import HfApi
            self.rev = HfApi().model_info(MODEL, revision=args.model_revision).sha
        except Exception:
            pass  # Preserve the model/tokenizer's recorded hash when API is unavailable.
        lid_path = args.lid or out.parent / "lid.176.bin"
        if not lid_path.is_file():
            print("Downloading fastText language ID model...", flush=True)
            urlretrieve(LID_URL, str(lid_path))
        self.lid = fasttext.load_model(str(lid_path))

    def encode(self, prompt, lang):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        # Avoid double BOS tokenization when applying the Gemma chat template.
        if self.tok.bos_token and self.tok("a").input_ids[0] == self.tok.bos_token_id:
            rendered = rendered.removeprefix(self.tok.bos_token)
        enc = self.tok(rendered, add_special_tokens=True, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        start_user = rendered.find(prompt)
        if start_user < 0:
            raise ValueError("Prompt missing from rendered chat template")
        end_user = start_user + len(prompt)
        lang_spans = [m.span() for m in re.finditer(r"\b" + re.escape(LANGUAGES[lang]) + r"\b", rendered[start_user:end_user], re.I)]
        lang_spans = [(start_user + a, start_user + b) for a, b in lang_spans]
        target = sorted({i for i, (a, b) in enumerate(offsets) if b > a and any(a < y and b > x for x, y in lang_spans)})
        valid_user = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start_user and b <= end_user and i not in target]
        nearby = sorted(valid_user, key=lambda i: min(abs(i-j) for j in target))[:len(target)] if target else []
        if not target or len(nearby) != len(target):
            raise ValueError("Could not align requested-language and nearby tokens: " + prompt[:120])
        return ids, target, sorted(nearby), len(lang_spans)

    @contextmanager
    def patch(self, mode, targets, nearby):
        state = {"prefill_calls": 0, "decode_calls": 0, "edited_calls": 0}
        if mode == "clean":
            yield state
            return
        if mode == "head_zero_decode":
            start, end = HEAD * self.head_dim, (HEAD+1) * self.head_dim
            def zero_hook(module, args):
                x = args[0]
                if x.shape[1] != 1:
                    state["prefill_calls"] += 1
                    return None
                state["decode_calls"] += 1
                y = x.clone()
                y[..., start:end] = 0
                state["edited_calls"] += 1
                return (y,) + args[1:]
            handle = self.proj.register_forward_pre_hook(zero_hook)
        else:
            head = CONTROL_HEAD if mode == "control_head_word_mask" else HEAD
            keys = nearby if mode == "nearby_word_mask" else targets
            def edge_hook(module, inputs, kwargs):
                hidden = kwargs.get("hidden_states", inputs[0] if inputs else None)
                if hidden is None:
                    raise RuntimeError("Gemma attention hook received no hidden states")
                if hidden.shape[1] != 1:  # prefill is never modified
                    state["prefill_calls"] += 1
                    return None
                state["decode_calls"] += 1
                mask = kwargs.get("attention_mask")
                if mask is None or mask.ndim != 4 or mask.shape[1] not in (1, self.head_count):
                    raise RuntimeError("Expected 4-D additive mask for Gemma eager attention; inspect Transformers version")
                if max(keys) >= mask.shape[-1]:
                    raise RuntimeError("Target prompt position is not present in decoding cache")
                m = mask.expand(mask.shape[0], self.head_count, mask.shape[2], mask.shape[3]).clone()
                m[:, head, :, keys] = torch.finfo(m.dtype).min
                kwargs["attention_mask"] = m
                state["edited_calls"] += 1
                return inputs, kwargs
            handle = self.attn.register_forward_pre_hook(edge_hook, with_kwargs=True)
        try:
            yield state
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(self, ids, mode, targets, nearby):
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with self.patch(mode, targets, nearby) as state:
            seq = self.model.generate(
                input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=self.a.max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True, pad_token_id=self.tok.eos_token_id,
                temperature=None, top_p=None, top_k=None,
            )
        if mode != "clean" and (state["prefill_calls"] != 1 or state["edited_calls"] < 1):
            raise RuntimeError("No causal intervention was applied: " + str((mode, state)))
        reply_ids = seq[0, x.shape[1]:].tolist()
        return self.tok.decode(reply_ids, skip_special_tokens=True), reply_ids, dict(state)

    @torch.inference_mode()
    def attention_metrics(self, ids, reply_ids, targets, nearby):
        # Teacher-forcing the model's own clean generation avoids the token-offset drift
        # that may arise from decoding and retokenizing generated text.
        n = len(ids)
        n_reply = min(32, len(reply_ids))
        if n_reply < 4:
            return {"attention_status": "too_short"}
        full = torch.tensor([ids + reply_ids[:n_reply]], device="cuda", dtype=torch.long)
        out = self.model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False, output_attentions=True)
        if out.attentions is None or out.attentions[LAYER] is None:
            raise RuntimeError("Model returned no attention matrices; eager output_attentions=True is required")
        att = out.attentions[LAYER][0].float()
        if att.shape[0] != self.head_count or not torch.isfinite(att).all():
            raise RuntimeError(f"Unexpected/nonfinite attention: {att.shape}")
        groups = {
            "last_prompt": [n - 1],
            "early_reply": list(range(n, min(n + 8, n+n_reply-1))),
            "later_reply": list(range(n+8, n+n_reply-1)),
        }
        data = {"attention_status": "ok", "n_reply_tokens_for_attention": n_reply}
        for group, queries in groups.items():
            if not queries:
                continue
            for h in range(self.head_count):
                data[f"{group}_h{h}_target_mass"] = float(att[h, queries][:, targets].sum(-1).mean().item())
            data[f"{group}_h{HEAD}_nearby_mass"] = float(att[HEAD, queries][:, nearby].sum(-1).mean().item())
        del out, att
        return data


def summarize(out, manifest, rows, attentions):
    counts = []
    for lang in [*LANGUAGES, "ALL"]:
        for mode in MODES:
            rs = [r for r in rows if r["condition"] == mode and (lang == "ALL" or r["language"] == lang)]
            if not rs:
                continue
            scorable = [r for r in rs if not r["skipped"]]
            counts.append({"language": lang, "condition": mode, "n": len(rs),
                           "scorable": len(scorable), "passes": sum(bool(r["passed"]) for r in scorable),
                           "pass_rate_scorable": round(sum(bool(r["passed"]) for r in scorable)/len(scorable), 6) if scorable else ""})
    save_csv(out / "summary.csv", counts)
    detail = {}
    for lang in [*LANGUAGES, "ALL"]:
        ars = [r for r in attentions if r["attention_status"] == "ok" and (lang == "ALL" or r["language"] == lang)]
        d = {"language": lang, "n": len(ars)}
        for group in ATTENTION_GROUPS:
            available = sorted(k for r in ars for k in r if k.startswith(group + "_") and k.endswith("_mass"))
            for key in sorted(set(available)):
                vals = [float(r[key]) for r in ars if key in r and r[key] not in ("", None)]
                d[key + "_mean"] = round(float(np.mean(vals)), 5) if vals else ""
        detail[lang] = d
    save_csv(out / "attention_summary.csv", list(detail.values()))
    if not rows or set(r["condition"] for r in rows) == {"clean"}:
        return
    group = {}
    for r in rows:
        group.setdefault(r["prompt_id"], {})[r["condition"]] = r
    ca = []
    for pid, d in group.items():
        base = d["clean"]
        for name, v in d.items():
            if name == "clean":
                continue
            ca.append({"prompt_id": pid, "language": base["language"], "condition": name,
                       "baseline_pass": base["passed"], "intervention_pass": v["passed"],
                       "baseline_scorable": not base["skipped"], "intervention_scorable": not v["skipped"],
                       "new_failure_on_baseline_correct": bool(base["passed"] and not v["passed"]),
                       "first_token_matches_clean": base["first_token_id"] == v["first_token_id"]})
    save_csv(out / "paired_outcomes.csv", ca)


def main():
    global MODEL, LAYER, HEAD, CONTROL_HEAD
    args = arguments()
    MODEL, LAYER, HEAD, CONTROL_HEAD = args.model, args.layer, args.head, args.control_head
    if args.per_language < 1 or args.max_new_tokens < 4:
        raise ValueError("Need at least one prompt per language and four generation tokens")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "smoke" if args.smoke else args.out / "full"
    out.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest = load_manifest(args.manifest)
        if args.smoke:
            ids = {lang:0 for lang in LANGUAGES}
            small = []
            for item in manifest:
                if ids[item["language"]] < 2:
                    small.append(item)
                    ids[item["language"]] += 1
            manifest = small
        corpus_hash, pools = None, None
        save_csv(out / "prompt_manifest.csv", manifest, ["prompt_id", "language", "source", "prompt"])
    else:
        manifest, corpus_hash, pools = build_manifest(out, 2 if args.smoke else args.per_language, args.seed)
    if args.smoke:
        assert len(manifest) == 8
    print(f"Selected {len(manifest)} explicit-language LCB crosslingual prompts, from {len(LANGUAGES)} languages", flush=True)
    exp = GemmaExperiment(args, out)
    try:
        model_cfg = exp.model.config.get_text_config() if hasattr(exp.model.config, "get_text_config") else exp.model.config
        metadata = {
            "model_id": MODEL, "requested_revision": args.model_revision, "resolved_revision": exp.rev,
            "transformers": __import__("transformers").__version__, "torch": torch.__version__,
            "dtype": "torch.float32", "attention_backend": "eager", "layer": LAYER, "head": HEAD,
            "control_head": CONTROL_HEAD, "head_dim": exp.head_dim, "n_query_heads": exp.head_count,
            "sample_seed": args.seed, "sample_per_language": 2 if args.smoke else args.per_language,
            "LCB_source_url": LCB_URL, "LCB_archive_sha256": corpus_hash,
            "LCB_source_pool_counts": pools, "max_new_tokens": args.max_new_tokens,
            "attention_only": args.attention_only,
            "limitations": ["fresh LCB prompts selected based only on explicit language names, not model outcomes",
                            f"L{LAYER}H{HEAD} and L{LAYER}H{CONTROL_HEAD} are query attention heads",
                            "masking starts in cached decoding; first token remains unchanged by design",
                            "LCB baseline accuracy can be low; report baseline-conditioned failures",
                            "measuring attention to language names does not establish semantic use of those tokens"]
        }
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results, attention_records = [], []
        modes = ("clean",) if args.attention_only else MODES
        for i, row in enumerate(manifest, 1):
            ids, targets, nearby, occurrences = exp.encode(row["prompt"], row["language"])
            current = {}
            for mode in modes:
                text, reply_ids, counts = exp.generate(ids, mode, targets, nearby)
                score = language_score(text, row["language"], exp.lid)
                current[mode] = {
                    "prompt_id": row["prompt_id"], "language": row["language"], "source": row["source"],
                    "condition": mode, "reply": text, "generated_tokens": len(reply_ids),
                    "first_token_id": reply_ids[0] if reply_ids else "", "n_target_tokens": len(targets),
                    "n_nearby_tokens": len(nearby), "n_language_mentions": occurrences,
                    **score, **counts,
                }
                results.append(current[mode])
                if mode == "clean":
                    metrics = exp.attention_metrics(ids, reply_ids, targets, nearby)
                    attention_records.append({"prompt_id": row["prompt_id"], "language": row["language"],
                        "source": row["source"], "n_target_tokens": len(targets),
                        "n_nearby_tokens": len(nearby), **metrics})
            if not args.attention_only and len({v["first_token_id"] for v in current.values()}) != 1:
                raise RuntimeError(f"First generated token differs under decode-only intervention for prompt {row['prompt_id']}")
            if i % 4 == 0 or i == len(manifest):
                save_csv(out / "condition_outputs.csv", results)
                save_csv(out / "attention_by_prompt.csv", attention_records)
                summarize(out, manifest, results, attention_records)
                print(f"Completed {i}/{len(manifest)} prompts", flush=True)
        print("\nAttention summary and score summary:", flush=True)
        for record in csv.DictReader((out / "attention_summary.csv").open(encoding="utf-8")):
            if record["language"] in ("it", "fr", "de", "es", "ALL"):
                print(record["language"], f"last prompt H{HEAD} target mass:",
                      record.get(f"last_prompt_h{HEAD}_target_mass_mean"),
                      "nearby:", record.get(f"last_prompt_h{HEAD}_nearby_mass_mean"))
        print((out / "summary.csv").read_text()[:3500])
    finally:
        import gc
        del exp
        gc.collect()
        torch.cuda.empty_cache()
    archive = args.out / ("gemma_l11h3_smoke_results.zip" if args.smoke else "gemma_l11h3_full_results.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob("*")):
            if f.is_file():
                z.write(f, arcname=f.name)
    print("\nSaved:", archive.resolve(), flush=True)


if __name__ == "__main__":
    main()
