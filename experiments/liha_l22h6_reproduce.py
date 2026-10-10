#!/usr/bin/env python3
"""Reproduce Qwen2.5-1.5B-Instruct L22H6 intervention analyses.

Run from the repository root. See the companion results README for the
experimental protocol, validation scope, and limitations.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch


LANGUAGES = {
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}
TOPICS = [
    "Explain three ways public libraries can help students.",
    "Describe how a community garden can help a neighborhood.",
    "Explain the benefits and limitations of solar energy.",
    "Write a short introduction to neural networks.",
    "Explain how wetlands help local wildlife.",
    "Describe practical ways to save water at home.",
    "Give advice to someone organizing a small science fair.",
    "Explain the difference between weather and climate.",
]

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER, TARGET, CONTROL = 22, 6, 8
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
PUNCT = str.maketrans("", "", string.punctuation)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["phase", "edge", "attention", "semantic", "logit", "all"],
        default="all",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/qwen-l22h6-mechanism/prompt_manifest_96.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("out/qwen-l22h6-mechanism")
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model commit; the exploratory run did not record one",
    )
    parser.add_argument(
        "--lid",
        type=Path,
        default=Path("out/qwen-l22h6-mechanism/lid.176.bin"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--smoke", action="store_true", help="Use two prompts per language"
    )
    return parser.parse_args()


class Experiment:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled PyTorch installation is required")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.a = args
        self.a.out.mkdir(parents=True, exist_ok=True)
        self.tok = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cuda").eval()
        assert self.model.config._attn_implementation == "eager"

        self.attn = self.model.model.layers[LAYER].self_attn
        self.proj = self.attn.o_proj
        self.head_count = self.model.config.num_attention_heads
        self.dim = getattr(self.model.config, "head_dim", None) or (
            self.model.config.hidden_size // self.head_count
        )
        self.span = slice(TARGET * self.dim, (TARGET + 1) * self.dim)

        if not args.lid.exists():
            print("Downloading fastText lid.176.bin...", flush=True)
            args.lid.parent.mkdir(exist_ok=True, parents=True)
            urlretrieve(LID_URL, str(args.lid))
        import fasttext

        self.lid = fasttext.load_model(str(args.lid))

    def formatted(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            strftime_now=lambda fmt: "2026-10-08",
        )

    def positions(
        self, prompt: str, requested: str, incidental: str | None = None
    ):
        full = self.formatted(prompt)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]

        def find_word(name):
            spans = [
                match.span()
                for match in re.finditer(r"\b" + re.escape(name) + r"\b", full, re.I)
            ]
            if not spans:
                raise ValueError(
                    "Language word absent from prompt: "
                    + repr(name)
                    + ", "
                    + prompt[:100]
                )
            return [
                i
                for i, (start, end) in enumerate(offsets)
                if end > start
                and any(start < word_end and end > word_start
                        for word_start, word_end in spans)
            ]

        requested_pos = find_word(LANGUAGES[requested])
        if incidental is not None:
            control_pos = find_word(LANGUAGES[incidental])
        else:
            eligible = [
                i
                for i, (start, end) in enumerate(offsets)
                if end > start and i not in requested_pos
            ]
            control_pos = sorted(
                eligible,
                key=lambda i: min(abs(i - j) for j in requested_pos),
            )[:len(requested_pos)]

        if not requested_pos or not control_pos:
            raise ValueError("Missing requested or control position")
        return enc["input_ids"], requested_pos, control_pos

    def score(self, text: str, lang: str):
        """Apply the line-level LCB language-consistency scoring rule."""
        cleaned = (
            text.split("\nQ:")[0]
            .strip()
            .translate(PUNCT)
            .replace("—", " ")
            .replace("،", "")
        )
        lines = [line for line in cleaned.split("\n") if len(line.split()) >= 5]
        if not lines:
            return {"skipped": True, "passed": False, "detected_languages": ""}

        labels = []
        for line in lines:
            (prob, label), = self.lid.f.predict(line + "\n", 1, 0.0, "strict")
            labels.append(label[9:] if prob > 0.3 else "unknown")
        return {
            "skipped": False,
            "passed": all(label == lang for label in labels),
            "detected_languages": ",".join(labels),
        }

    @contextmanager
    def phase_ablation(self, head: int | None, phase: str):
        """Zero one head at the output projection input during selected phases."""
        counts = {"prefill_calls": 0, "decode_calls": 0, "edited_calls": 0}
        if head is None:
            yield counts
            return
        span = slice(head * self.dim, (head + 1) * self.dim)

        def hook(module, args):
            x = args[0]
            current = "decode" if x.shape[1] == 1 else "prefill"
            counts[current + "_calls"] += 1
            if phase == "all" or phase == current:
                out = x.clone()
                out[..., span] = 0
                counts["edited_calls"] += 1
                return (out,) + args[1:]
            return None

        handle = self.proj.register_forward_pre_hook(hook)
        try:
            yield counts
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate_phase(self, pids: list[int], head: int | None, phase: str):
        x = torch.tensor([pids], dtype=torch.long, device="cuda")
        with self.phase_ablation(head, phase) as counts:
            seq = self.model.generate(
                input_ids=x,
                attention_mask=torch.ones_like(x),
                max_new_tokens=self.a.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )

        if head is not None:
            if counts["prefill_calls"] != 1 or counts["edited_calls"] < 1:
                raise RuntimeError("Unexpected phase-ablation hook counts: "
                                   + repr(counts))
        ids = seq[0, len(pids):].tolist()
        return self.tok.decode(ids, skip_special_tokens=True), ids, counts

    def phase(self, manifest):
        """Compare full, prefill-only, and decode-only head interventions."""
        conditions = [
            ("clean", None, "all"),
            ("L22H6_all", TARGET, "all"),
            ("L22H6_prefill", TARGET, "prefill"),
            ("L22H6_decode", TARGET, "decode"),
            ("L22H8_control", CONTROL, "all"),
        ]
        records = []
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            pids, _, _ = self.positions(row.prompt, row.language)
            if len(pids) > 384:
                raise ValueError("Prompt exceeds the 384-token limit")
            first_tokens = {}
            for name, head, when in conditions:
                reply, ids, counts = self.generate_phase(pids, head, when)
                first_tokens[name] = ids[0] if ids else None
                records.append({
                    "language": row.language,
                    "source": row.source,
                    "prompt": row.prompt,
                    "condition": name,
                    "reply": reply,
                    "generated_tokens": len(ids),
                    "first_token_id": first_tokens[name],
                    **self.score(reply, row.language),
                    **counts,
                })
            if first_tokens["L22H6_decode"] != first_tokens["clean"]:
                raise RuntimeError("Decode-only ablation changed the first token")
            if i % 8 == 0:
                pd.DataFrame(records).to_csv(
                    self.a.out / "reproduced_phase.csv", index=False
                )
                print("Phase:", i, "/", len(manifest), flush=True)
        pd.DataFrame(records).to_csv(
            self.a.out / "reproduced_phase.csv", index=False
        )
        return records

    @contextmanager
    def intervention(self, mode: str, key_positions: dict):
        """Intervene during cached single-token decoding, not prompt prefill."""
        state = {"prefill_calls": 0, "decode_calls": 0, "edited_calls": 0}
        if mode == "clean":
            yield state
            return

        if mode == "head_zero":
            def zero_hook(module, args):
                x = args[0]
                if x.shape[1] != 1:
                    state["prefill_calls"] += 1
                    return None
                state["decode_calls"] += 1
                y = x.clone()
                y[..., self.span] = 0
                state["edited_calls"] += 1
                return (y,) + args[1:]

            handle = self.proj.register_forward_pre_hook(zero_hook)
        else:
            head = (
                TARGET if mode in ("target_language", "target_nearby") else CONTROL
            )
            keys = key_positions[
                "nearby" if mode == "target_nearby" else "language"
            ]

            def mask_hook(module, args, kwargs):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None:
                    raise RuntimeError("Missing hidden states in attention hook")
                if hidden.shape[1] != 1:
                    state["prefill_calls"] += 1
                    return None
                state["decode_calls"] += 1
                mask = kwargs.get("attention_mask")
                if mask is None or mask.ndim != 4 or mask.shape[1] not in (
                    1, self.head_count
                ):
                    raise RuntimeError(
                        "Requires a 4D additive attention mask and "
                        "Transformers 4.57.6 eager attention"
                    )
                if max(keys) >= mask.shape[-1]:
                    raise RuntimeError("Key position not cached")
                modified = mask.expand(
                    mask.shape[0], self.head_count, mask.shape[2], mask.shape[3]
                ).clone()
                modified[:, head, :, keys] = torch.finfo(mask.dtype).min
                kwargs["attention_mask"] = modified
                state["edited_calls"] += 1
                return args, kwargs

            handle = self.attn.register_forward_pre_hook(
                mask_hook, with_kwargs=True
            )
        try:
            yield state
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(
        self, pids: list[int], mode: str, keys: dict, max_new: int | None = None
    ):
        x = torch.tensor([pids], dtype=torch.long, device="cuda")
        mask = torch.ones_like(x)
        with self.intervention(mode, keys) as counts:
            seq = self.model.generate(
                input_ids=x,
                attention_mask=mask,
                max_new_tokens=max_new or self.a.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        if mode != "clean" and (
            counts["prefill_calls"] != 1 or counts["edited_calls"] < 1
        ):
            raise RuntimeError("Unexpected hook counts: " + repr(counts))
        ids = seq[0, len(pids):].tolist()
        return self.tok.decode(ids, skip_special_tokens=True), ids, counts

    @torch.inference_mode()
    def attention(self, pids, rids, langpos, nearpos):
        """Measure head attention to designated prompt spans on clean tokens."""
        n = len(pids)
        ids = torch.tensor([pids + rids[:32]], dtype=torch.long, device="cuda")
        outputs = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_attentions=True,
        )
        a = outputs.attentions[LAYER][0].float()
        if not bool(torch.isfinite(a).all()):
            raise RuntimeError("Nonfinite attention")

        def mass(head, queries, keys):
            return float(a[head, queries][:, keys].sum(-1).mean().item())

        nreply = min(32, len(rids))
        groups = {
            "last_prompt": [n - 1],
            "early_reply": list(range(n, min(n + 8, n + nreply - 1))),
            "later_reply": list(range(n + 8, n + nreply - 1)),
        }
        res = {}
        for group, queries in groups.items():
            if queries:
                res[group + "_target_lang"] = mass(TARGET, queries, langpos)
                res[group + "_target_nearby"] = mass(TARGET, queries, nearpos)
                res[group + "_control_lang_mean"] = float(
                    np.mean([mass(h, queries, langpos) for h in (4, 8, 9)])
                )
        return res

    def edge(self, manifest):
        records = []
        attention = []
        modes = [
            "clean", "target_language", "target_nearby",
            "control_language", "head_zero",
        ]
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            pids, language, nearby = self.positions(row.prompt, row.language)
            if len(pids) > 384:
                raise ValueError("Prompt exceeds the 384-token limit")
            token_first = []
            for mode in modes:
                reply, ids, counts = self.generate(
                    pids, mode, {"language": language, "nearby": nearby}
                )
                score = self.score(reply, row.language)
                token_first.append(ids[0] if ids else None)
                records.append({
                    "language": row.language,
                    "source": row.source,
                    "prompt": row.prompt,
                    "condition": mode,
                    "reply": reply,
                    "first_token_id": token_first[-1],
                    "generated_tokens": len(ids),
                    **score,
                    **counts,
                })
                if mode == "clean":
                    # Reuse generated token IDs; decoding and re-tokenizing can
                    # change token boundaries and the corresponding query indices.
                    if len(ids) >= 4:
                        attention.append({
                            "language": row.language,
                            "source": row.source,
                            "prompt": row.prompt,
                            "n_language_tokens": len(language),
                            "n_reply_tokens": min(32, len(ids)),
                            **self.attention(pids, ids, language, nearby),
                        })
            if len(set(token_first)) != 1:
                raise RuntimeError(
                    "First generated token changed under decode-only intervention"
                )
            if i % 8 == 0:
                pd.DataFrame(records).to_csv(
                    self.a.out / "reproduced_edge.csv", index=False
                )
                print("Edge and attention:", i, "/", len(manifest), flush=True)
        pd.DataFrame(records).to_csv(
            self.a.out / "reproduced_edge.csv", index=False
        )
        pd.DataFrame(attention).to_csv(
            self.a.out / "reproduced_attention.csv", index=False
        )
        return records

    def attention_only(self, manifest):
        """Generate clean continuations and measure attention without edits."""
        rows = []
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            pids, language, nearby = self.positions(row.prompt, row.language)
            reply, ids, _ = self.generate(
                pids, "clean", {"language": language, "nearby": nearby}
            )
            if len(ids) < 4:
                continue
            rows.append({
                "language": row.language,
                "source": row.source,
                "prompt": row.prompt,
                "n_language_tokens": len(language),
                "n_reply_tokens": min(32, len(ids)),
                **self.attention(pids, ids, language, nearby),
            })
            if i % 12 == 0:
                print("Attention only:", i, "/", len(manifest), flush=True)
        pd.DataFrame(rows).to_csv(
            self.a.out / "reproduced_attention.csv", index=False
        )

    def semantic(self):
        """Test requested-language versus incidental-language token masking."""
        pairs = [("it", "es"), ("fr", "de")]
        data = []
        for left, right in pairs:
            for lang, distraction in [(left, right), (right, left)]:
                for i, topic in enumerate(TOPICS):
                    instruction = "Reply entirely in " + LANGUAGES[lang] + "."
                    incidental = (
                        'A catalog includes an unrelated label, "'
                        + LANGUAGES[distraction]
                        + ' handbook". That label does not specify your response language.'
                    )
                    prompt = (
                        instruction + " " + incidental + " " + topic
                        if i % 2 == 0
                        else incidental + " " + topic + " " + instruction
                    )
                    data.append((
                        lang,
                        distraction,
                        i,
                        "requested_first" if i % 2 == 0 else "requested_last",
                        prompt,
                    ))
        if self.a.smoke:
            data = [item for item in data if item[2] < 2]

        rows = []
        for ix, (lang, distraction, topic_index, placement, prompt) in enumerate(
            data, 1
        ):
            pids, requested, incidental = self.positions(prompt, lang, distraction)
            if len(pids) > 384:
                raise ValueError("Semantic prompt too long")
            ids = torch.tensor([pids], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                att = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    output_attentions=True,
                    use_cache=False,
                ).attentions[LAYER][0, TARGET, -1].float()
            asked = float(att[requested].sum().item())
            unrelated = float(att[incidental].sum().item())
            first = []
            conditions = [
                ("clean", "clean"),
                ("mask_requested_word", "target_language"),
                ("mask_incidental_word", "target_nearby"),
                ("control_head_requested", "control_language"),
                ("zero_head_decode", "head_zero"),
            ]
            for label, mode in conditions:
                reply, token_ids, _ = self.generate(
                    pids, mode, {"language": requested, "nearby": incidental}
                )
                if not token_ids:
                    raise RuntimeError("Empty generation")
                first.append(token_ids[0])
                rows.append({
                    "language": lang,
                    "incidental_language": distraction,
                    "topic_index": topic_index,
                    "placement": placement,
                    "prompt": prompt,
                    "condition": label,
                    "reply": reply,
                    "requested_attention_mass": asked,
                    "incidental_attention_mass": unrelated,
                    "first_token_id": token_ids[0],
                    "generated_tokens": len(token_ids),
                    **self.score(reply, lang),
                })
            if len(set(first)) != 1:
                raise RuntimeError("Semantic first-token mismatch")
            if ix % 8 == 0:
                pd.DataFrame(rows).to_csv(
                    self.a.out / "reproduced_semantic.csv", index=False
                )
                print("Semantic:", ix, "/", len(data), flush=True)
        pd.DataFrame(rows).to_csv(
            self.a.out / "reproduced_semantic.csv", index=False
        )

    @torch.inference_mode()
    def logits(self, manifest):
        """Record first-divergence margins; not independent causal attribution."""
        records = []
        for lang, count in [("it", 8), ("fr", 4), ("de", 4), ("es", 4)]:
            rows = manifest[manifest.language == lang].sample(
                n=min(count, len(manifest[manifest.language == lang])),
                random_state=20261008,
            )
            for row in rows.itertuples(index=False):
                pids, keys, control = self.positions(row.prompt, row.language)
                mapping = {"language": keys, "nearby": control}
                _, clean_ids, _ = self.generate(pids, "clean", mapping, max_new=40)
                _, changed_ids, _ = self.generate(
                    pids, "head_zero", mapping, max_new=40
                )
                j = next(
                    (i for i, (a, b) in enumerate(zip(clean_ids, changed_ids))
                     if a != b),
                    None,
                )
                if j is None or j == 0:
                    records.append({
                        "language": lang,
                        "prompt": row.prompt,
                        "status": "no_divergence" if j is None else "first_token_divergence",
                    })
                    continue

                ids = torch.tensor([pids + clean_ids[:j]], device="cuda")
                mask = torch.ones_like(ids)
                original = self.model(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).logits[0, -1].float()

                # This shared-prefix replay edits continuation positions only.
                def hook(module, args):
                    x = args[0].clone()
                    x[:, len(pids):, self.span] = 0
                    return (x,) + args[1:]

                handle = self.proj.register_forward_pre_hook(hook)
                try:
                    perturbed = self.model(
                        input_ids=ids, attention_mask=mask, use_cache=False
                    ).logits[0, -1].float()
                finally:
                    handle.remove()

                a, b = clean_ids[j], changed_ids[j]
                if not (
                    bool(torch.isfinite(original).all())
                    and bool(torch.isfinite(perturbed).all())
                ):
                    raise RuntimeError("Nonfinite logits")
                before = float((original[a] - original[b]).item())
                after = float((perturbed[a] - perturbed[b]).item())
                records.append({
                    "language": lang,
                    "prompt": row.prompt,
                    "status": "compared",
                    "first_divergence_index": j,
                    "clean_next_token": self.tok.decode([a]),
                    "decode_next_token": self.tok.decode([b]),
                    "clean_model_predicts_clean_token": int(
                        original.argmax().item() == a
                    ),
                    "ablated_model_predicts_decode_token": int(
                        perturbed.argmax().item() == b
                    ),
                    "clean_minus_decode_logit_margin": before,
                    "margin_after_decode_ablation": after,
                    "causal_margin_change": after - before,
                })
        pd.DataFrame(records).to_csv(
            self.a.out / "reproduced_logit_diagnostic.csv", index=False
        )


def main():
    a = get_args()
    manifest = pd.read_csv(a.manifest)
    required = {"language", "source", "prompt"}
    if not required.issubset(manifest.columns):
        raise ValueError("Invalid manifest; expected: " + repr(required))
    if manifest.prompt.duplicated().any():
        raise ValueError("Manifest has duplicate prompts")
    if a.smoke:
        manifest = manifest.groupby("language", sort=False).head(2).reset_index(drop=True)

    exp = Experiment(a)
    if a.mode in ("phase", "all"):
        exp.phase(manifest)
    if a.mode in ("edge", "all"):
        exp.edge(manifest)
    if a.mode == "attention":
        exp.attention_only(manifest)
    if a.mode in ("semantic", "all"):
        exp.semantic()
    if a.mode in ("logit", "all"):
        exp.logits(manifest)

    resolved_revision = (
        getattr(exp.model.config, "_commit_hash", None)
        or exp.tok.init_kwargs.get("_commit_hash")
    )
    metadata = {
        "model_id": a.model,
        "model_revision": a.revision,
        "resolved_model_revision": resolved_revision,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "float32",
        "attention_implementation": "eager",
        "logit_subsample_random_state": 20261008,
        "note": "The original run did not record the model commit; updated weights may change the output.",
    }
    (a.out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Completed:", a.mode, "Output:", a.out)


if __name__ == "__main__":
    main()
