"""Validate downstream first-token attention redistribution after GPT-2 head ablation.

The intervention zeros one attention head at the input to GPT-2's c_proj output
projection. For a target layer, the script measures changes in first-token
attention only in later layers, which are the layers that can causally respond
to the ablation in a single forward pass.

Two attention summaries are recorded for every downstream head:
- ``last``: attention from the final prompt position to the first prompt token.
- ``all``: attention to the first prompt token averaged over prompt positions.

The script runs every head in the target layer so that the selected target can
be compared with layer-matched controls. Prompt-level attention deltas are
stored in a compressed NPZ file for subsequent matched-null analysis.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


N_LAYERS = 12
N_HEADS = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure downstream first-token attention redistribution after "
            "true GPT-2 head ablation."
        )
    )
    parser.add_argument(
        "--prompts",
        default="prompts/prompts_european.csv",
        help="CSV with prompt and language columns.",
    )
    parser.add_argument(
        "--per-lang",
        type=int,
        default=500,
        help="Prompts per language. Use <=0 for all rows.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Layer whose heads are ablated.",
    )
    parser.add_argument(
        "--target-head",
        type=int,
        default=10,
        help="Head used as the target within --layer.",
    )
    parser.add_argument("--bs", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help=(
            "Number of paired sign-flip permutations computed during collection. "
            "Use analyze_redistribution.py for the final high-precision test."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default="out/gpt2_redistribution",
    )
    return parser.parse_args()


def load_rows(path, per_lang):
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if per_lang <= 0:
        return rows

    counts = {}
    selected = []

    for row in rows:
        lang = row["language"]
        count = counts.get(lang, 0)

        if count < per_lang:
            selected.append(row)
            counts[lang] = count + 1

    return selected


def ablation_hook(model, layer, head, head_dim):
    def hook(module, args):
        hidden = args[0].clone()
        start = head * head_dim
        end = start + head_dim
        hidden[..., start:end] = 0.0
        return (hidden,) + tuple(args[1:])

    return model.transformer.h[
        layer
    ].attn.c_proj.register_forward_pre_hook(hook)


@torch.inference_mode()
def collect_attention(
    model,
    tokenizer,
    rows,
    device,
    target_layer,
    batch_size,
    head_dim,
    ablate_head=None,
):
    candidate_ids = [
        f"L{layer}H{head}"
        for layer in range(target_layer + 1, N_LAYERS)
        for head in range(N_HEADS)
    ]

    n_candidates = len(candidate_ids)
    final_prompt = np.zeros(
        (len(rows), n_candidates),
        dtype=np.float32,
    )
    all_prompt = np.zeros(
        (len(rows), n_candidates),
        dtype=np.float32,
    )

    prompts = [row["prompt"] for row in rows]
    handle = None

    if ablate_head is not None:
        handle = ablation_hook(
            model,
            target_layer,
            ablate_head,
            head_dim,
        )

    try:
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))

            batch = tokenizer(
                prompts[start:end],
                return_tensors="pt",
                padding=True,
            )
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            output = model(
                **batch,
                output_attentions=True,
                use_cache=False,
            )
            attentions = output.attentions
            lengths = (
                batch["attention_mask"]
                .sum(dim=1)
                .tolist()
            )

            for i, length in enumerate(lengths):
                col = 0

                for layer in range(
                    target_layer + 1,
                    N_LAYERS,
                ):
                    attn = attentions[layer][i]

                    final_prompt[
                        start + i,
                        col : col + N_HEADS,
                    ] = (
                        attn[:, length - 1, 0]
                        .float()
                        .cpu()
                        .numpy()
                    )

                    if length > 1:
                        all_value = attn[:, 1:length, 0].mean(dim=-1)
                    else:
                        all_value = attn[:, 0, 0]

                    all_prompt[
                        start + i,
                        col : col + N_HEADS,
                    ] = (
                        all_value
                        .float()
                        .cpu()
                        .numpy()
                    )

                    col += N_HEADS

            print(
                f"  {end}/{len(rows)} prompts",
                flush=True,
            )

    finally:
        if handle is not None:
            handle.remove()

    return candidate_ids, final_prompt, all_prompt


def topk_stat(delta, k):
    means = delta.mean(axis=0)
    indices = np.argsort(means)[-k:][::-1]

    return (
        float(means[indices].mean()),
        indices,
        means,
    )


def paired_signflip_test(
    delta,
    k,
    n_perm,
    seed,
    perm_batch=1000,
):
    observed, _, _ = topk_stat(delta, k)
    rng = np.random.default_rng(seed)
    n_prompts = delta.shape[0]
    null_parts = []

    remaining = n_perm

    while remaining:
        batch_size = min(perm_batch, remaining)

        signs = rng.integers(
            0,
            2,
            size=(batch_size, n_prompts),
            dtype=np.int8,
        )
        signs = signs.astype(np.float32) * 2.0 - 1.0

        perm_means = (signs @ delta) / n_prompts
        top_values = np.partition(
            perm_means,
            -k,
            axis=1,
        )[:, -k:]

        null_parts.append(top_values.mean(axis=1))
        remaining -= batch_size

    null = np.concatenate(null_parts)

    p_value = (
        1 + np.sum(null >= observed)
    ) / (
        len(null) + 1
    )

    return {
        "observed_topk_mean": observed,
        "paired_signflip_p": float(p_value),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "null_p99": float(np.percentile(null, 99)),
    }


def analyze_metric(
    delta,
    candidate_ids,
    k,
    n_perm,
    seed,
):
    result = paired_signflip_test(
        delta,
        k,
        n_perm,
        seed,
    )

    _, indices, means = topk_stat(delta, k)
    result["top_heads"] = [
        {
            "head": candidate_ids[index],
            "mean_delta": float(means[index]),
        }
        for index in indices
    ]
    return result


def main():
    args = parse_args()

    if not 0 <= args.layer < N_LAYERS:
        raise ValueError(f"Layer out of range: {args.layer}")
    if not 0 <= args.target_head < N_HEADS:
        raise ValueError(f"Head out of range: {args.target_head}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        args.prompts,
        args.per_lang,
    )

    print("=== REDISTRIBUTION VALIDATION ===")
    print("prompts:", len(rows))
    print("target layer:", args.layer)
    print(
        "target head:",
        f"L{args.layer}H{args.target_head}",
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        attn_implementation="eager",
    ).to(device).eval()

    head_dim = (
        model.config.hidden_size
        // model.config.num_attention_heads
    )

    if model.config.num_hidden_layers != N_LAYERS:
        raise ValueError(
            "This script is configured for GPT-2 small "
            f"({N_LAYERS} layers)."
        )
    if model.config.num_attention_heads != N_HEADS:
        raise ValueError(
            "This script is configured for GPT-2 small "
            f"({N_HEADS} heads per layer)."
        )

    print("\nRunning clean baseline...")

    candidate_ids, base_last, base_all = collect_attention(
        model,
        tokenizer,
        rows,
        device,
        args.layer,
        args.bs,
        head_dim,
    )

    print(
        "downstream candidates:",
        len(candidate_ids),
    )

    all_results = {}
    arrays = {
        "base_last": base_last,
        "base_all": base_all,
    }

    for head in range(N_HEADS):
        name = f"L{args.layer}H{head}"
        print(f"\nAblating {name}...")

        ids2, ablated_last, ablated_all = collect_attention(
            model,
            tokenizer,
            rows,
            device,
            args.layer,
            args.bs,
            head_dim,
            ablate_head=head,
        )

        if ids2 != candidate_ids:
            raise RuntimeError("Candidate-head ordering changed across runs")

        delta_last = ablated_last - base_last
        delta_all = ablated_all - base_all

        arrays[f"{name}_delta_last"] = delta_last
        arrays[f"{name}_delta_all"] = delta_all

        all_results[name] = {
            "last": analyze_metric(
                delta_last,
                candidate_ids,
                args.top_k,
                args.n_perm,
                args.seed + head * 10,
            ),
            "all": analyze_metric(
                delta_all,
                candidate_ids,
                args.top_k,
                args.n_perm,
                args.seed + head * 10 + 1,
            ),
        }

    target = f"L{args.layer}H{args.target_head}"

    for metric in ("last", "all"):
        target_stat = all_results[
            target
        ][metric]["observed_topk_mean"]

        controls = {
            head: result[metric]["observed_topk_mean"]
            for head, result in all_results.items()
            if head != target
        }
        control_values = np.array(list(controls.values()))

        empirical_p = (
            1 + np.sum(control_values >= target_stat)
        ) / (
            len(control_values) + 1
        )
        rank = (
            1 + np.sum(control_values > target_stat)
        )

        all_results[target][
            metric
        ]["layer_matched_control_rank"] = int(rank)
        all_results[target][
            metric
        ]["layer_matched_control_p"] = float(empirical_p)
        all_results[target][
            metric
        ]["layer_matched_controls"] = controls

    summary = {
        "model": "gpt2",
        "n_prompts": len(rows),
        "target_layer": args.layer,
        "target_head": target,
        "candidate_layers": list(
            range(args.layer + 1, N_LAYERS)
        ),
        "n_candidates": len(candidate_ids),
        "top_k": args.top_k,
        "n_permutations": args.n_perm,
        "metrics": {
            "last": (
                "first-token attention from the final prompt position"
            ),
            "all": (
                "first-token attention averaged over prompt positions"
            ),
        },
        "null": (
            "paired prompt-level condition swap (sign flip), "
            "with top-k selection repeated on every permutation"
        ),
        "results": all_results,
    }

    with open(
        out_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    np.savez_compressed(
        out_dir / "attention_deltas.npz",
        candidate_ids=np.array(candidate_ids),
        **arrays,
    )

    print("\n=== LAYER-MATCHED RESULTS ===")

    for metric in ("last", "all"):
        print(f"\nMetric: {metric}")

        ordered = sorted(
            all_results.items(),
            key=lambda item: item[1][metric][
                "observed_topk_mean"
            ],
            reverse=True,
        )

        for name, result in ordered:
            row = result[metric]
            marker = "  <-- TARGET" if name == target else ""

            print(
                f"{name:6s} "
                f"T={row['observed_topk_mean']:+.6f} "
                f"paired_p={row['paired_signflip_p']:.6g}"
                f"{marker}"
            )

        row = all_results[target][metric]

        print(
            f"\n{target} layer-matched rank: "
            f"{row['layer_matched_control_rank']}/{N_HEADS}"
        )
        print(
            f"{target} layer-matched empirical p: "
            f"{row['layer_matched_control_p']:.4f}"
        )
        print(f"{target} top downstream heads:")

        for top_head in row["top_heads"]:
            print(
                f"  {top_head['head']:6s} "
                f"{top_head['mean_delta']:+.6f}"
            )

    print("\nSaved:")
    print(out_dir / "summary.json")
    print(out_dir / "attention_deltas.npz")
    print("\nREDISTRIBUTION VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
