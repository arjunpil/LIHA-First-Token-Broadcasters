"""Run the final matched-null analysis for GPT-2 attention redistribution."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redistribution-dir",
        default="rerun/out/gpt2_redistribution",
        help="Directory produced by redistribution_validation.py.",
    )
    parser.add_argument(
        "--zero-summary",
        default="rerun/results/gpt2/summary.json",
    )
    parser.add_argument(
        "--mean-summary",
        default="rerun/results/gpt2-mean-ablation/summary.json",
    )
    parser.add_argument("--target", default="L6H10")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for matched sign-flip tests.",
    )
    parser.add_argument(
        "--correlation-seed",
        type=int,
        default=1000,
        help="Seed for Spearman permutation tests.",
    )
    parser.add_argument(
        "--out",
        default="rerun/results/gpt2-redistribution/matched_null_statistics.json",
    )
    return parser.parse_args()


def matched_signflip_test(
    delta_np,
    *,
    k,
    n_perm,
    seed,
    batch=1000,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    delta = torch.tensor(
        delta_np,
        dtype=torch.float32,
        device=device,
    )

    n_prompts = delta.shape[0]
    observed_head_means = delta.mean(dim=0)
    observed = (
        torch.topk(
            observed_head_means,
            k=k,
        )
        .values
        .mean()
        .item()
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    exceedances = 0
    done = 0
    null_values = np.empty(
        n_perm,
        dtype=np.float32,
    )

    while done < n_perm:
        current = min(batch, n_perm - done)

        signs = torch.randint(
            0,
            2,
            (current, n_prompts),
            generator=generator,
            device=device,
            dtype=torch.int8,
        )
        signs = signs.float().mul_(2.0).sub_(1.0)

        perm_head_means = (
            signs @ delta
        ) / n_prompts

        perm_stats = (
            torch.topk(
                perm_head_means,
                k=k,
                dim=1,
            )
            .values
            .mean(dim=1)
        )

        values = (
            perm_stats
            .detach()
            .cpu()
            .numpy()
        )
        null_values[done : done + current] = values

        exceedances += int(
            (perm_stats >= observed)
            .sum()
            .item()
        )
        done += current

    p_value = (
        exceedances + 1
    ) / (
        n_perm + 1
    )

    return {
        "observed": float(observed),
        "p": float(p_value),
        "null_mean": float(null_values.mean()),
        "null_p95": float(np.percentile(null_values, 95)),
        "null_p99": float(np.percentile(null_values, 99)),
        "null_p999": float(np.percentile(null_values, 99.9)),
        "exceedances": int(exceedances),
    }


def spearman_permutation(
    x,
    y,
    *,
    n_perm,
    seed,
    batch=10000,
):
    rank_x = rankdata(x).astype(np.float64)
    rank_y = rankdata(y).astype(np.float64)

    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()

    denominator = np.sqrt(
        np.sum(rank_x ** 2)
        * np.sum(rank_y ** 2)
    )
    observed = float(
        np.sum(rank_x * rank_y)
        / denominator
    )

    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0

    while done < n_perm:
        current = min(
            batch,
            n_perm - done,
        )

        order = np.argsort(
            rng.random(
                (current, len(rank_y))
            ),
            axis=1,
        )
        perm_y = rank_y[order]
        statistics = (
            perm_y @ rank_x
        ) / denominator

        extreme += int(
            np.sum(
                np.abs(statistics)
                >= abs(observed) - 1e-12
            )
        )
        done += current

    p_value = (
        extreme + 1
    ) / (
        n_perm + 1
    )

    return float(observed), float(p_value)


def main():
    args = parse_args()

    redistribution_dir = Path(args.redistribution_dir)
    summary = json.loads(
        (
            redistribution_dir / "summary.json"
        ).read_text(encoding="utf-8")
    )
    arrays = np.load(
        redistribution_dir / "attention_deltas.npz",
        allow_pickle=True,
    )

    target = args.target
    if target not in summary["results"]:
        raise KeyError(f"Target not present in summary: {target}")

    primary_tests = {}

    for index, metric in enumerate(("last", "all")):
        delta = arrays[f"{target}_delta_{metric}"]

        result = matched_signflip_test(
            delta,
            k=args.top_k,
            n_perm=args.n_perm,
            seed=args.seed + index,
        )
        result["p_bonferroni_2"] = min(
            1.0,
            2.0 * result["p"],
        )
        primary_tests[metric] = result

    zero_summary = json.loads(
        Path(args.zero_summary).read_text(encoding="utf-8")
    )
    mean_summary = json.loads(
        Path(args.mean_summary).read_text(encoding="utf-8")
    )

    layer = int(target[1:].split("H")[0])
    n_heads = 12
    heads = [
        f"L{layer}H{head}"
        for head in range(n_heads)
    ]

    zero_c2w = np.array([
        zero_summary[
            "modes"
        ][
            "head"
        ][
            "table"
        ][head][
            "full"
        ][
            "c2w"
        ]
        for head in heads
    ])
    mean_c2w = np.array([
        mean_summary[
            "table"
        ][head][
            "c2w"
        ]
        for head in heads
    ])
    last_redistribution = np.array([
        summary["results"][head][
            "last"
        ][
            "observed_topk_mean"
        ]
        for head in heads
    ])
    all_redistribution = np.array([
        summary["results"][head][
            "all"
        ][
            "observed_topk_mean"
        ]
        for head in heads
    ])

    relationships = [
        (
            "zero c2w vs LAST redistribution",
            zero_c2w,
            last_redistribution,
        ),
        (
            "mean c2w vs LAST redistribution",
            mean_c2w,
            last_redistribution,
        ),
        (
            "zero c2w vs ALL redistribution",
            zero_c2w,
            all_redistribution,
        ),
        (
            "mean c2w vs ALL redistribution",
            mean_c2w,
            all_redistribution,
        ),
    ]

    correlation_results = {}

    for index, (name, x, y) in enumerate(
        relationships
    ):
        rho, p_value = spearman_permutation(
            x,
            y,
            n_perm=args.n_perm,
            seed=args.correlation_seed + index,
        )
        correlation_results[name] = {
            "rho": rho,
            "permutation_p": p_value,
        }

    layer_matched = {}

    for metric in ("last", "all"):
        values = {
            head: result[metric][
                "observed_topk_mean"
            ]
            for head, result
            in summary["results"].items()
        }
        target_value = values[target]
        control_values = np.array([
            value
            for head, value in values.items()
            if head != target
        ])
        rank = int(
            1 + np.sum(control_values > target_value)
        )
        empirical_p = float(
            (
                1
                + np.sum(
                    control_values >= target_value
                )
            )
            / (
                len(control_values) + 1
            )
        )

        layer_matched[metric] = {
            "rank": rank,
            "n_heads": len(values),
            "empirical_p": empirical_p,
            "target_value": float(target_value),
        }

    output = {
        "primary_target": target,
        "n_prompts": int(summary["n_prompts"]),
        "top_k": args.top_k,
        "n_permutations": args.n_perm,
        "matched_null_base_seed": args.seed,
        "matched_null_metric_seeds": {
            "last": args.seed,
            "all": args.seed + 1,
        },
        "correlation_base_seed": args.correlation_seed,
        "correlation_seeds": {
            name: args.correlation_seed + index
            for index, (name, _, _) in enumerate(relationships)
        },
        "primary_matched_null_tests": primary_tests,
        "layer_matched_comparison": layer_matched,
        "causal_effect_correlations": correlation_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("target:", target)
    print("prompts:", summary["n_prompts"])

    for metric in ("last", "all"):
        result = primary_tests[metric]
        comparison = layer_matched[metric]

        print(f"\n{metric}:")
        print(
            "  observed top-k mean:",
            f"{result['observed']:+.8f}",
        )
        print(
            "  matched p:",
            f"{result['p']:.8g}",
        )
        print(
            "  Bonferroni p:",
            f"{result['p_bonferroni_2']:.8g}",
        )
        print(
            "  layer-matched rank:",
            f"{comparison['rank']}/{comparison['n_heads']}",
        )
        print(
            "  layer-matched empirical p:",
            f"{comparison['empirical_p']:.6g}",
        )

    print("\nCausal-effect correlations:")
    for name, result in correlation_results.items():
        print(
            f"  {name}: "
            f"rho={result['rho']:+.4f}, "
            f"p={result['permutation_p']:.6g}"
        )

    print("\nsaved:", out_path)


if __name__ == "__main__":
    main()
