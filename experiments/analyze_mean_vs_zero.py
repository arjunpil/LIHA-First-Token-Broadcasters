"""Analyze GPT-2 mean ablation against the corrected zero-ablation reference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mean-dir",
        default="out/gpt2_mean_zero",
        help="Directory containing gens.jsonl and prompts.csv from mean_vs_zero.py.",
    )
    parser.add_argument(
        "--zero-summary",
        default="results/gpt2/summary.json",
        help="Corrected zero-ablation summary produced by experiments/analyze.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/gpt2-mean-ablation",
    )
    return parser.parse_args()


def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def head_key(head):
    layer, index = head[1:].split("H")
    return int(layer), int(index)


def main():
    args = parse_args()

    mean_dir = Path(args.mean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zero_summary = json.loads(
        Path(args.zero_summary).read_text(encoding="utf-8")
    )
    zero_table = zero_summary["modes"]["head"]["table"]

    runs = [
        json.loads(line)
        for line in (
            mean_dir / "gens.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_condition = {
        run["cond"]: run
        for run in runs
    }

    mean_conditions = sorted(
        condition
        for condition in by_condition
        if condition.startswith("mean:")
    )

    if "base" not in by_condition:
        raise RuntimeError("Missing baseline condition")
    if len(mean_conditions) != len(zero_table):
        raise RuntimeError(
            "Mean and zero runs contain different numbers of heads: "
            f"{len(mean_conditions)} vs {len(zero_table)}"
        )

    with open(
        mean_dir / "prompts.csv",
        encoding="utf-8",
    ) as handle:
        prompt_rows = list(csv.DictReader(handle))

    expected = [
        row["language"]
        for row in prompt_rows
    ]

    DetectorFactory.seed = 0

    unique_texts = {
        text
        for run in runs
        for text in run["texts"]
    }
    label_cache = {
        text: safe_detect(text)
        for text in unique_texts
    }

    def labels_for(condition):
        return [
            label_cache[text]
            for text in by_condition[condition]["texts"]
        ]

    base_labels = labels_for("base")
    baseline_acc = float(
        np.mean([
            pred == gold
            for pred, gold in zip(base_labels, expected)
        ])
    )

    zero_baseline = float(zero_summary["baseline_acc_full"])
    if not np.isclose(
        baseline_acc,
        zero_baseline,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Baseline mismatch between mean and zero runs: "
            f"{baseline_acc} vs {zero_baseline}"
        )

    base_nll = float(
        np.mean(
            list(by_condition["base"]["nll"].values())
        )
    )

    def mean_metrics(head):
        condition = f"mean:{head}"
        labels = labels_for(condition)

        sr = np.mean([
            ablated != base
            for ablated, base in zip(labels, base_labels)
        ])
        c2w = np.mean([
            base == gold and ablated != gold
            for base, ablated, gold
            in zip(base_labels, labels, expected)
        ])
        w2c = np.mean([
            base != gold and ablated == gold
            for base, ablated, gold
            in zip(base_labels, labels, expected)
        ])
        acc = np.mean([
            ablated == gold
            for ablated, gold in zip(labels, expected)
        ])
        mean_nll = np.mean(
            list(by_condition[condition]["nll"].values())
        )

        return {
            "sr": float(sr),
            "c2w": float(c2w),
            "w2c": float(w2c),
            "acc": float(acc),
            "dnll": float(mean_nll - base_nll),
        }

    heads = sorted(zero_table, key=head_key)
    records = []
    mean_table = {}

    for head in heads:
        mean_result = mean_metrics(head)
        zero_result = zero_table[head]
        mean_table[head] = mean_result

        records.append({
            "head": head,
            "mean_sr": mean_result["sr"],
            "zero_sr": zero_result["full"]["sr"],
            "mean_c2w": mean_result["c2w"],
            "zero_c2w": zero_result["full"]["c2w"],
            "mean_w2c": mean_result["w2c"],
            "zero_w2c": zero_result["full"]["w2c"],
            "mean_acc": mean_result["acc"],
            "zero_acc": zero_result["full"]["acc"],
            "mean_dnll": mean_result["dnll"],
            "zero_dnll": zero_result["dnll"],
            "zero_sr_ci_lo": zero_result["full"]["sr_ci"][0],
            "zero_sr_ci_hi": zero_result["full"]["sr_ci"][1],
        })

    frame = pd.DataFrame(records)

    def correlation(mean_col, zero_col):
        rho, p_value = spearmanr(
            frame[mean_col],
            frame[zero_col],
        )
        return float(rho), float(p_value)

    sr_rho, sr_p = correlation("mean_sr", "zero_sr")
    c2w_rho, c2w_p = correlation("mean_c2w", "zero_c2w")
    w2c_rho, w2c_p = correlation("mean_w2c", "zero_w2c")
    dnll_rho, dnll_p = correlation("mean_dnll", "zero_dnll")

    summary = {
        "baseline_acc": baseline_acc,
        "language_detection_seed": 0,
        "n_prompts": len(prompt_rows),
        "n_heads": len(heads),
        "table": mean_table,
        "agreement_with_zero": {
            "spearman_sr": sr_rho,
            "spearman_sr_p": sr_p,
            "spearman_c2w": c2w_rho,
            "spearman_c2w_p": c2w_p,
            "spearman_w2c": w2c_rho,
            "spearman_w2c_p": w2c_p,
            "spearman_dnll": dnll_rho,
            "spearman_dnll_p": dnll_p,
        },
    }

    frame.to_csv(
        out_dir / "mean_vs_zero_comparison.csv",
        index=False,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("baseline accuracy:", baseline_acc)
    print("heads:", len(heads))
    print(
        "Spearman SR:",
        f"rho={sr_rho:.3f}",
        f"p={sr_p:.3g}",
    )
    print(
        "Spearman c2w:",
        f"rho={c2w_rho:.3f}",
        f"p={c2w_p:.3g}",
    )
    print(
        "Spearman w2c:",
        f"rho={w2c_rho:.3f}",
        f"p={w2c_p:.3g}",
    )
    print(
        "Spearman dNLL:",
        f"rho={dnll_rho:.3f}",
        f"p={dnll_p:.3g}",
    )
    print("saved:", out_dir / "summary.json")
    print("saved:", out_dir / "mean_vs_zero_comparison.csv")


if __name__ == "__main__":
    main()
