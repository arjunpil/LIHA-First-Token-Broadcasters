#!/usr/bin/env python3
"""Validate and summarize committed L22H6 response-level measurements.

This script uses the saved CSVs and does not perform model inference.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "qwen-l22h6-mechanism"
)


def paired(df, first, second, index):
    """Return an exact paired comparison on scorable clean-correct prompts."""
    passed = df.pivot(index=index, columns="condition", values="passed").astype(bool)
    skipped = df.pivot(index=index, columns="condition", values="skipped").astype(bool)
    valid = passed.loc[~skipped.any(axis=1)]
    valid = valid.loc[valid.clean]

    a = valid[first]
    b = valid[second]
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    discordant = only_a + only_b
    p_value = binomtest(only_a, discordant).pvalue if discordant else 1.0
    return len(valid), int(a.sum()), int(b.sum()), only_a, only_b, p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    phase = pd.read_csv(args.data / "liha_l22h6_phase_validation.csv")
    edge = pd.read_csv(args.data / "liha_l22h6_language_edge_validation.csv")
    semantic = pd.read_csv(args.data / "liha_l22h6_semantic_contrast.csv")
    attention = pd.read_csv(args.data / "liha_l22h6_attention_analysis.csv")
    logit = pd.read_csv(args.data / "liha_l22h6_logit_divergence.csv")

    assert len(phase) == 480 and phase.prompt.nunique() == 96
    assert len(edge) == 480 and edge.prompt.nunique() == 96
    assert len(semantic) == 160 and semantic.prompt.nunique() == 32
    assert len(attention) == 96 and len(logit) == 20

    for frame in (phase, edge, semantic):
        assert not frame.duplicated(["prompt", "condition"]).any()
        assert frame.skipped.sum() == 0
    assert attention.select_dtypes("number").notna().all().all()

    print("DATA CHECK: PASS (phase 480, edge 480, semantic 160, attention 96, logit 20)")
    print("\nPhase intervention, Italian (24 prompts):")
    print(
        phase.query("language == 'it'")
        .groupby("condition").passed.agg(["sum", "count"])
        .to_string()
    )

    print("\nTargeted edge mask, clean-correct paired comparison:")
    edge_comp = paired(
        edge, "target_nearby", "target_language", ["language", "source", "prompt"]
    )
    print("n, nearby_pass, language_pass, nearby_only, language_only, p:", edge_comp)
    assert edge_comp[:5] == (95, 95, 88, 7, 0)

    print("\nSemantic contrast, clean-correct paired comparison:")
    sem_comp = paired(
        semantic, "mask_incidental_word", "mask_requested_word",
        ["language", "prompt"],
    )
    print("n, incidental_pass, requested_pass, incidental_only, requested_only, p:", sem_comp)
    assert sem_comp[:5] == (31, 31, 27, 4, 0)

    print("\nMean attention to requested-language tokens:")
    columns = [
        "last_prompt_target_lang",
        "early_reply_target_lang",
        "later_reply_target_lang",
    ]
    print(attention.groupby("language")[columns].mean().round(3).to_string())

    print("\nFirst-divergence logit diagnostics (not independent attribution):")
    print(logit.groupby(["language", "status"]).size().to_string())
    print(
        "\nInterpretation: exploratory; the 96 prompts are not an independent "
        "benchmark, and the original model revision is unrecorded."
    )


if __name__ == "__main__":
    main()
