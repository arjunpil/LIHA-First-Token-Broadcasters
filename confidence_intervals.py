"""
confidence_intervals.py — Bootstrap confidence intervals for all switch rates.

Reads existing ablation_sweep.csv and bloom_ablation_sweep.csv and computes
95% confidence intervals via bootstrap resampling over the 25 prompts.

This addresses the key reviewer concern: are the switch rate differences
between heads statistically meaningful or could they be noise?

Run time: ~5 minutes (no GPU needed, pure numpy)
"""

import numpy as np
import pandas as pd
import json
import os

RESULTS_DIR = "results"
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_PROMPTS = 25
LANGUAGES = ["en", "fr", "de", "es", "it"]

def bootstrap_ci(successes, n, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL):
    """
    Bootstrap confidence interval for a proportion.
    successes: number of switches observed
    n: total number of prompts
    Returns: (lower, upper, mean)
    """
    p_obs = successes / n
    # Resample with replacement
    samples = np.random.binomial(n, p_obs, size=n_bootstrap) / n
    alpha = 1 - ci
    lower = np.percentile(samples, 100 * alpha / 2)
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return lower, upper, p_obs

def add_confidence_intervals(df, n_prompts=N_PROMPTS):
    """Add CI columns to a sweep dataframe."""
    lowers, uppers = [], []
    for _, row in df.iterrows():
        switches = round(row["switch_rate"] * n_prompts)
        lower, upper, _ = bootstrap_ci(switches, n_prompts)
        lowers.append(lower)
        uppers.append(upper)
    df = df.copy()
    df["switch_rate_lower"] = lowers
    df["switch_rate_upper"] = uppers
    df["switch_rate_ci"] = [
        f"{r:.3f} [{l:.3f}, {u:.3f}]"
        for r, l, u in zip(df["switch_rate"], lowers, uppers)
    ]
    return df

def run_gpt2_cis():
    print("=== GPT-2 CONFIDENCE INTERVALS ===")
    df = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
    df = add_confidence_intervals(df)
    df.to_csv(f"{RESULTS_DIR}/ablation_sweep_ci.csv", index=False)

    top10 = df.nlargest(10, "switch_rate")
    print("\nTop 10 heads with 95% bootstrap CIs:")
    print(f"{'Head':<10} {'Switch Rate':>12} {'95% CI':>20}")
    print("-" * 45)
    for _, row in top10.iterrows():
        print(f"{row['head_id']:<10} {row['switch_rate']:>12.3f} "
              f"  [{row['switch_rate_lower']:.3f}, {row['switch_rate_upper']:.3f}]")

    # Key stat: is L6H1 CI non-overlapping with next tier?
    l6h1 = df[df["head_id"] == "L6H1"].iloc[0]
    next_tier = df[df["switch_rate"] == 0.24].iloc[0]
    print(f"\nL6H1 CI:        [{l6h1['switch_rate_lower']:.3f}, {l6h1['switch_rate_upper']:.3f}]")
    print(f"Next tier CI:   [{next_tier['switch_rate_lower']:.3f}, {next_tier['switch_rate_upper']:.3f}]")
    overlap = l6h1["switch_rate_lower"] < next_tier["switch_rate_upper"]
    print(f"CIs overlap: {overlap}")
    print("→ If False, L6H1 is statistically distinguishable from the next tier")

    return df

def run_bloom_cis():
    print("\n=== BLOOM CONFIDENCE INTERVALS ===")
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/bloom_ablation_sweep.csv")
    except FileNotFoundError:
        print("bloom_ablation_sweep.csv not found, skipping")
        return None

    df = add_confidence_intervals(df)
    df.to_csv(f"{RESULTS_DIR}/bloom_ablation_sweep_ci.csv", index=False)

    top10 = df.nlargest(10, "switch_rate")
    print("\nTop 10 BLOOM heads with 95% bootstrap CIs:")
    print(f"{'Head':<10} {'Switch Rate':>12} {'95% CI':>20}")
    print("-" * 45)
    for _, row in top10.iterrows():
        print(f"{row['head_id']:<10} {row['switch_rate']:>12.3f} "
              f"  [{row['switch_rate_lower']:.3f}, {row['switch_rate_upper']:.3f}]")

    return df

def run_baseline_ci():
    print("\n=== BASELINE ACCURACY CONFIDENCE INTERVALS ===")

    for model, fname in [("GPT-2", "baseline.csv"), ("BLOOM", "bloom_baseline.csv")]:
        try:
            df = pd.read_csv(f"{RESULTS_DIR}/{fname}")
            acc = df["correct"].mean()
            n = len(df)
            correct = df["correct"].sum()
            lower, upper, _ = bootstrap_ci(correct, n)
            print(f"{model} baseline: {acc:.3f} [{lower:.3f}, {upper:.3f}] (95% CI, n={n})")
        except FileNotFoundError:
            print(f"{model}: {fname} not found")

def run_per_language_cis():
    print("\n=== PER-LANGUAGE SWITCH RATE CIs (Top 5 GPT-2 heads) ===")
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/per_language_ablation.csv")
        top5 = ["L6H1", "L10H4", "L7H3", "L3H1", "L1H10"]

        print(f"\n{'Head':<8}", end="")
        for lang in LANGUAGES:
            print(f"  {lang.upper():>16}", end="")
        print()
        print("-" * 95)

        for head_id in top5:
            row = df[df["head_id"] == head_id]
            if row.empty:
                continue
            row = row.iloc[0]
            print(f"{head_id:<8}", end="")
            for lang in LANGUAGES:
                sr = row[f"switch_{lang}"]
                switches = round(sr * 5)  # 5 prompts per language
                lower, upper, _ = bootstrap_ci(switches, 5)
                print(f"  {sr:.2f}[{lower:.2f},{upper:.2f}]", end="")
            print()
    except FileNotFoundError:
        print("per_language_ablation.csv not found")

def print_paper_stats():
    print("\n=== KEY NUMBERS FOR PAPER (with CIs) ===")
    try:
        df = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep_ci.csv")
        l6h1 = df[df["head_id"] == "L6H1"].iloc[0]
        n_total = len(df)
        n_above_10 = (df["switch_rate"] > 0.10).sum()
        n_zero = (df["switch_rate"] == 0.0).sum()

        print(f"Total heads tested: {n_total}")
        print(f"Heads with switch rate > 0.10: {n_above_10} ({100*n_above_10/n_total:.1f}%)")
        print(f"Heads with switch rate = 0.0: {n_zero} ({100*n_zero/n_total:.1f}%)")
        print(f"L6H1 switch rate: {l6h1['switch_rate']:.3f} "
              f"[{l6h1['switch_rate_lower']:.3f}, {l6h1['switch_rate_upper']:.3f}]")

        # ── How to write about accuracy in the paper ─────────────────────────
        print("\n=== PAPER FRAMING GUIDANCE FOR ACCURACY NUMBERS ===")
        try:
            multi_df = pd.read_csv(f"{RESULTS_DIR}/multi_ablation.csv")
            has_ci   = "ci_lower" in multi_df.columns
            base_row = multi_df[multi_df["k"] == 0].iloc[0]
            k1_row   = multi_df[multi_df["k"] == 1].iloc[0]
            k10_row  = multi_df[multi_df["k"] == 10].iloc[0]

            n_prompts = None
            # Try to infer n from data/ CSVs
            for lang in ["en","fr","de","es","it"]:
                p = f"data/prompts_{lang}.csv"
                if os.path.exists(p):
                    import pandas as _pd
                    sub = _pd.read_csv(p)
                    n_prompts = (n_prompts or 0) + len(sub)

            n_str = f"{n_prompts}" if n_prompts else "N"

            if has_ci:
                print(f"\nSuggested paper sentence (Section 6):")
                print(f'  "Ablating L6H1 alone (k=1) raises accuracy from '
                      f'{base_row["accuracy"]:.0%} [{base_row["ci_lower"]:.2f}, '
                      f'{base_row["ci_upper"]:.2f}] to '
                      f'{k1_row["accuracy"]:.0%} [{k1_row["ci_lower"]:.2f}, '
                      f'{k1_row["ci_upper"]:.2f}] (95\\% bootstrap CI, '
                      f'n={n_str} prompts): a consistent directional improvement '
                      f'reflecting compensatory redistribution."')
                # Check CI overlap
                base_hi = base_row["ci_upper"]
                k1_lo   = k1_row["ci_lower"]
                if k1_lo > base_hi:
                    print("  → CIs do NOT overlap: improvement is statistically robust.")
                    print("     Use strong claim: 'statistically robust improvement'")
                else:
                    print("  → CIs overlap: use hedged language:")
                    print("     'a consistent directional improvement, though CIs overlap'")
            else:
                print("\nRe-run multi_ablation.py to get CI columns in multi_ablation.csv")
        except FileNotFoundError:
            print("multi_ablation.csv not found")

        print(f"\n→ Use these numbers in your paper's results section")
    except FileNotFoundError:
        print("ablation_sweep_ci.csv not found — run GPT-2 CI first")

if __name__ == "__main__":
    gpt2_df = run_gpt2_cis()
    bloom_df = run_bloom_cis()
    run_baseline_ci()
    run_per_language_cis()
    print_paper_stats()
    print(f"\n✓ Done. CI results saved to results/*_ci.csv")
    print("Next: run random_baseline.py")