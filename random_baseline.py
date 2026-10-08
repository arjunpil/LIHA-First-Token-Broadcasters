"""
random_baseline.py — Establish a random ablation baseline.

Key question: if you ablate a randomly chosen head, what switch rate
do you expect by chance? This baseline makes the top head findings
meaningful by comparison.

We compute:
1. Mean switch rate across ALL heads (the true random baseline)
2. Distribution of switch rates (mean, std, percentiles)
3. How many standard deviations above mean L6H1 is
4. Same comparison for BLOOM

No GPU needed — reads existing CSV results.
Run time: ~1 minute
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

RESULTS_DIR = "results"

def analyze_random_baseline(df, model_name, top_head_id, top_head_switch_rate):
    print(f"\n=== {model_name} RANDOM ABLATION BASELINE ===")

    switch_rates = df["switch_rate"].values
    mean_sr   = switch_rates.mean()
    std_sr    = switch_rates.std()
    median_sr = np.median(switch_rates)
    p75_sr    = np.percentile(switch_rates, 75)
    p90_sr    = np.percentile(switch_rates, 90)
    p95_sr    = np.percentile(switch_rates, 95)

    print(f"Total heads:         {len(df)}")
    print(f"Mean switch rate:    {mean_sr:.4f}")
    print(f"Std switch rate:     {std_sr:.4f}")
    print(f"Median switch rate:  {median_sr:.4f}")
    print(f"75th percentile:     {p75_sr:.4f}")
    print(f"90th percentile:     {p90_sr:.4f}")
    print(f"95th percentile:     {p95_sr:.4f}")
    print(f"Max switch rate:     {switch_rates.max():.4f} ({top_head_id})")

    # Z-score of top head
    z = (top_head_switch_rate - mean_sr) / std_sr
    print(f"\n{top_head_id} is {z:.2f} standard deviations above mean")
    print(f"Effect size vs random: {top_head_switch_rate / mean_sr:.1f}x")

    # One-sample t-test: is top head switch rate significantly above mean?
    # H0: top head switch rate = population mean
    # Use bootstrap approach since we have the full distribution
    n_above = (switch_rates >= top_head_switch_rate).sum()
    percentile_rank = 100 * (1 - n_above / len(switch_rates))
    print(f"{top_head_id} is at the {percentile_rank:.1f}th percentile of all heads")

    # Fraction of heads that are "significantly" impactful
    # Define threshold as mean + 2*std
    threshold = mean_sr + 2 * std_sr
    n_significant = (switch_rates > threshold).sum()
    print(f"\nHeads above mean + 2σ ({threshold:.4f}): {n_significant} "
          f"({100*n_significant/len(df):.1f}%)")
    print("→ These are the 'significantly impactful' heads for the paper")

    # Save distribution stats
    stats_dict = {
        "model": model_name,
        "n_heads": len(df),
        "mean_switch_rate": mean_sr,
        "std_switch_rate": std_sr,
        "median_switch_rate": median_sr,
        "p75": p75_sr,
        "p90": p90_sr,
        "p95": p95_sr,
        "max_switch_rate": switch_rates.max(),
        "top_head": top_head_id,
        "top_head_switch_rate": top_head_switch_rate,
        "top_head_z_score": z,
        "top_head_effect_size": top_head_switch_rate / mean_sr,
        "top_head_percentile": percentile_rank,
        "n_above_threshold": n_significant,
        "threshold_2sigma": threshold,
    }
    return stats_dict

def compare_models(gpt2_stats, bloom_stats):
    print("\n=== CROSS-MODEL RANDOM BASELINE COMPARISON ===")
    print(f"{'Metric':<35} {'GPT-2':>10} {'BLOOM':>10}")
    print("-" * 57)
    metrics = [
        ("Mean switch rate (random head)", "mean_switch_rate"),
        ("Std switch rate", "std_switch_rate"),
        ("Top head switch rate", "top_head_switch_rate"),
        ("Top head z-score", "top_head_z_score"),
        ("Top head effect size (vs random)", "top_head_effect_size"),
        ("Top head percentile", "top_head_percentile"),
    ]
    for label, key in metrics:
        g = gpt2_stats[key]
        b = bloom_stats[key]
        if isinstance(g, float):
            print(f"{label:<35} {g:>10.3f} {b:>10.3f}")
        else:
            print(f"{label:<35} {str(g):>10} {str(b):>10}")

    print(f"\n→ Key paper sentence:")
    print(f"  GPT-2: L6H1's switch rate of {gpt2_stats['top_head_switch_rate']:.2f} is "
          f"{gpt2_stats['top_head_z_score']:.1f}σ above the mean random-head rate of "
          f"{gpt2_stats['mean_switch_rate']:.3f}, placing it at the "
          f"{gpt2_stats['top_head_percentile']:.0f}th percentile of all heads.")
    print(f"  BLOOM: L9H14's switch rate of {bloom_stats['top_head_switch_rate']:.2f} is "
          f"{bloom_stats['top_head_z_score']:.1f}σ above mean of "
          f"{bloom_stats['mean_switch_rate']:.3f}.")

if __name__ == "__main__":
    # GPT-2
    gpt2_df = pd.read_csv(f"{RESULTS_DIR}/ablation_sweep.csv")
    gpt2_stats = analyze_random_baseline(
        gpt2_df, "GPT-2", "L6H1", 0.32
    )

    # BLOOM
    try:
        bloom_df = pd.read_csv(f"{RESULTS_DIR}/bloom_ablation_sweep.csv")
        bloom_stats = analyze_random_baseline(
            bloom_df, "BLOOM-1b7", "L9H14", 0.20
        )
        compare_models(gpt2_stats, bloom_stats)
    except FileNotFoundError:
        print("\nbloom_ablation_sweep.csv not found, skipping BLOOM analysis")

    # Save stats
    results = pd.DataFrame([gpt2_stats, bloom_stats] if "bloom_stats" in dir() else [gpt2_stats])
    results.to_csv(f"{RESULTS_DIR}/random_baseline_stats.csv", index=False)
    print(f"\n✓ Done. Stats saved to {RESULTS_DIR}/random_baseline_stats.csv")
    print("Next: run probing_experiment.py")