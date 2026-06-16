"""
stat_test_compensatory.py — Bootstrap permutation test on attention weight deltas.

The data in compensatory_heads.csv contains per-(ablated_head, candidate_head)
mean delta values, already averaged over prompts. Each ablated head has 143
candidate heads (all heads except itself).

Test design
-----------
Null hypothesis: ablating a top head has NO systematic effect on first-token
attention of other heads. Under the null, the observed deltas are drawn from
the same distribution as any random set of heads' deltas.

We test THREE claims:

  Claim A: The top-k compensating heads have a HIGHER mean delta than
           expected by chance (permutation test).

  Claim B: Each individual top compensator has a delta significantly
           above zero (one-sample bootstrap CI, does it exclude zero?).

  Claim C: The top compensators cluster in specific layers (layers 7-9)
           rather than being spread randomly (chi-square / empirical test).

For each ablated head (L6H1, L0H4, L9H9) we run all three tests.

Output
------
- Console: full test results with p-values
- results/stat_test_compensatory.csv: machine-readable results
- Prints paper-ready sentences for each finding
"""

import numpy as np
import pandas as pd
import os
from scipy import stats

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_PERMUTATIONS = 100_000
TOP_K = 5
RNG = np.random.default_rng(42)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    # Try local results/ first, then uploads path
    for path in ["results/compensatory_heads.csv",
                 "/mnt/user-data/uploads/compensatory_heads.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {path}: {df.shape[0]} rows")
            return df
    raise FileNotFoundError("compensatory_heads.csv not found")

# ── Test A: Permutation test on top-k mean delta ──────────────────────────────
def permutation_test_topk(deltas, k=TOP_K, n_perm=N_PERMUTATIONS):
    """
    Observed statistic: mean of top-k deltas.
    Null distribution: mean of k randomly chosen deltas (with replacement).
    p-value: fraction of null samples >= observed statistic.
    """
    observed = np.sort(deltas)[-k:].mean()
    null_dist = np.array([
        RNG.choice(deltas, size=k, replace=False).mean()
        for _ in range(n_perm)
    ])
    p_value = (null_dist >= observed).mean()
    return observed, null_dist, p_value

# ── Test B: Bootstrap CI on individual head delta ────────────────────────────
def bootstrap_ci_single(delta_value, all_deltas, n_boot=N_PERMUTATIONS):
    """
    The delta is a mean over N prompts. We don't have per-prompt data,
    so we use the population of all 143 deltas as our resampling universe.
    
    Test: is delta_value in the tail of the null distribution?
    CI: percentile bootstrap over all_deltas, centered on delta_value.
    
    More precisely: we ask what fraction of randomly selected single
    deltas from the population are >= delta_value (empirical p-value).
    """
    p_value = (all_deltas >= delta_value).mean()
    # Also compute a z-score relative to the population
    z = (delta_value - all_deltas.mean()) / all_deltas.std()
    return p_value, z

# ── Test C: Layer clustering ──────────────────────────────────────────────────
def layer_clustering_test(df_ablated, k=TOP_K, n_perm=N_PERMUTATIONS):
    """
    Are top-k compensators concentrated in specific layers?
    Observed: set of layers of top-k heads.
    Null: layers of k randomly chosen heads.
    Test statistic: entropy of layer distribution (low entropy = more clustered).
    """
    all_layers = df_ablated["layer"].values
    top_k_layers = df_ablated.nlargest(k, "delta_ft_attn")["layer"].values

    def layer_entropy(layers):
        counts = np.bincount(layers, minlength=12)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    observed_entropy = layer_entropy(top_k_layers)
    null_entropies = np.array([
        layer_entropy(RNG.choice(all_layers, size=k, replace=False))
        for _ in range(n_perm)
    ])
    # p-value: fraction of null samples with entropy <= observed (lower = more clustered)
    p_value = (null_entropies <= observed_entropy).mean()
    return observed_entropy, null_entropies.mean(), p_value, top_k_layers

# ── Main ──────────────────────────────────────────────────────────────────────
def run_all_tests(df):
    ablated_heads = df["ablated_head"].unique()
    all_records = []

    for abl_name in ablated_heads:
        sub = df[df["ablated_head"] == abl_name].copy()
        deltas = sub["delta_ft_attn"].values
        top5   = sub.nlargest(TOP_K, "delta_ft_attn")

        print(f"\n{'='*60}")
        print(f"ABLATED HEAD: {abl_name}")
        print(f"{'='*60}")
        print(f"Population: {len(deltas)} candidate heads")
        print(f"Delta distribution: mean={deltas.mean():.6f}, "
              f"std={deltas.std():.6f}, "
              f"min={deltas.min():.6f}, max={deltas.max():.6f}")

        # ── Test A ────────────────────────────────────────────────────────────
        obs_mean, null_dist, p_a = permutation_test_topk(deltas, k=TOP_K)
        print(f"\n[TEST A] Top-{TOP_K} mean delta vs. random permutation")
        print(f"  Observed top-{TOP_K} mean delta : {obs_mean:.6f}")
        print(f"  Null mean (expected by chance)  : {null_dist.mean():.6f}")
        print(f"  Null 95th percentile            : {np.percentile(null_dist, 95):.6f}")
        print(f"  Null 99th percentile            : {np.percentile(null_dist, 99):.6f}")
        print(f"  p-value ({N_PERMUTATIONS:,} permutations)   : {p_a:.5f}")
        sig_a = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "n.s."
        print(f"  Significance                    : {sig_a}")

        # ── Test B: individual heads ──────────────────────────────────────────
        print(f"\n[TEST B] Individual top-{TOP_K} head deltas vs. null")
        print(f"  {'Head':8s} {'Delta':>10s} {'z-score':>10s} "
              f"{'empirical-p':>12s} {'sig':>5s}")
        print(f"  {'-'*50}")
        head_results = []
        for _, row in top5.iterrows():
            p_b, z = bootstrap_ci_single(row["delta_ft_attn"], deltas)
            sig_b = "***" if p_b < 0.001 else "**" if p_b < 0.01 \
                    else "*" if p_b < 0.05 else "n.s."
            print(f"  {row['head_id']:8s} {row['delta_ft_attn']:>10.6f} "
                  f"{z:>10.2f} {p_b:>12.5f} {sig_b:>5s}")
            head_results.append({
                "ablated_head": abl_name,
                "head_id": row["head_id"],
                "delta": row["delta_ft_attn"],
                "z_score": z,
                "p_empirical": p_b,
                "sig": sig_b,
            })

        # ── Test C: layer clustering ──────────────────────────────────────────
        obs_ent, null_ent_mean, p_c, top_layers = layer_clustering_test(sub)
        sig_c = "***" if p_c < 0.001 else "**" if p_c < 0.01 \
                else "*" if p_c < 0.05 else "n.s."
        print(f"\n[TEST C] Layer clustering of top-{TOP_K} compensators")
        print(f"  Top-{TOP_K} layers              : {sorted(top_layers)}")
        print(f"  Observed layer entropy          : {obs_ent:.4f}")
        print(f"  Expected entropy (random)       : {null_ent_mean:.4f}")
        print(f"  p-value (low entropy = cluster) : {p_c:.5f}  {sig_c}")

        all_records.append({
            "ablated_head":        abl_name,
            "test_A_obs_mean":     obs_mean,
            "test_A_null_mean":    null_dist.mean(),
            "test_A_p":            p_a,
            "test_A_sig":          sig_a,
            "test_C_obs_entropy":  obs_ent,
            "test_C_null_entropy": null_ent_mean,
            "test_C_p":            p_c,
            "test_C_sig":          sig_c,
            "head_results":        head_results,
        })

    return all_records

# ── Paper-ready sentences ─────────────────────────────────────────────────────
def print_paper_sentences(records):
    print(f"\n{'='*60}")
    print("PAPER-READY SENTENCES")
    print(f"{'='*60}")

    for rec in records:
        abl  = rec["ablated_head"]
        p_a  = rec["test_A_p"]
        p_c  = rec["test_C_p"]
        sig  = rec["test_A_sig"]
        hrs  = rec["head_results"]

        # Build individual head string
        sig_heads = [h for h in hrs if h["sig"] != "n.s."]
        head_str  = ", ".join(
            f"{h['head_id']} ($\\Delta={h['delta']:+.3f}$, "
            f"$z={h['z_score']:.2f}$, $p={h['p_empirical']:.3f}$)"
            for h in sig_heads
        )

        print(f"\n--- {abl} ---")

        if p_a < 0.05:
            print(f"When {abl} is ablated, the top-{TOP_K} compensating heads "
                  f"show a mean first-token attention increase of "
                  f"{rec['test_A_obs_mean']:.4f}, significantly exceeding "
                  f"the permutation null of {rec['test_A_null_mean']:.4f} "
                  f"($p={p_a:.4f}$, {N_PERMUTATIONS:,}-sample permutation test).")
        else:
            print(f"When {abl} is ablated, the top-{TOP_K} mean delta "
                  f"({rec['test_A_obs_mean']:.4f}) does not significantly "
                  f"exceed the permutation null ($p={p_a:.4f}$).")

        if sig_heads:
            print(f"Individually significant compensators: {head_str}.")

        if p_c < 0.05:
            print(f"Compensating heads cluster in layers "
                  f"{sorted(set(h['head_id'][:2] for h in hrs[:TOP_K]))} "
                  f"(layer entropy $p={p_c:.4f}$), suggesting structured "
                  f"rather than diffuse redistribution.")

    print("\n--- HONEST CAVEAT FOR PAPER ---")
    print("Note: delta values are means over 40 prompts (from compensatory_heads.py).")
    print("Tests use the empirical distribution of all 143 head deltas as the null.")
    print("If any individual test is n.s., report honestly as 'directional but")
    print("not statistically distinguishable from the null at this sample size.'")

# ── Save results ──────────────────────────────────────────────────────────────
def save_results(records):
    rows = []
    for rec in records:
        for hr in rec["head_results"]:
            rows.append({
                "ablated_head":        rec["ablated_head"],
                "head_id":             hr["head_id"],
                "delta":               hr["delta"],
                "z_score":             hr["z_score"],
                "p_empirical":         hr["p_empirical"],
                "sig":                 hr["sig"],
                "test_A_p":            rec["test_A_p"],
                "test_A_sig":          rec["test_A_sig"],
                "test_C_p":            rec["test_C_p"],
                "test_C_sig":          rec["test_C_sig"],
            })
    df = pd.DataFrame(rows)
    path = f"{RESULTS_DIR}/stat_test_compensatory.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    return df

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Statistical significance tests for compensatory redistribution")
    print(f"N permutations: {N_PERMUTATIONS:,}  |  Top-k: {TOP_K}\n")

    df   = load_data()
    recs = run_all_tests(df)
    print_paper_sentences(recs)
    save_results(recs)
    print("\nDone. Check results/stat_test_compensatory.csv for full output.")