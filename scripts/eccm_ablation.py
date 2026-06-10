"""
eccm_ablation.py

ECCM ablation study.

Compares how well PSC, FSC, RSC, and full ECCM correlate with actual
merge improvement — both at fixed blend ratios and CMA-ES optimised.

Changes from previous version:
- analyse_file(): deduplicates to one row per pair (best improvement)
  before computing Spearman r. The old version included all 5 blend-ratio
  rows per pair, inflating N and biasing correlation toward middle ratios.
- analyse_m2n2(): same deduplication on the fixed CSV before inner join.
- m2n2 CSV paths in main() now use m2n2_results_topN{N}.csv to match
  the file produced by the updated merge_with_m2n2.py.
- M2N2_TOP_N constant added — update this whenever TOP_N changes in
  merge_with_m2n2.py.
"""

import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


# ── Shared deduplication helper ───────────────────────────────────────────────

def _best_per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per (model_a, model_b) pair, keeping the row with
    the highest improvement (i.e. the best blend ratio).

    Used in both analyse_file() and analyse_m2n2() so that Spearman r is
    computed over unique pairs, not over 5 blend-ratio rows per pair.
    Identical dedup logic to select_top_pairs.py and top_n_sweep.py.
    """
    return (
        df.sort_values(["eccm", "improvement"], ascending=[False, False])
        .drop_duplicates(subset=["model_a", "model_b"], keep="first")
        .reset_index(drop=True)
    )


# ── Analysis functions ────────────────────────────────────────────────────────

def analyse_file(csvpath: str, name: str) -> None:
    """
    Spearman r of each sub-metric vs fixed-ratio improvement.
    Operates on deduplicated pairs (one row per pair, best blend ratio).
    """
    if not Path(csvpath).exists():
        print(f"\n[SKIP] {name}: {csvpath} not found.")
        return

    df = _best_per_pair(pd.read_csv(csvpath))
    metrics = ["psc", "fsc", "rsc", "eccm"]

    print(f"\n{name} — fixed-ratio improvement  ({len(df)} unique pairs)")
    print("-" * 60)
    print(f"{'Metric':8} {'Spearman r':>11} {'p-value':>11}")
    print("-" * 60)
    for m in metrics:
        r, p = spearmanr(df[m], df["improvement"])
        print(f"{m:8} {r:11.4f} {p:11.4g}")


def analyse_m2n2(fixed_csv: str, m2n2_csv: str, name: str) -> None:
    """
    Spearman r of each sub-metric vs CMA-ES optimised improvement.
    Inner join on unique (model_a, model_b) pairs — only pairs that
    were in the top-N budget and ran through CMA-ES are included.
    """
    if not Path(fixed_csv).exists():
        print(f"\n[SKIP] {name} M2N2: {fixed_csv} not found.")
        return
    if not Path(m2n2_csv).exists():
        print(f"\n[SKIP] {name} M2N2: {m2n2_csv} not found. Run merge_with_m2n2.py first.")
        return

    fixed  = _best_per_pair(pd.read_csv(fixed_csv))
    m2n2   = pd.read_csv(m2n2_csv)
    joined = fixed.merge(
        m2n2[["model_a", "model_b", "opt_improvement"]],
        on=["model_a", "model_b"],
        how="inner",
    )

    metrics = ["psc", "fsc", "rsc", "eccm"]
    print(f"\n{name} — CMA-ES opt_improvement  ({len(joined)} pairs in top-N budget)")
    print("-" * 60)
    print(f"{'Metric':8} {'Spearman r':>11} {'p-value':>11}")
    print("-" * 60)
    for m in metrics:
        r, p = spearmanr(joined[m], joined["opt_improvement"])
        print(f"{m:8} {r:11.4f} {p:11.4g}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    BASE = r"C:\Users\User\Desktop\ICTer\WordTemplate-1"

    # Must match TOP_N_FRAUD / TOP_N_CHURN in merge_with_m2n2.py
    # and M2N2_TOP_N in benchmarks.py
    M2N2_TOP_N = 100

    fraud_fixed = fr"{BASE}\results\merges\fraud\merge_results_new_eccm.csv"
    churn_fixed = fr"{BASE}\results\merges\churn\merge_results_new_eccm.csv"
    fraud_m2n2  = fr"{BASE}\results\merges\fraud\m2n2_results_topN{M2N2_TOP_N}.csv"
    churn_m2n2  = fr"{BASE}\results\merges\churn\m2n2_results_topN{M2N2_TOP_N}.csv"

    # Fixed-ratio ablation (all 1,128 pairs, best blend ratio per pair)
    analyse_file(fraud_fixed, "FRAUD")
    analyse_file(churn_fixed, "CHURN")

    # CMA-ES ablation (top-N pairs only - those that ran through merge_with_m2n2.py)
    analyse_m2n2(fraud_fixed, fraud_m2n2, "FRAUD")
    analyse_m2n2(churn_fixed, churn_m2n2, "CHURN")


if __name__ == "__main__":
    main()