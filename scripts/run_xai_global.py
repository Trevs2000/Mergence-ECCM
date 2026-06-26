"""
run_xai_global.py - CLI runner for global XAI reports.

Joins fixed-ratio results with CMA-ES results, picks the top-N pairs
by ECCM, prints a narrative explanation, and saves bar-chart PNGs.

Changes from previous version:
- Passes M2N2_TOP_N into load_m2n2_results().
- Uses fallback mode when the CMA-ES file is absent.
- Deduplicates fixed-ratio rows to one row per pair in fallback mode.
"""

from pathlib import Path

import pandas as pd
from scripts.xai_explanantions import (
    BASE,
    load_fixed_results,
    load_m2n2_results,
    explain_pair,
    plot_pair,
)

# Must match TOP_N_FRAUD / TOP_N_CHURN in merge_with_m2n2.py
M2N2_TOP_N = 100


def run_for_task(task: str, top_n: int = 5):
    fixed = load_fixed_results(task)
    m2n2  = load_m2n2_results(task, top_n=M2N2_TOP_N)

    if m2n2.empty:
        # CMA-ES file missing: still run fixed-ratio XAI safely.
        joined = (
            fixed
            .sort_values(["eccm", "improvement"], ascending=[False, False])
            .drop_duplicates(subset=["model_a", "model_b"], keep="first")
            .reset_index(drop=True)
            .rename(columns={"eccm": "eccm_fixed"})
            .copy()
        )

        for col in [
            "opt_best_ratio",
            "opt_improvement",
            "opt_vs_fixed",
            "opt_best_auc",
            "best_parent_auc",
            "fixed_best_auc",
        ]:
            joined[col] = float("nan")
    else:
        # Join on pair id; suffixes handle overlapping column names.
        joined = fixed.merge(
            m2n2,
            on=["model_a", "model_b"],
            how="inner",
            suffixes=("_fixed", "_opt"),
        )

        # Keep one fixed-ratio row per pair: the best improvement row.
        joined = (
            joined
            .sort_values(["eccm_fixed", "improvement"], ascending=[False, False])
            .drop_duplicates(subset=["model_a", "model_b"], keep="first")
            .reset_index(drop=True)
        )

    top = joined.sort_values("eccm_fixed", ascending=False).head(top_n)

    out_dir = Path(f"{BASE}\\results\\xai\\{task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in top.iterrows():
        print(f"\n--- {task}: {row['model_a']} + {row['model_b']} ---")
        print(explain_pair(row, task))
        plot_pair(row, task, out_dir=str(out_dir))


if __name__ == "__main__":
    run_for_task("fraud", top_n=5)
    run_for_task("churn", top_n=5)