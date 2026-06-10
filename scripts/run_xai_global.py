# run_xai_global.py — full updated file

"""
run_xai_global.py - CLI runner for global XAI reports.

Joins fixed-ratio results with CMA-ES results, picks the top-N pairs
by ECCM, prints a narrative explanation, and saves bar-chart PNGs.

Change: M2N2_TOP_N constant added and passed to load_m2n2_results()
so the correct m2n2_results_topN{N}.csv file is loaded.
Must match TOP_N in merge_with_m2n2.py, benchmarks.py, eccm_ablation.py.
"""

import pandas as pd
from scripts.xai_explanantions import (
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
        # CMA-ES hasn't run yet — still produce fixed-ratio XAI
        joined = (
            fixed
            .sort_values(["eccm", "improvement"], ascending=[False, False])
            .drop_duplicates(subset=["model_a", "model_b"], keep="first")
            .reset_index(drop=True)
            .rename(columns={"eccm": "eccm_fixed"})
            .copy()
        )
        for col in ["opt_best_ratio", "opt_improvement", "opt_vs_fixed",
                    "opt_best_auc", "best_parent_auc", "fixed_best_auc"]:
            joined[col] = float("nan")
        # eccm column has no suffix in this path — rename to match explain_pair
        joined = joined.rename(columns={"eccm": "eccm_fixed"})
    else:
        # Join on model_a + model_b; suffixes disambiguate shared column names
        joined = fixed.merge(
            m2n2, on=["model_a", "model_b"], suffixes=("_fixed", "_opt")
        )

    top = joined.sort_values("eccm_fixed", ascending=False).head(top_n)

    for _, row in top.iterrows():
        print("\n--- " + task + ": " + row["model_a"] + " + " + row["model_b"] + " ---")
        print(explain_pair(row, task))
        plot_pair(row, task, out_dir="./results/xai/" + task)


if __name__ == "__main__":
    run_for_task("fraud", top_n=5)
    run_for_task("churn", top_n=5)