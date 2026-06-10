"""
top_n_sweep.py — Validate ECCM top-N efficiency claim.

What this does
--------------
Reads the Step-5 merge CSV (learned-weight eccm column) for each task
and answers: "At candidate budget N, how much of the Lift@K signal does
ECCM retain compared to evaluating all pairs?"

For N in [10, 20, 50, 100, 200, 500, 1128]:
  1. Take the top-N pairs by ECCM score (same logic as select_top_pairs).
  2. Within those N pairs compute Precision@K and Lift@K for K in {10,20,50}.
  3. Print a table and save results/benchmarks/top_n_sweep_results.csv.

It does NOT run CMA-ES — only reads the existing CSV. Runtime < 5 seconds.

Relationship to select_top_pairs.py
------------------------------------
This script uses the same deduplication logic as select_top_pairs
(sort by eccm desc, then improvement desc, then drop_duplicates).
Both files must use identical deduplication so that select_top_pairs(N=100)
picks the exact same pairs that appear in the N=100 row of this sweep.

Output
------
Console table + results/benchmarks/top_n_sweep_results.csv

Auto-generated presentation claim
----------------------------------
At the end the script prints the validated claim sentence you can use
directly in the NBQSA presentation and Q&A.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
BASE = r"C:\\Users\\User\\Desktop\\ICTer\\WordTemplate-1"

TASKS = {
    "fraud": f"{BASE}\\results\\merges\\fraud\\merge_results_new_eccm.csv",
    "churn": f"{BASE}\\results\\merges\\churn\\merge_results_new_eccm.csv",
}

OUTPUT_DIR = f"{BASE}\\results\\benchmarks"
OUTPUT_CSV = f"{OUTPUT_DIR}\\top_n_sweep_results.csv"

# N values to test — fine-grained at the low end, coarser at the high end
# Note: 1128 = C(48,2) = 24 RF + 12 ET + 12 MLP pairs. Have to update if pool changes.
TOP_N_VALUES = [10, 20, 50, 100, 200, 500, 1128]

# K values for Precision@K / Lift@K within the budget
EVAL_AT_K = [10, 20, 50]

# Lift@10 thresholds used to auto-generate the presentation claim
# fraud: low threshold because 87.1% base rate makes most Lift values low
# churn: high threshold because 9.2% base rate amplifies ECCM's selectivity
LIFT_THRESHOLD = {"fraud": 1.10, "churn": 4.00}


# ── Core logic ────────────────────────────────────────────────────────────────

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per (model_a, model_b) pair.
    Identical deduplication logic to select_top_pairs.py —
    sort eccm desc, improvement desc, keep first.
    Both scripts MUST stay in sync on this logic.
    """
    return (
        df.sort_values(["eccm", "improvement"], ascending=[False, False])
        .drop_duplicates(subset=["model_a", "model_b"], keep="first")
        .reset_index(drop=True)
    )


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].mean())


def run_sweep(merge_csv: str, task: str) -> pd.DataFrame:
    """
    Phase 1 — Load and deduplicate.
    Phase 2 — Loop over N values, compute Precision@K and Lift@K.
    Phase 3 — Return results DataFrame.
    """
    # ── Phase 1 ───────────────────────────────────────────────────────────────
    df = pd.read_csv(merge_csv)
    best = _deduplicate(df)   # one row per pair, best blend ratio

    total_pairs = len(best)
    base_rate = (best["improvement"] > 0).mean()
    labels_all = (best["improvement"] > 0).astype(int).values
    scores_all = best["eccm"].values   # already sorted desc by eccm

    print(f"\n{'='*72}")
    print(
        f"  {task.upper()} — {total_pairs} unique pairs"
        f"  |  base success rate: {base_rate:.1%}"
    )
    print(f"{'='*72}")
    header = f"  {'N':>6}  {'budget':>11}"
    for k in EVAL_AT_K:
        header += f"  {'P@'+str(k):>6}  {'Lift@'+str(k):>7}"
    print(header)
    print(f"  {'-'*70}")

    rows = []

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    for n in TOP_N_VALUES:
        n_actual = min(n, total_pairs)

        # Slice the top-N rows (already sorted by eccm desc from _deduplicate)
        budget_scores = scores_all[:n_actual]
        budget_labels = labels_all[:n_actual]

        metrics = {}
        line = f"  {n_actual:>6}  {n_actual:>5}/{total_pairs:<5}"

        for k in EVAL_AT_K:
            prec = precision_at_k(budget_scores, budget_labels, k)
            # Lift is always relative to the GLOBAL base rate so values are
            # comparable across different N budgets
            lift = round(prec / base_rate, 3) if base_rate > 0 else float("nan")
            metrics[f"prec@{k}"] = round(prec, 4)
            metrics[f"lift@{k}"] = lift
            line += f"  {prec:>6.3f}  {lift:>7.3f}x"

        print(line)
        rows.append({
            "task": task,
            "N": n_actual,
            "total_pairs": total_pairs,
            "base_rate": round(base_rate, 4),
            **metrics,
        })

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    return pd.DataFrame(rows)


def _print_claim(df: pd.DataFrame) -> None:
    """
    Auto-generates the validated efficiency claim for the presentation.
    Finds the largest N where Lift@10 still meets the task threshold.
    """
    print("\n── Validated Presentation Claim ──────────────────────────────────────")
    for task in df["task"].unique():
        t = df[df["task"] == task]
        threshold = LIFT_THRESHOLD.get(task, 1.0)
        above = t[t["lift@10"] >= threshold]
        total = int(t["total_pairs"].iloc[0])

        if above.empty:
            print(
                f"  [{task.upper()}] Lift@10 never reaches {threshold}x. "
                f"Use N={total} (full pool)."
            )
        else:
            max_n = int(above["N"].max())
            lift10 = float(above.loc[above["N"] == max_n, "lift@10"].iloc[0])
            print(
                f"  [{task.upper()}] ECCM maintains Lift@10 ≥ {threshold}x "
                f"up to N={max_n} / {total} total pairs. "
                f"Lift@10 at N={max_n}: {lift10:.2f}x.\n"
                f"           → Set TOP_N_{task.upper()} = {max_n} "
                f"in merge_with_m2n2.py and M2N2_TOP_N_{task.upper()} "
                f"in benchmarks.py"
            )
    print("──────────────────────────────────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []

    for task, csv_path in TASKS.items():
        if not Path(csv_path).exists():
            print(
                f"[SKIP] {task}: {csv_path} not found. "
                f"Run merge_and_evaluate.py (Step 5) first."
            )
            continue
        result_df = run_sweep(csv_path, task)
        all_rows.append(result_df)

    if not all_rows:
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {OUTPUT_CSV}")

    _print_claim(combined)


if __name__ == "__main__":
    main()