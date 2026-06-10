"""
xai_explanantions.py

Generates plain-English XAI narratives and bar-chart plots for
the top-N merge pairs.  Used by run_xai_global.py.

Changes from previous version:
- load_m2n2_results() now accepts top_n parameter and builds the correct
  filename m2n2_results_topN{top_n}.csv to match merge_with_m2n2.py output.
- explain_pair() guards against NaN opt_improvement (pair not in CMA-ES budget).
- plot_pair() skips the AUC comparison chart gracefully when opt_best_auc
  is NaN rather than crashing on min(vals) with a NaN in the list.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fixed_results(task: str) -> pd.DataFrame:
    """Load the full fixed-ratio merge results (all 1,128 pairs × 5 ratios)."""
    return pd.read_csv(f"./results/merges/{task}/merge_results_new_eccm.csv")


def load_m2n2_results(task: str, top_n: int = 100) -> pd.DataFrame:
    """
    Load the CMA-ES results for the top-N budget run.

    Args:
        task  : "fraud" or "churn"
        top_n : must match TOP_N used in merge_with_m2n2.py (default 100)

    Returns empty DataFrame (not a crash) if the file does not exist yet,
    so run_xai_global.py can still produce fixed-ratio XAI output while
    merge_with_m2n2.py is still running.
    """
    path = Path(f"./results/merges/{task}/m2n2_results_topN{top_n}.csv")
    if not path.exists():
        print(
            f"  [XAI] {task}: {path.name} not found. "
            f"CMA-ES columns will be NaN — fixed-ratio XAI will still run."
        )
        return pd.DataFrame()
    return pd.read_csv(path)


# ── XAI narrative ─────────────────────────────────────────────────────────────

def explain_pair(row: pd.Series, task: str) -> str:
    """
    Generate a plain-English explanation for a single merged pair.

    Args:
        row:  a row from the joined fixed + m2n2 DataFrame
              (produced by run_xai_global joining on model_a, model_b)
        task: "fraud" or "churn"

    Returns:
        Multi-sentence explanation string.
        CMA-ES sentences are omitted if opt_improvement is NaN
        (pair was not in the top-N CMA-ES budget).
    """
    a, b   = row["model_a"], row["model_b"]
    psc    = row["psc"]
    fsc    = row["fsc"]
    rsc    = row["rsc"]
    eccm   = row["eccm_fixed"]
    base_i = row["improvement"]

    agreement = (
        "often agree"        if fsc > 0.90 else
        "agree on most cases" if fsc > 0.65 else
        "frequently disagree"
    )
    ranking = "very similar" if rsc > 0.90 else "moderately similar"

    lines = [
        f"For {task}, models {a} and {b} were selected with ECCM={eccm:.3f}.",
        f"PSC={psc:.3f}, FSC={fsc:.3f}, RSC={rsc:.3f}: "
        f"predictions {agreement} and feature rankings are {ranking}.",
        (f"Fixed-ratio merging improved AUC by {base_i:.6f}."
         if base_i >= 0
         else f"Fixed-ratio merging reduced AUC by {abs(base_i):.6f}."),
    ]

    # CMA-ES sentences — only when this pair was in the top-N budget
    opt_i = row.get("opt_improvement", np.nan)
    opt_r = row.get("opt_best_ratio",  np.nan)
    delta = row.get("opt_vs_fixed",    np.nan)

    if pd.notna(opt_i) and pd.notna(opt_r) and pd.notna(delta):
        lines.append(f"CMA-ES found an optimal blend ratio of {opt_r:.3f} for {a}.")
        lines.append(
            f"This gained an additional {delta:.6f} AUC over the fixed grid."
            if delta > 0
            else f"The fixed grid was already near-optimal (CMA-ES delta = {delta:.6f})."
        )
    else:
        lines.append(
            "This pair was not in the CMA-ES top-N budget; "
            "blend-ratio optimisation was not run."
        )

    return " ".join(lines)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pair(row: pd.Series, task: str, out_dir: str) -> None:
    """
    Save two bar charts for a pair:
      1. PSC / FSC / RSC sub-metric scores
      2. Best parent AUC vs fixed-merge AUC vs CMA-ES-merge AUC
         (chart 2 is skipped if opt_best_auc is NaN)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    a, b = row["model_a"], row["model_b"]
    stem = f"{task}_{a}_{b}"

    # ── Chart 1: sub-metrics bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        ["PSC", "FSC", "RSC"],
        [row["psc"], row["fsc"], row["rsc"]],
        color=["#4c72b0", "#55a868", "#c44e52"],
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"{task} — {a} + {b}")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{stem}_metrics.png", dpi=150)
    plt.close(fig)

    # ── Chart 2: AUC comparison bar ───────────────────────────────────────
    # Requires opt_best_auc from the CMA-ES results.
    # Skipped gracefully when the pair was not in the top-N budget.
    opt_auc       = row.get("opt_best_auc",    np.nan)
    best_par_auc  = row.get("best_parent_auc", np.nan)
    fixed_best    = row.get("fixed_best_auc",  np.nan)

    if any(pd.isna(v) for v in [opt_auc, best_par_auc, fixed_best]):
        print(
            f"  [XAI] Skipping AUC chart for {a}+{b} "
            f"(CMA-ES columns not available for this pair)."
        )
        return

    vals   = [best_par_auc, fixed_best, opt_auc]
    labels = ["Best parent", "Fixed merge", "CMA-ES merge"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    y_min = min(vals) - 0.001
    y_max = max(vals) + 0.001
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"{task} — {a} + {b} AUC")
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{stem}_auc.png", dpi=150)
    plt.close(fig)