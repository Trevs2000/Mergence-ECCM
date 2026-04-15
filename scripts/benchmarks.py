"""
benchmarks.py
 
Compares ECCM against four baselines to evaluate how well each method
predicts whether merging two models will improve AUC before the merge runs.
 
Baselines used:
  1. Random          - random scores, the theoretical floor (should score ~0.5 AUC-ROC)
  2. PSC-only        - cosine similarity of feature importances alone
  3. FSC-only        - prediction agreement between models alone
  4. AUC-Max         - score each pair by the stronger model's AUC
 
Why these four because,
  Random gives us a floor to beat.
  PSC and FSC are the two sub-metrics with the highest learned weights in ECCM,
  so isolating them tells us how much the composite score adds over its best
  individual signal (this answers RQ2 in thesis).
  AUC-Max is the intuitive baseline a practitioner would try first - it requires
  no compatibility calculation at all.
 
Metrics computed (per task, over 276 pairs at their best blend ratio):
  Spearman r  - rank correlation with actual AUC improvement (primary metric)
  p-value     - whether the correlation is statistically significant (want < 0.05)
  AUC-ROC     - can the score separate successful merges from unsuccessful ones?
  Precision@K - of the top-K ranked pairs, what fraction actually improved?
  Lift@K      - Precision@K divided by the base success rate (> 1.0 beats chance)

Outputs saved to results/benchmarks/:
    benchmark_results_{task}.csv  - per-pair scores for every method
    benchmark_summary.csv         - summary table across both tasks
"""
 
import os
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
 
 
# ── Config ────────────────────────────────────────────────────────────────────
 
TASKS = {
    "fraud": {
        "merge_csv":  "./results/merges/fraud/merge_results_new_eccm.csv",
        "models_dir": "./models/fraud",
    },
    "churn": {
        "merge_csv":  "./results/merges/churn/merge_results_new_eccm.csv",
        "models_dir": "./models/churn",
    },
}
 
OUTPUT_DIR         = "./results/benchmarks"
MAIN_VARIANT_RANGE = range(0, 24)  # v00-v23, same as MergePipeline
PRECISION_AT_K     = [10, 20, 50]
SEED               = 42
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def load_models(models_dir: str, variant_range) -> dict:
    """Load RF pkl files whose numeric suffix falls in variant_range."""
    models = {}
    for pkl in sorted(Path(models_dir).glob("*.pkl")):
        parts = pkl.stem.rsplit("_v", 1)
        if len(parts) != 2:
            continue
        try:
            vid = int(parts[1])
        except ValueError:
            continue
        if vid in variant_range:
            models[pkl.stem] = joblib.load(pkl)
    return models
 
 
def build_auc_map(pair_df: pd.DataFrame) -> dict:
    """
    Build a model_id -> AUC lookup from the collapsed pair DataFrame.
 
    Each model can appear as model_a in some pairs and model_b in others,
    so we collect from both roles row by row.
    """
    auc_map = {}
    for _, row in pair_df.iterrows():
        auc_map[row["model_a"]] = float(row["auc_a"])
        auc_map[row["model_b"]] = float(row["auc_b"])
    return auc_map
 
 
def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Of the top-K pairs ranked by score, what fraction are labelled successful?
 
    Formula: P@K = relevant items in top K / K
    """
    k = min(k, len(scores))
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].mean())
 
 
def evaluate(scores: np.ndarray, improvement: np.ndarray, success: np.ndarray) -> dict:
    """
    Compute all metrics for one scoring method.
 
    Args:
        scores:      predicted compatibility scores for each pair
        improvement: actual AUC improvement at the best blend ratio
        success:     1 if improvement > 0, else 0
 
    Returns:
        dict with spearman_r, spearman_p, auc_roc, prec@K for each K
    """
    r, p = spearmanr(scores, improvement)
 
    try:
        auc = roc_auc_score(success, scores)
    except ValueError:
        auc = float("nan")
 
    result = {
        "spearman_r": round(float(r), 4),
        "spearman_p": round(float(p), 6),
        "auc_roc":    round(float(auc), 4),
    }
    for k in PRECISION_AT_K:
        result[f"prec@{k}"] = round(precision_at_k(scores, success, k), 4)
    return result
 
 
# ── Baseline scorers ──────────────────────────────────────────────────────────
 
def score_random(n: int, seed: int = SEED) -> np.ndarray:
    """Uniformly random scores in [0, 1]. Fixed seed keeps results reproducible."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)
 
 
def score_auc_max(pair_ids: list, auc_map: dict) -> np.ndarray:
    """
    Score each pair by max(AUC_A, AUC_B).
 
    This is what a practitioner might try without any compatibility metric:
    assume the stronger parent model predicts merge success.
    """
    return np.array([max(auc_map[a], auc_map[b]) for a, b in pair_ids])
 
 
# ── Core pipeline ─────────────────────────────────────────────────────────────
 
def build_pair_df(merge_df: pd.DataFrame, models: dict, task: str) -> pd.DataFrame:
    """
    Collapse merge_df from one row per (pair x ratio) to one row per pair,
    keeping the best blend ratio result for each pair, then attach all scores.
 
    PSC, FSC, and ECCM come directly from the CSV since they were already
    computed in merge_and_evaluate.py (ECCM sub-metrics don't change across ratios).
    AUC-Max is computed here from the validation AUCs stored in the CSV.
    """
    # Keep the row with the highest improvement for each pair
    best = (
        merge_df
        .sort_values("improvement", ascending=False)
        .groupby(["model_a", "model_b"], sort=False)
        .first()
        .reset_index()
    )
 
    pair_ids = list(zip(best["model_a"], best["model_b"]))
    auc_map  = build_auc_map(best)
 
    # Baseline and ECCM scores
    best["score_random"]  = score_random(len(best), SEED)
    best["score_psc"]     = best["psc"].values
    best["score_fsc"]     = best["fsc"].values
    best["score_eccm"]    = best["eccm"].values
    best["score_auc_max"] = score_auc_max(pair_ids, auc_map)
 
    # Ground-truth labels
    # gt_improvement: actual AUC gain at the best blend ratio
    # gt_success: 1 if the merge improved AUC at all, else 0
    best["gt_improvement"] = best["improvement"].values
    best["gt_success"]     = (best["improvement"] > 0).astype(int).values
    best["task"]           = task
 
    return best
 
 
def build_summary(pair_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Evaluate all methods and return a summary row per method."""
    improvement = pair_df["gt_improvement"].values
    success     = pair_df["gt_success"].values
    base_rate   = success.mean()
 
    methods = {
        "Random":      pair_df["score_random"].values,
        "PSC-only":    pair_df["score_psc"].values,
        "FSC-only":    pair_df["score_fsc"].values,
        "AUC-Max":     pair_df["score_auc_max"].values,
        "ECCM (full)": pair_df["score_eccm"].values,
    }
 
    rows = []
    for name, scores in methods.items():
        ev = evaluate(scores, improvement, success)
 
        # Lift@K: how many times better than random at picking good pairs?
        # > 1.0 means the method beats random chance at this cutoff
        lift = {f"lift@{k}": round(ev[f"prec@{k}"] / base_rate, 3) for k in PRECISION_AT_K}
 
        rows.append({
            "task":              task,
            "method":            name,
            "spearman_r":        ev["spearman_r"],
            "spearman_p":        ev["spearman_p"],
            "auc_roc":           ev["auc_roc"],
            **{f"prec@{k}": ev[f"prec@{k}"] for k in PRECISION_AT_K},
            **lift,
            "base_success_rate": round(base_rate, 4),
            "n_pairs":           len(pair_df),
        })
 
    return pd.DataFrame(rows)
 
 
def print_summary(df: pd.DataFrame, task: str) -> None:
    t    = df[df["task"] == task].copy()
    sep  = "-" * 95
    base = t["base_success_rate"].iloc[0]
    n    = int(t["n_pairs"].iloc[0])
 
    print(f"\n{'='*95}")
    print(f"  BENCHMARK - {task.upper()}   ({n} pairs, base success rate: {base:.1%})")
    print(f"{'='*95}")
 
    header = f"{'Method':<18} {'Spear.r':>9} {'p-val':>8} {'AUC-ROC':>9}"
    for k in PRECISION_AT_K:
        header += f" {'P@'+str(k):>7}"
    for k in PRECISION_AT_K:
        header += f" {'Lift@'+str(k):>8}"
    print(header)
    print(sep)
 
    for _, row in t.iterrows():
        line = f"{row['method']:<18} {row['spearman_r']:>9.4f} {row['spearman_p']:>8.4f} {row['auc_roc']:>9.4f}"
        for k in PRECISION_AT_K:
            line += f" {row[f'prec@{k}']:>7.3f}"
        for k in PRECISION_AT_K:
            line += f" {row[f'lift@{k}']:>8.3f}"
        if "ECCM" in row["method"]:
            line = ">" + line[1:]
        print(line)
 
    print(sep)
    print(
        "  Spearman r : rank correlation with actual AUC improvement\n"
        "  p-val      : < 0.05 means the correlation is statistically significant\n"
        "  AUC-ROC    : ability to separate successful merges from unsuccessful ones\n"
        "  P@K        : fraction of top-K pairs that actually improved AUC\n"
        f"  Lift@K     : P@K / {base:.3f} - values > 1.0 beat random pair selection\n"
    )
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_summaries = []
 
    for task, cfg in TASKS.items():
        merge_csv  = cfg["merge_csv"]
        models_dir = cfg["models_dir"]
 
        if not Path(merge_csv).exists():
            print(f"[SKIP] {task}: {merge_csv} not found. Run merge_and_evaluate.py first.")
            continue
 
        print(f"\n[{task.upper()}] Loading merge results...")
        merge_df = pd.read_csv(merge_csv)
 
        print(f"[{task.upper()}] Loading models...")
        models = load_models(models_dir, MAIN_VARIANT_RANGE)
        if not models:
            print(f"[SKIP] {task}: no models found in {models_dir}.")
            continue
 
        print(f"[{task.upper()}] Building scores and evaluating...")
        pair_df  = build_pair_df(merge_df, models, task)
        pair_csv = f"{OUTPUT_DIR}/benchmark_results_{task}.csv"
        pair_df.to_csv(pair_csv, index=False)
        print(f"  Saved → {pair_csv}")
 
        summary = build_summary(pair_df, task)
        all_summaries.append(summary)
        print_summary(summary, task)
 
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        out_path = f"{OUTPUT_DIR}/benchmark_summary.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSummary saved → {out_path}")
 
 
if __name__ == "__main__":
    main()