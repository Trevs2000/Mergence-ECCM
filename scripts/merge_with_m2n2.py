"""
merge_with_m2n2.py
 
CMA-ES blend ratio optimisation for top-N ECCM-selected model pairs.
 
Changes from previous version (NN extension):
  - CMAESMerger.optimise() previously called model_a.predict_proba() and
    model_b.predict_proba() directly on lines 61-62. This crashes for any
    PyTorch or Keras model. Both calls are now routed through get_proba(),
    imported from merge_and_evaluate.
  - M2N2Pipeline.run() previously called .predict_proba() on lines 143-144
    for the per-pair parent AUC computation. Same fix applied.
  - get_proba is imported from merge_and_evaluate so the NN handling logic
    lives in exactly one place.
 
Everything else (CMA-ES config, dummy variable, result schema) is unchanged.
"""
 
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import cma
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
 
from scripts.select_top_pairs import select_top_pairs
from scripts.merge_and_evaluate import get_proba
from train_fraud_models import FraudMLP
from train_churn_models import ChurnMLP

TOP_N = 100
 
# ── Output path ───────────────────────────────────────────────────────────────
BASE = r"C:\Users\User\Desktop\ICTer\WordTemplate-1"

# ── CMA-ES optimiser ──────────────────────────────────────────────────────────
class CMAESMerger:
    """
    Optimise the blend ratio r in [0,1] via CMA-ES.
 
    Objective: maximise AUC(r * P_a + (1-r) * P_b, y_val)
    CMA-ES minimises, so we minimise negative AUC.
 
    get_proba() is used instead of .predict_proba() so this class works
    with RF, ET, PyTorch, and Keras models without any conditional branching
    at the call site.
    """
 
    def __init__(self, sigma0: float = 0.3, max_iter: int = 50, popsize: int = 10):
        self.sigma0   = sigma0
        self.max_iter = max_iter
        self.popsize  = popsize
 
    def optimise(
        self,
        model_a,
        model_b,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Returns:
            best_ratio    : optimal blend weight for model_a (float in [0,1])
            best_auc      : AUC at optimal ratio
            n_evaluations : total fitness evaluations used
        """
        # get_proba handles RF, PyTorch, Keras — replaces .predict_proba()
        pa = get_proba(model_a, X_val)
        pb = get_proba(model_b, X_val)
 
        def neg_auc(x):
            r = np.clip(x[0], 0.0, 1.0)
            return -roc_auc_score(y_val, r * pa + (1 - r) * pb)
 
        opts = cma.CMAOptions()
        opts["maxiter"] = self.max_iter
        opts["popsize"] = self.popsize
        opts["bounds"]  = [[0.0, 0.0], [1.0, 1.0]]  # bounds for [ratio, dummy]
        opts["verbose"] = -9
        opts["seed"]    = 42
 
        # x[0] = blend ratio, x[1] = dummy (CMA-ES requires dim >= 2)
        es      = cma.CMAEvolutionStrategy([0.5, 0.5], self.sigma0, opts)
        n_evals = 0
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [neg_auc(s) for s in solutions])
            n_evals  += len(solutions)
 
        r = es.result
        return {
            "best_ratio":    float(np.clip(r.xbest[0], 0.0, 1.0)),
            "best_auc":      float(-r.fbest),
            "n_evaluations": int(n_evals),
        }
 
 
# ── Full M2N2 pipeline ────────────────────────────────────────────────────────
class M2N2Pipeline:
    """
    Runs CMA-ES on the top-N ECCM-ranked pairs and records results.
    Supports RF, ET, PyTorch, and Keras models via get_proba().
    """
 
    def __init__(
        self,
        models_dir: str,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        output_dir: str,
        sigma0:     float = 0.3,
        max_iter:   int   = 30,
        popsize:    int   = 8,
    ):
        self.models_dir = models_dir
        self.X_val      = X_val
        self.y_val      = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.merger     = CMAESMerger(sigma0=sigma0, max_iter=max_iter, popsize=popsize)
        self._cache: dict = {}
 
    def _load(self, model_id: str):
        if model_id not in self._cache:
            self._cache[model_id] = joblib.load(
                Path(self.models_dir) / f"{model_id}.pkl"
            )
        return self._cache[model_id]
 
    def run(
        self,
        top_pairs:       pd.DataFrame,
        fixed_csv:       str,
        output_filename = f"m2n2_results_topN{TOP_N}.csv",
    ) -> pd.DataFrame:
        """
        Args:
            top_pairs:       [model_a, model_b, eccm] from select_top_pairs()
            fixed_csv:       path to merge_results_new_eccm.csv
            output_filename: output CSV name
 
        Returns:
            DataFrame saved to output_dir / output_filename
        """
        fixed = pd.read_csv(fixed_csv)
        rows  = []
 
        for i, pair in top_pairs.iterrows():
            mid_a, mid_b = pair["model_a"], pair["model_b"]
            ma = self._load(mid_a)
            mb = self._load(mid_b)
 
            opt = self.merger.optimise(ma, mb, self.X_val, self.y_val)
 
            mask       = (fixed["model_a"] == mid_a) & (fixed["model_b"] == mid_b)
            best_fixed = float(fixed.loc[mask, "auc_merged"].max())
 
            # get_proba() replaces .predict_proba() — works for all model types
            auc_a    = roc_auc_score(self.y_val, get_proba(ma, self.X_val))
            auc_b    = roc_auc_score(self.y_val, get_proba(mb, self.X_val))
            best_par = max(auc_a, auc_b)
 
            rows.append({
                "model_a":           mid_a,
                "model_b":           mid_b,
                "eccm":              pair["eccm"],
                "auc_a":             round(auc_a,          10),
                "auc_b":             round(auc_b,          10),
                "best_parent_auc":   round(best_par,       10),
                "fixed_best_auc":    round(best_fixed,     10),
                "fixed_improvement": round(best_fixed - best_par, 10),
                "opt_best_ratio":    opt["best_ratio"],
                "opt_best_auc":      round(opt["best_auc"], 10),
                "opt_improvement":   round(opt["best_auc"] - best_par, 10),
                "opt_vs_fixed":      round(opt["best_auc"] - best_fixed, 10),
                "opt_n_evals":       opt["n_evaluations"],
                "timestamp":         datetime.now().isoformat(),
            })
 
            delta = rows[-1]["opt_vs_fixed"]
            print(
                f"  {i+1:3d}. {mid_a} + {mid_b}  |  "
                f"Fixed: {best_fixed:.6f}  |  "
                f"CMA-ES: {opt['best_auc']:.6f} (r={opt['best_ratio']:.4f})  |  "
                f"Δ={delta:+.6f}"
            )
 
        df       = pd.DataFrame(rows)
        out_path = Path(self.output_dir) / output_filename
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} results → {out_path}")
        return df
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
 
    # Fraud
    fraud_df = pd.read_csv("./data/fraud_preprocessed.csv")
    X_f = fraud_df.drop("Class", axis=1).values
    y_f = fraud_df["Class"].values
    _, X_val_f, _, y_val_f = train_test_split(
        X_f, y_f, test_size=0.2, random_state=0, stratify=y_f
    )
    top_f = select_top_pairs(
        f"{BASE}/results/merges/fraud/merge_results_new_eccm.csv", TOP_N
    )
    M2N2Pipeline(
        f"{BASE}/models/fraud",
        X_val_f,
        y_val_f,
        f"{BASE}/results/merges/fraud"
    ).run(top_f, f"{BASE}/results/merges/fraud/merge_results_new_eccm.csv")
 
    # Churn
    churn_val = pd.read_csv("./data/churn_val_with_churn_col.csv")
    X_val_c   = churn_val.drop("Churn", axis=1).values
    y_val_c   = churn_val["Churn"].values
    top_c = select_top_pairs(
        f"{BASE}/results/merges/churn/merge_results_new_eccm.csv", TOP_N
    )
    M2N2Pipeline(
        f"{BASE}/models/churn",
        X_val_c,
        y_val_c,
        f"{BASE}/results/merges/churn"
    ).run(top_c, f"{BASE}/results/merges/churn/merge_results_new_eccm.csv")