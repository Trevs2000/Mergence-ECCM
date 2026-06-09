"""
merge_and_evaluate.py
 
Fixed-ratio merge experiment for fraud and churn tasks.
 
Changes from previous version (NN extension):
  - Added get_proba() helper that resolves P(class=1) from any supported
    model type: sklearn (predict_proba), PyTorch (nn.Module), Keras/TF.
    This replaces the two direct .predict_proba() calls in MergePipeline.run().
  - MergePipeline.run() now calls get_proba(ma, X_val) and get_proba(mb, X_val)
    instead of ma.predict_proba(X_val)[:, 1] and mb.predict_proba(X_val)[:, 1].
  - evaluate_baselines() still uses .predict_proba() because benchmark models
    are always RF (v100-v104) — no change needed there.
  - epc_ood column added to the output CSV. When True, the EPC prediction for
    that pair was made against RF training history despite one or both models
    being a NN. This is purely additive — existing downstream scripts that read
    this CSV (benchmarks.py, eccm_ablation.py, select_top_pairs.py) ignore
    unknown columns and are unaffected.
 
Everything else is unchanged.
"""
 
import importlib
import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from metrics.eccm import ECCMCalculator
 
# ── Output path ───────────────────────────────────────────────────────────────
BASE = r"C:\Users\User\Desktop\ICTer\WordTemplate-1"
 
# ── Constants ─────────────────────────────────────────────────────────────────
BLEND_RATIOS        = [0.3, 0.4, 0.5, 0.6, 0.7]
RF_VARIANT_RANGE    = range(0, 24)      # v000–v023  RF merge candidates
ET_VARIANT_RANGE    = range(200, 212)   # v200–v211  ExtraTrees merge candidates
NN_VARIANT_RANGE = range(300, 312)
BENCH_VARIANT_RANGE = range(100, 105)   # v100–v104  evaluation-only
 

# ── NN Model Definitions (required for unpickling) ───────────────────────────
class FraudMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ChurnMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── Universal probability extractor ──────────────────────────────────────────
def get_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Return P(class=1) as a 1-D np.ndarray for any supported model type.
 
    Detection order:
      1. sklearn / any model with predict_proba  (RF, ET, LogReg, …)
      2. PyTorch nn.Module
      3. Keras / TensorFlow Model
 
    This is the single place in the codebase that handles the NN prediction
    interface. All callers in this file use this function — .predict_proba()
    is never called directly on an unknown model type.
 
    Args:
        model: Any trained model object.
        X:     Feature array, shape (n_samples, n_features), float32/64.
 
    Returns:
        1-D np.ndarray of float32 in [0, 1], length n_samples.
 
    Raises:
        ValueError if no supported interface is found.
    """
    # ── 1. sklearn interface (RF, ET, LogReg, SVM with probability=True, …) ──
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary: return P(class=1); multiclass: return max class probability
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba.flatten().astype(np.float32)
 
    # ── 2. PyTorch nn.Module ──────────────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn

        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                X_t  = torch.tensor(X.astype(np.float32), dtype=torch.float32)
                out  = model(X_t)
                if isinstance(out, tuple):
                    out = out[0]
                out_np = out.cpu().float().numpy()

            if out_np.ndim == 1:
                out_np = out_np.reshape(-1, 1)

            n_out = out_np.shape[1]
            if n_out == 1:
                # Single logit output — apply sigmoid
                return (1.0 / (1.0 + np.exp(-np.clip(out_np[:, 0], -30, 30)))).astype(np.float32)
            else:
                # Multi-logit — softmax → P(class=1)
                e = np.exp(out_np - out_np.max(axis=1, keepdims=True))
                proba = e / e.sum(axis=1, keepdims=True)
                return proba[:, 1].astype(np.float32)
    except ImportError:
        pass
 
    # ── 3. Keras / TensorFlow Model ───────────────────────────────────────────
    for _keras_mod in ("keras", "tensorflow.keras"):
        try:
            keras_model = importlib.import_module(_keras_mod).Model
            if isinstance(model, keras_model):
                out_np = model.predict(X, verbose=0)
                if out_np.ndim == 1 or out_np.shape[1] == 1:
                    return out_np.flatten().astype(np.float32)
                return out_np[:, 1].astype(np.float32)
        except ImportError:
            continue
 
    raise ValueError(
        f"get_proba: cannot extract probabilities from {type(model).__name__}. "
        "Supported: sklearn predict_proba, PyTorch nn.Module, Keras/TF Model."
    )
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def load_models_by_range(models_dir: str, variant_range) -> dict:
    """Load model pkl files whose numeric suffix falls within variant_range."""
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
 
 
def load_all_merge_candidates(models_dir: str) -> dict:
    """Load RF (v000–v023) and ExtraTrees (v200–v211) together."""
    rf_models = load_models_by_range(models_dir, RF_VARIANT_RANGE)
    et_models = load_models_by_range(models_dir, ET_VARIANT_RANGE)
    nn_models = load_models_by_range(models_dir, NN_VARIANT_RANGE)
    all_models = {**rf_models, **et_models, **nn_models}
        print(
        f"  Loaded {len(rf_models)} RF + {len(et_models)} ET + {len(nn_models)} MLP"
        f" = {len(all_models)} total candidates"
        f" → C({len(all_models)},2) = {len(all_models)*(len(all_models)-1)//2} pairs"
    )
    return all_models
 
 
# ── Core pipeline ─────────────────────────────────────────────────────────────
class MergePipeline:
    """
    Runs the fixed-ratio merge experiment for a single task.
 
    Supports RF, ExtraTrees, PyTorch, and Keras models as merge candidates.
    All probability extraction goes through get_proba() — never .predict_proba()
    directly — so the pipeline is model-type agnostic.
    """
 
    def __init__(
        self,
        models_dir: str,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        task:       str = "unknown",
        output_dir: str = f"{BASE}\\results\\merges",
    ):
        self.models_dir = models_dir
        self.X_val      = X_val
        self.y_val      = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.eccm_calc  = ECCMCalculator(task=task)
 
    def run(self, num_pairs: Optional[int] = None) -> pd.DataFrame:
        """
        Run all merge experiments.
 
        Returns DataFrame with one row per (pair × ratio).
        The epc_ood column is True when EPC was queried against RF history
        for a NN pair — downstream scripts can filter or flag these rows.
        """
        models = load_all_merge_candidates(self.models_dir)
        all_pairs = list(combinations(sorted(models.keys()), 2))
        # Only slice when an explicit cap is given — None = run all 1,128 pairs
        pairs = all_pairs if num_pairs is None else all_pairs[:num_pairs]
 
        print(f"\n{len(pairs)} pairs × {len(BLEND_RATIOS)} ratios = "
              f"{len(pairs) * len(BLEND_RATIOS)} experiments\n")
 
        rows = []
        for i, (mid_a, mid_b) in enumerate(pairs, 1):
            ma, mb = models[mid_a], models[mid_b]
 
            # ECCM sub-metrics computed once per pair (invariant across ratios)
            eccm = self.eccm_calc.compute(ma, mb, X=self.X_val)
 
            # Use get_proba() — works for RF, ET, PyTorch, Keras
            pa    = get_proba(ma, self.X_val)
            pb    = get_proba(mb, self.X_val)
            auc_a = roc_auc_score(self.y_val, pa)
            auc_b = roc_auc_score(self.y_val, pb)
            best  = max(auc_a, auc_b)
 
            for ratio in BLEND_RATIOS:
                auc_m = roc_auc_score(self.y_val, ratio * pa + (1 - ratio) * pb)
                impr  = auc_m - best
                rows.append({
                    "model_a":     mid_a,
                    "model_b":     mid_b,
                    "auc_a":       round(auc_a,  10),
                    "auc_b":       round(auc_b,  10),
                    "auc_merged":  round(auc_m,  10),
                    "improvement": round(impr,   10),
                    "success":     int(impr > 0),
                    "psc":         eccm["psc"],
                    "fsc":         eccm["fsc"],
                    "rsc":         eccm["rsc"],
                    "eccm":        eccm["eccm"],
                    "epc_ood":     eccm["epc_ood"],   # NEW: OOD flag from Issue #6
                    "blend_ratio": ratio,
                    "timestamp":   datetime.now().isoformat(),
                })
 
            print(f"  {i:3d}. {mid_a} + {mid_b}"
                  + (" [EPC OOD]" if eccm["epc_ood"] else ""))
 
        df       = pd.DataFrame(rows)
        csv_path = f"{self.output_dir}/merge_results_new_eccm.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows → {csv_path}")
        return df
 
 
# ── Baseline evaluation ───────────────────────────────────────────────────────
def evaluate_baselines(
    models_dir: str,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    task:       str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Evaluate benchmark RF models (v100–v104).
    These are always RF — predict_proba is safe here.
    """
    bench_models = load_models_by_range(models_dir, BENCH_VARIANT_RANGE)
    if not bench_models:
        print("  No benchmark models found (expected v100–v104).")
        return pd.DataFrame(columns=["model_id", "model_type", "auc", "task"])
 
    rows = []
    for mid, model in sorted(bench_models.items()):
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"model_id": mid, "model_type": "benchmark_rf",
                     "auc": round(auc, 10), "task": task})
        print(f"  Benchmark {mid}: AUC={auc:.6f}")
 
    df       = pd.DataFrame(rows)
    csv_path = f"{output_dir}/baseline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} baseline rows → {csv_path}")
    return df
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
 
    # ── Fraud ──────────────────────────────────────────────────────────────────
    fraud_df = pd.read_csv("./data/fraud_preprocessed.csv")
    X_f = fraud_df.drop("Class", axis=1).values
    y_f = fraud_df["Class"].values
    _, X_val_f, _, y_val_f = train_test_split(
        X_f, y_f, test_size=0.2, random_state=0, stratify=y_f
    )
 
    MergePipeline(
        models_dir=f"{BASE}\\models\\fraud",
        X_val=X_val_f,
        y_val=y_val_f,
        task="fraud",
        output_dir=f"{BASE}\\results\\merges\\fraud",
    ).run()
 
    evaluate_baselines(
        models_dir=f"{BASE}\\models\\fraud",
        X_val=X_val_f,
        y_val=y_val_f,
        task="fraud",
        output_dir=f"{BASE}\\results\\merges\\fraud",
    )
 
    # ── Churn ──────────────────────────────────────────────────────────────────
    churn_val = pd.read_csv("./data/churn_val_with_churn_col.csv")
    X_val_c   = churn_val.drop("Churn", axis=1).values
    y_val_c   = churn_val["Churn"].values
 
    MergePipeline(
        models_dir=f"{BASE}\\models\\churn",
        X_val=X_val_c,
        y_val=y_val_c,
        task="churn",
        output_dir=f"{BASE}\\results\\merges\\churn",
    ).run()
 
    evaluate_baselines(
        models_dir=f"{BASE}\\models\\churn",
        X_val=X_val_c,
        y_val=y_val_c,
        task="churn",
        output_dir=f"{BASE}\\results\\merges\\churn",
    )