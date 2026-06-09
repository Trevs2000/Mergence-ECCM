"""
eccm.py - Evolutionary Compatibility & Co-evolution Metric (ECCM)
 
ECCM = w_psc × PSC + w_fsc × FSC + w_rsc × RSC + w_epc × EPC
 
Fixes applied:
 
  Issue #4 — either_nn only checked model_a:
    The previous line was:
        either_nn = _is_nn_model(model_a)
    If model_a is RF and model_b is NN, either_nn=False. The RF synthetic
    fallback runs (which works for model_a), but the NN branch warning is
    never raised, and the data_mode is silently reported as 'synthetic'
    rather than flagging that a NN is present without real calibration data.
    Fixed to:
        either_nn = _is_nn_model(model_a) or _is_nn_model(model_b)
 
  Issue #6 — no OOD flag when EPC is queried with NN scores against RF history:
    EPC was trained exclusively on RF/ET merge records. When called for a NN
    pair, the k-NN lookup runs against RF history and returns a prediction
    with no indication that it is out-of-distribution. The returned dict now
    includes an 'epc_ood' key (bool). True = at least one model is a NN and
    EPC was trained on RF/ET data — the prediction should be treated as
    unreliable. The Streamlit UI can use this flag to show a warning badge.
 
    This key is additive — existing callers that do not read 'epc_ood'
    are unaffected.
 
All other logic (weights, tier thresholds, fallback chain, impute_fsc) is
unchanged from the previous version.
"""
 
import warnings
import numpy as np
from typing import Optional
 
from .psc import PSCCalculator
from .fsc import FSCCalculator
from .rsc import RSCCalculator
from .epc import EPCTrainer
 
 
# ── Task-specific weights ─────────────────────────────────────────────────────
TASK_WEIGHTS = {
    "fraud":   {"w_psc": 0.333, "w_fsc": 0.333, "w_rsc": 0.333, "w_epc": 0.0},
    "churn":   {"w_psc": 0.333, "w_fsc": 0.333, "w_rsc": 0.333, "w_epc": 0.0},
    "unknown": {"w_psc": 0.333, "w_fsc": 0.333, "w_rsc": 0.333, "w_epc": 0.0},
}
 
# ── Tier thresholds ───────────────────────────────────────────────────────────
TIER_THRESHOLDS = {
    "fraud":   {"low_to_med": 0.840, "med_to_high": 0.935},
    "churn":   {"low_to_med": 0.960, "med_to_high": 0.988},
    "unknown": {"low_to_med": 0.650, "med_to_high": 0.850},
}
 
 
def get_tier(eccm_score: float, task: str = "unknown") -> tuple[str, str, str]:
    t = TIER_THRESHOLDS.get(task, TIER_THRESHOLDS["unknown"])
    if eccm_score >= t["med_to_high"]:
        return "High Compatibility",   "#28a745", "✅"
    elif eccm_score >= t["low_to_med"]:
        return "Medium Compatibility", "#ffc107", "⚠️"
    else:
        return "Low Compatibility",    "#dc3545", "❌"
 
 
def get_success_probability(eccm_score: float, task: str = "unknown") -> float:
    t    = TIER_THRESHOLDS.get(task, TIER_THRESHOLDS["unknown"])
    low  = t["low_to_med"]
    high = t["med_to_high"]
    if eccm_score >= high:
        return float(np.clip(
            0.80 + 0.20 * (eccm_score - high) / max(1.0 - high, 1e-6),
            0.80, 1.0,
        ))
    elif eccm_score >= low:
        return float(np.clip(
            0.40 + 0.40 * (eccm_score - low) / max(high - low, 1e-6),
            0.40, 0.80,
        ))
    return float(np.clip(0.40 * eccm_score / max(low, 1e-6), 0.0, 0.40))
 
 
# ── Synthetic data fallback for RF/ET only ────────────────────────────────────
 
def synthetic_validation_from_rf(model, n_samples: int = 500) -> np.ndarray:
    """
    Generate synthetic feature array from a RandomForest's split thresholds.
    Only called for RF/ET models — NNs have no estimators_ or tree_ attributes.
    Unchanged from original.
    """
    n_features = model.n_features_in_
    feat_min   = np.full(n_features,  np.inf)
    feat_max   = np.full(n_features, -np.inf)
 
    for tree in model.estimators_:
        t = tree.tree_
        for fi, th in zip(t.feature, t.threshold):
            if fi < 0:
                continue
            feat_min[fi] = min(feat_min[fi], th)
            feat_max[fi] = max(feat_max[fi], th)
 
    no_splits           = feat_min == np.inf
    feat_min[no_splits] = 0.0
    feat_max[no_splits] = 1.0
    margin              = (feat_max - feat_min) * 0.05
    feat_min           -= margin
    feat_max           += margin
 
    return np.random.default_rng(42).uniform(
        feat_min, feat_max, size=(n_samples, n_features)
    ).astype(np.float32)
 
 
# ── Model-type helpers ────────────────────────────────────────────────────────
 
def _is_tree_model(model) -> bool:
    """True for sklearn tree-based estimators (RF, ET, GBM, …)."""
    return hasattr(model, "estimators_") and hasattr(model, "feature_importances_")
 
 
def _is_nn_model(model) -> bool:
    """True for PyTorch nn.Module or Keras/TF Model."""
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return True
    except ImportError:
        pass
    try:
        import keras
        if isinstance(model, keras.Model):
            return True
    except ImportError:
        pass
    try:
        from tensorflow import keras as tf_keras
        if isinstance(model, tf_keras.Model):
            return True
    except ImportError:
        pass
    return False
 
 
# ── Main ECCM calculator ──────────────────────────────────────────────────────
 
class ECCMCalculator:
 
    def __init__(self, task: str = "unknown", epc_path: str = None):
        self.task  = task
        w          = TASK_WEIGHTS.get(task, TASK_WEIGHTS["unknown"])
        self.w_psc = w["w_psc"]
        self.w_fsc = w["w_fsc"]
        self.w_rsc = w["w_rsc"]
        self.w_epc = w["w_epc"]
        self.psc_calc = PSCCalculator(method="cosine")
        self.fsc_calc = FSCCalculator(strategy="correlation")
        self.rsc_calc = RSCCalculator()

        # Load saved EPC (with accumulated history) if path provided.
        # Falls back to empty EPCTrainer if path doesn't exist yet
        # this handles the very first run before any history exists.
        self.epc = EPCTrainer()
        if epc_path:
            try:
                self.epc.load(epc_path)
            except FileNotFoundError:
                pass  # first run, no history yet so epc stays empty
 
    # ── Data resolution — Issues #4 fix ──────────────────────────────────────
 
    def _resolve_data(
        self,
        model_a,
        model_b,
        X: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], str]:
        """
        Resolve the calibration data for FSC and RSC.
 
        Issue #4 fix: checks BOTH models for NN type (was model_a only).
        If model_a=RF and model_b=NN, either_nn is now correctly True,
        the NN warning is raised, and data_mode is 'pscrsc_only' when no
        real X is provided — not 'synthetic', which would have been
        misleading.
 
        RF/ET fallback chain (only when neither model is a NN):
          1. User-supplied X
          2. model_a.X_train_sample_  (embedded during training)
          3. Synthetic data from tree split thresholds
          4. None → FSC will be imputed from k-NN history
 
        NN path (when either model is a NN):
          1. User-supplied X
          2. model_a.X_train_sample_  (if manually embedded — optional)
          3. None → FSC imputed; CKA not possible without real data
        """
        # Issue #4: check both models
        either_nn = _is_nn_model(model_a) or _is_nn_model(model_b)
 
        # Step 1: user-supplied data (works for all model types)
        if X is not None and len(X) > 0:
            return X, "full"
 
        # Step 2: embedded training sample (works for both RF and NN if set)
        X_embedded = getattr(model_a, "X_train_sample_", None)
        if X_embedded is not None and len(X_embedded) > 0:
            return X_embedded, "embedded"
 
        # ── NN path: no further fallbacks ────────────────────────────────────
        if either_nn:
            warnings.warn(
                "ECCMCalculator: calibration data X was not supplied and no "
                "embedded X_train_sample_ was found. At least one model is a "
                "neural network — FSC and RSC (CKA) require real data. "
                "FSC will be imputed from k-NN history; RSC will return 0.5. "
                "Pass X to compute() or embed X_train_sample_ in the model "
                "at training time for accurate scores.",
                UserWarning, stacklevel=3,
            )
            return None, "pscrsc_only"
 
        # ── RF/ET path: synthetic fallback ───────────────────────────────────
        try:
            X_synth = synthetic_validation_from_rf(model_a)
            return X_synth, "synthetic"
        except Exception:
            pass
 
        return None, "pscrsc_only"
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def compute(
        self,
        model_a,
        model_b,
        X:        Optional[np.ndarray] = None,
        epc_pred: float                = 0.0,
    ) -> dict:
        """
        Compute ECCM and all sub-metrics.
 
        Args:
            model_a, model_b: Any supported model (RF, ET, PyTorch, Keras).
            X:        Calibration data (n_samples, n_features) as np.ndarray.
                      Required for NN models; optional for RF/ET.
            epc_pred: Pre-computed EPC value. Skips k-NN lookup if non-zero.
 
        Returns dict with keys:
            psc, fsc, rsc, epc, eccm,
            data_mode, epc_reliability, epc_neighbours,
            tier, tier_colour, tier_emoji, p_success, weights,
            epc_ood   ← NEW (Issue #6): True when EPC is queried with NN
                        scores against RF/ET training history.
        """
        X_data, data_mode = self._resolve_data(model_a, model_b, X)
 
        # ── Sub-metrics ───────────────────────────────────────────────────────
        psc = self.psc_calc.compute(model_a, model_b)
        rsc = self.rsc_calc.compute(model_a, model_b, X_data)
        fsc = (
            self.fsc_calc.compute(model_a, model_b, X_data)
            if X_data is not None
            else self._impute_fsc(psc, rsc)
        )
 
        # ── EPC (k-NN contextual) ─────────────────────────────────────────────
        epc_reliability, epc_neighbours = 0.5, []
        if epc_pred == 0.0:
            try:
                epc_pred, epc_reliability, epc_neighbours = (
                    self.epc.predict_with_context(psc, fsc, rsc)
                )
            except Exception:
                epc_pred = 0.0
 
        # ── Issue #6: OOD flag for EPC ────────────────────────────────────────
        # EPC was trained on RF/ET merge records. If either model in this pair
        # is a NN, the k-NN lookup is querying with scores that may be
        # systematically different from the RF/ET training distribution.
        # Surface this as an explicit flag so the UI can warn the user.
        epc_ood = _is_nn_model(model_a) or _is_nn_model(model_b)
 
        # ── Composite score ───────────────────────────────────────────────────
        eccm = float(np.clip(
            self.w_psc * psc
            + self.w_fsc * fsc
            + self.w_rsc * rsc
            + self.w_epc * epc_pred,
            0.0, 1.0,
        ))
 
        tier, tier_colour, tier_emoji = get_tier(eccm, self.task)
 
        return {
            "psc":             float(psc),
            "fsc":             float(fsc),
            "rsc":             float(rsc),
            "epc":             float(epc_pred),
            "eccm":            eccm,
            "data_mode":       data_mode,
            "epc_reliability": epc_reliability,
            "epc_neighbours":  epc_neighbours,
            "tier":            tier,
            "tier_colour":     tier_colour,
            "tier_emoji":      tier_emoji,
            "p_success":       get_success_probability(eccm, self.task),
            "weights": {
                "w_psc": self.w_psc, "w_fsc": self.w_fsc,
                "w_rsc": self.w_rsc, "w_epc": self.w_epc,
            },
            # Issue #6: True = EPC prediction is OOD (NN pair, RF training data)
            "epc_ood": epc_ood,
        }
 
    def _impute_fsc(self, psc: float, rsc: float) -> float:
        """Last-resort FSC imputation from k-NN history. Unchanged."""
        h = self.epc._history
        if h is None or len(h) == 0:
            return 0.5
        dists   = np.linalg.norm(h[:, [0, 2]] - np.array([psc, rsc]), axis=1)
        k       = min(self.epc.k, len(dists))
        nn_idx  = np.argsort(dists)[:k]
        weights = 1.0 / (dists[nn_idx] + self.epc.epsilon)
        return float(np.average(h[nn_idx, 1], weights=weights))