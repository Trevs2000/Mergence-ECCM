"""
fsc.py - Functional Space Compatibility (FSC)
 
Fix applied (Issue #5):
  Double-sigmoid when model already ends with nn.Sigmoid():
    The previous version unconditionally applied sigmoid to single-output
    PyTorch models. If the model's final layer is nn.Sigmoid(), the output
    is already in [0,1] and a second sigmoid compresses everything toward
    0.5 (std drops from ~0.34 to ~0.08 in typical cases).
 
  The Perplexity-suggested heuristic (check flat.std() > 0.01) is fragile:
    - A well-calibrated model where all predictions are near 0.5 has
      std < 0.01 even with raw logits, causing the check to misfire.
    - The std threshold has no principled basis.
 
  Correct fix: explicit output_is_logits parameter (default True).
    - output_is_logits=True (default): sigmoid / softmax applied to output.
      Use this when your model outputs raw logits (nn.Linear as final layer).
    - output_is_logits=False: output is taken as-is.
      Use this when your model ends with nn.Sigmoid() or nn.Softmax().
 
  This is an explicit contract rather than a heuristic, which is the right
  design for a library function. The default (True) matches the most common
  PyTorch pattern (CrossEntropyLoss training → raw logit output).
 
  Backward compatibility: all existing RF/ET calls are unaffected because
  the output_is_logits parameter only applies to the PyTorch prediction path.
"""
 
import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from typing import Optional
 
 
class FSCCalculator:
    """
    Functional Space Compatibility (FSC).
 
    Args:
        strategy:        'correlation' (Pearson r of soft predictions, default)
                         or 'agreement' (hard-label match rate).
        output_is_logits: For PyTorch models only.
                          True  (default) = model outputs raw logits; sigmoid /
                                softmax will be applied automatically.
                          False = model already outputs probabilities (e.g. final
                                layer is nn.Sigmoid() or nn.Softmax()); no
                                additional activation is applied.
        output_index:    For multi-output NNs, which output column to use as
                         P(positive class). Default 1 (binary classifiers).
    """
 
    def __init__(
        self,
        strategy:         str  = "correlation",
        output_is_logits: bool = True,
        output_index:     int  = 1,
    ):
        self.strategy         = strategy
        self.output_is_logits = output_is_logits
        self.output_index     = output_index
 
    # ── Model-type detection ──────────────────────────────────────────────────
 
    @staticmethod
    def _is_pytorch(model) -> bool:
        try:
            import torch.nn as nn
            return isinstance(model, nn.Module)
        except ImportError:
            return False
 
    @staticmethod
    def _is_keras(model) -> bool:
        try:
            import keras
            return isinstance(model, keras.Model)
        except ImportError:
            pass
        try:
            from tensorflow import keras as tf_keras
            return isinstance(model, tf_keras.Model)
        except ImportError:
            pass
        return False
 
    # ── Prediction extraction ─────────────────────────────────────────────────
 
    def get_predictions(self, model, X: np.ndarray) -> np.ndarray:
        """
        Return a 1-D array of soft prediction scores in [0, 1].
        For classifiers: P(positive class).
        For regressors:  raw output values.
        """
        if self._is_pytorch(model):
            return self._predict_pytorch(model, X)
        if self._is_keras(model):
            return self._predict_keras(model, X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, 1] if proba.shape[1] == 2 else np.max(proba, axis=1)
            return proba
        return model.predict(X)
 
    def _predict_pytorch(self, model, X: np.ndarray) -> np.ndarray:
        """
        Run PyTorch model in eval mode and return soft predictions.
 
        Issue #5 fix: activation (sigmoid/softmax) is only applied when
        output_is_logits=True. When output_is_logits=False the raw output
        is returned directly, preventing double-sigmoid for models that
        already end with nn.Sigmoid() or nn.Softmax().
        """
        import torch
 
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
            out    = model(X_t)
            if isinstance(out, tuple):
                out = out[0]
            out_np = out.cpu().float().numpy()
 
        # Flatten scalar outputs
        if out_np.ndim == 1:
            out_np = out_np.reshape(-1, 1)
 
        n_out = out_np.shape[1]
 
        # ── output_is_logits=False: model already outputs probabilities ───────
        if not self.output_is_logits:
            if n_out == 1:
                return out_np[:, 0]
            return out_np[:, min(self.output_index, n_out - 1)]
 
        # ── output_is_logits=True (default): apply activation ─────────────────
        if n_out == 1:
            # Binary with single logit → sigmoid
            return self._sigmoid(out_np[:, 0])
        elif n_out == 2:
            # Binary with 2-logit output → softmax → P(class=1)
            proba = self._softmax(out_np)
            return proba[:, self.output_index]
        else:
            # Multiclass → softmax → max probability
            proba = self._softmax(out_np)
            return np.max(proba, axis=1)
 
    def _predict_keras(self, model, X: np.ndarray) -> np.ndarray:
        """Run Keras model and return soft predictions."""
        out_np = model.predict(X, verbose=0)
 
        if out_np.ndim == 1 or out_np.shape[1] == 1:
            return out_np.flatten()
        elif out_np.shape[1] == 2:
            return out_np[:, self.output_index]
        return np.max(out_np, axis=1)
 
    # ── Numeric helpers ───────────────────────────────────────────────────────
 
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
 
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
 
    # ── Similarity metrics ────────────────────────────────────────────────────
 
    def correlation_similarity(
        self, pred_a: np.ndarray, pred_b: np.ndarray
    ) -> float:
        """Pearson r of soft predictions, normalised to [0, 1]."""
        try:
            r, _ = pearsonr(pred_a, pred_b)
            if np.isnan(r):
                return 0.5
        except Exception:
            return 0.5
        return float((np.clip(r, -1.0, 1.0) + 1) / 2)
 
    def agreement_similarity(
        self,
        pred_a:    np.ndarray,
        pred_b:    np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """Hard-label agreement: fraction of samples classified identically."""
        binary_a = (pred_a > threshold).astype(int)
        binary_b = (pred_b > threshold).astype(int)
        return float(accuracy_score(binary_a, binary_b))
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def compute(self, model_a, model_b, X: np.ndarray) -> float:
        """
        Compute FSC between two models on data X.
 
        Returns float in [0, 1]:
          1.0 = identical predictions
          0.5 = neutral / fallback
          0.0 = opposite predictions
        """
        try:
            pred_a = self.get_predictions(model_a, X)
            pred_b = self.get_predictions(model_b, X)
        except Exception as e:
            warnings.warn(
                f"FSC: prediction extraction failed ({e}). Returning 0.5.",
                UserWarning, stacklevel=2,
            )
            return 0.5
 
        if self.strategy == "correlation":
            return self.correlation_similarity(pred_a, pred_b)
        return self.agreement_similarity(pred_a, pred_b)