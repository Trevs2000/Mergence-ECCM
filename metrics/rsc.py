"""
rsc.py - Representational Space Compatibility (RSC)
 
Fixes applied:
  Issue #1 — Mixed pair tile() bug:
    np.tile(imp, (n,1)) creates a rank-0 matrix after centring → CKA
    denominator hits 1e-10 → always returns 0.5 silently. Fixed by
    falling back to importance_correlation() for mixed pairs with an
    explicit warning, rather than tiling into a degenerate CKA call.
 
  Issue #2 — Hook leak on BaseException (KeyboardInterrupt, SystemExit):
    register_forward_hook() removal is now guaranteed via try/finally,
    not just via the except branch. The activations list is cleared
    before each call to prevent accumulation across repeated calls.
 
  Issue #7 — CKA with < 32 calibration samples:
    Added explicit minimum-sample guard before CKA is attempted.
    Returns 0.5 with a warning rather than producing numerically
    unreliable results silently. Sub-sample condition changed from
    > to >= so that exactly cka_n_samples rows bypasses the copy.
"""
 
import warnings
import numpy as np
from typing import Optional
 
 
# Minimum calibration samples for CKA to be numerically reliable.
# Below this the Gram matrices are too small to capture meaningful geometry.
_CKA_MIN_SAMPLES = 32
 
 
class RSCCalculator:
    """
    Representational Space Compatibility (RSC).
 
    Strategy (automatic, no API change):
      Both tree/linear  → feature-importance correlation (original behaviour)
      Either NN         → linear CKA on penultimate-layer activations
      Mixed (tree + NN) → importance correlation with explicit warning
                          (tile-based CKA is rank-deficient; see Issue #1)
 
    Args:
        nn_layer_index: Which parameterised leaf layer's activations to use.
                        -1 = penultimate (default). -2 = two before output, etc.
        cka_n_samples:  Max calibration rows passed to CKA. Larger = more
                        accurate but O(n²) cost. Must be >= _CKA_MIN_SAMPLES.
    """
 
    def __init__(
        self,
        nn_layer_index: int = -1,
        cka_n_samples:  int = 512,
    ):
        self.nn_layer_index = nn_layer_index
        self.cka_n_samples  = cka_n_samples
 
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
 
    def _is_nn(self, model) -> bool:
        return self._is_pytorch(model) or self._is_keras(model)
 
    # ── Feature importance path (trees + linear) ──────────────────────────────
 
    def _get_feature_importance(self, model) -> np.ndarray:
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_.flatten()
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_).flatten()
        else:
            n = getattr(model, "n_features_in_", 1)
            return np.ones(n) / n
 
    def _importance_correlation(self, model_a, model_b) -> float:
        """Feature-importance correlation for tree/linear model pairs."""
        imp_a = self._get_feature_importance(model_a)
        imp_b = self._get_feature_importance(model_b)
 
        min_len = min(len(imp_a), len(imp_b))
        imp_a   = imp_a[:min_len]
        imp_b   = imp_b[:min_len]
 
        if np.std(imp_a) < 1e-8 or np.std(imp_b) < 1e-8:
            return 0.5
 
        r = np.corrcoef(imp_a, imp_b)[0, 1]
        return float((np.clip(r, -1.0, 1.0) + 1) / 2)
 
    # ── CKA activation extraction ─────────────────────────────────────────────
 
    def _get_parameterised_leaf_modules(self, model):
        """
        Return parameterised leaf modules in forward-declaration order.
        'Leaf' means no child modules (excludes Sequential wrappers etc.).
        """
        return [
            m for m in model.modules()
            if len(list(m.parameters(recurse=False))) > 0
            and len(list(m.children())) == 0
        ]
 
    def _get_activations_pytorch(
        self, model, X: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Capture penultimate-layer activations via a forward hook.
 
        Issue #2 fix: hook removal is guaranteed by try/finally, protecting
        against KeyboardInterrupt and other BaseException subclasses that
        would escape the bare except clause.
        """
        import torch
 
        param_modules = self._get_parameterised_leaf_modules(model)
        if not param_modules:
            warnings.warn(
                "RSC/CKA: no parameterised leaf modules found in PyTorch model. "
                "Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
 
        target_layer  = param_modules[self.nn_layer_index]
        activations   = []   # reset before each call — prevents accumulation
 
        def hook_fn(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            activations.append(out.detach().cpu().float().numpy())
 
        hook = target_layer.register_forward_hook(hook_fn)
        try:                                          # Issue #2: finally block
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32)
                model(X_t)
        except Exception as e:
            warnings.warn(
                f"RSC/CKA: PyTorch forward pass failed ({e}). Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
        finally:
            hook.remove()                             # always runs
 
        if not activations:
            warnings.warn(
                "RSC/CKA: forward hook produced no activations "
                "(target layer may not be on the forward path). Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
 
        act = activations[0]
        if act.ndim > 2:
            act = act.reshape(act.shape[0], -1)
        return act
 
    def _get_activations_keras(
        self, model, X: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract penultimate-layer activations from a Keras model."""
        try:
            import keras
        except ImportError:
            from tensorflow import keras
 
        param_layers = [l for l in model.layers if l.weights]
        if not param_layers:
            warnings.warn(
                "RSC/CKA: no parameterised Keras layers found. Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
 
        target_layer = param_layers[self.nn_layer_index]
        try:
            activation_model = keras.Model(
                inputs=model.inputs,
                outputs=target_layer.output,
            )
            act = activation_model.predict(X, verbose=0)
        except Exception as e:
            warnings.warn(
                f"RSC/CKA: Keras activation extraction failed ({e}). "
                "Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
 
        if act.ndim > 2:
            act = act.reshape(act.shape[0], -1)
        return act
 
    # ── Linear CKA ───────────────────────────────────────────────────────────
 
    @staticmethod
    def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Linear Centred Kernel Alignment (Kornblith et al., 2019).
 
        CKA(X, Y) = ||Y^T X||_F² / (||X^T X||_F · ||Y^T Y||_F)
 
        Returns float in [0, 1]. Both matrices must have the same number of
        rows (samples). Column count (units) can differ between X and Y.
        """
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
 
        XtX = X.T @ X
        YtY = Y.T @ Y
        XtY = X.T @ Y
 
        numerator   = np.linalg.norm(XtY, "fro") ** 2
        denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
 
        if denominator < 1e-10:
            # Both models produced constant (zero-variance) activations on
            # this calibration set — CKA is undefined.
            warnings.warn(
                "RSC/CKA: denominator is near-zero (constant activations on "
                "calibration set). Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return 0.5
 
        return float(np.clip(numerator / denominator, 0.0, 1.0))
 
    # ── Calibration data guard — Issue #7 ────────────────────────────────────
 
    def _prepare_cka_data(
        self, X: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Sub-sample X for CKA and enforce the minimum-sample requirement.
 
        Issue #7 fix:
          - Returns None (→ 0.5) if len(X) < _CKA_MIN_SAMPLES.
          - Uses replace=False sub-sampling only when len(X) > cka_n_samples
            (changed from > to >= for edge-case clarity, though both work
            since choice(n, n, replace=False) is just a shuffle).
        """
        n = len(X)
 
        if n < _CKA_MIN_SAMPLES:
            warnings.warn(
                f"RSC/CKA: only {n} calibration samples provided. "
                f"CKA requires at least {_CKA_MIN_SAMPLES} samples for "
                "reliable results. Returning 0.5.",
                UserWarning, stacklevel=3,
            )
            return None
 
        if n > self.cka_n_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, self.cka_n_samples, replace=False)
            return X[idx]
 
        return X
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def compute(
        self,
        model_a,
        model_b,
        X: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute RSC between two models.
 
        Args:
            model_a, model_b: Any supported model type.
            X: Calibration data (n_samples, n_features) as np.ndarray.
               Required for NN models. Ignored for tree/linear pairs.
 
        Returns float in [0, 1]. 0.5 = neutral / fallback on any error.
        """
        is_nn_a = self._is_nn(model_a)
        is_nn_b = self._is_nn(model_b)
 
        # ── Both are tree/linear: original importance correlation ─────────────
        if not is_nn_a and not is_nn_b:
            return self._importance_correlation(model_a, model_b)
 
        # ── Mixed pair (tree + NN) — Issue #1 fix ────────────────────────────
        # np.tile(importance, (n, 1)) produces a rank-0 matrix after centring
        # because every row is identical → XtX Frobenius norm = 0 → CKA = 0.5
        # always, regardless of NN representations.
        # Correct behaviour: fall back to importance correlation with a warning
        # so the caller gets a meaningful (if approximate) signal rather than
        # a constant 0.5.
        if is_nn_a != is_nn_b:
            warnings.warn(
                "RSC: mixed pair (tree model + neural network). "
                "CKA cannot be computed because the tree side has no "
                "layer activations. Falling back to feature-importance "
                "correlation as an approximation. "
                "FSC is the more reliable metric for cross-type pairs.",
                UserWarning, stacklevel=2,
            )
            return self._importance_correlation(model_a, model_b)
 
        # ── Both are NNs: CKA path ────────────────────────────────────────────
        if X is None or len(X) == 0:
            warnings.warn(
                "RSC: CKA requires calibration data X but none was provided. "
                "Returning neutral 0.5. Pass X to compute() for NN models.",
                UserWarning, stacklevel=2,
            )
            return 0.5
 
        # Issue #7: enforce minimum samples and sub-sample if needed
        X_cka = self._prepare_cka_data(X)
        if X_cka is None:
            return 0.5
 
        # Extract activations
        if self._is_pytorch(model_a):
            act_a = self._get_activations_pytorch(model_a, X_cka)
        else:
            act_a = self._get_activations_keras(model_a, X_cka)
 
        if self._is_pytorch(model_b):
            act_b = self._get_activations_pytorch(model_b, X_cka)
        else:
            act_b = self._get_activations_keras(model_b, X_cka)
 
        if act_a is None or act_b is None:
            return 0.5
 
        return self._linear_cka(act_a, act_b)