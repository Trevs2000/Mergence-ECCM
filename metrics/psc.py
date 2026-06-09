"""
psc.py - Parameter Space Compatibility (PSC)
 
Fix applied (Issue #3):
  _align() now checks the size ratio between the two weight vectors.
  If the ratio exceeds _MAX_RELIABLE_RATIO (10x), PSC is unreliable because
  truncating a 5M-parameter model to 512 elements compares only the first-layer
  weights (architecturally determined by input size) against the full small model.
  In that case: warn loudly and return 0.5 from compute() rather than a
  silently misleading score.
 
All other logic is unchanged from the previous version.
"""
 
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Sequence
 
 
_PYTORCH_NON_PARAM_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)
 
_DEFAULT_MAX_PARAMS    = 5_000_000
_SAMPLE_SEED           = 42
_MAX_RELIABLE_RATIO    = 10   # beyond this, truncation-based PSC is meaningless
 
 
class PSCCalculator:
    """
    Parameter Space Compatibility (PSC).
 
    Args:
        method:      'cosine' (default) or 'euclidean'.
        layer_types: Optional list of PyTorch layer class name substrings to
                     include (e.g. ['Linear', 'Conv']). None = all layers.
        max_params:  Maximum weight-vector length before uniform sub-sampling.
                     Set to None to disable (not recommended for LLMs).
    """
 
    def __init__(
        self,
        method:      str                       = "cosine",
        layer_types: Optional[Sequence[str]]   = None,
        max_params:  Optional[int]             = _DEFAULT_MAX_PARAMS,
    ):
        if method not in ("cosine", "euclidean"):
            raise ValueError(f"Unknown method: {method!r}. Choose 'cosine' or 'euclidean'.")
        self.method      = method
        self.layer_types = layer_types
        self.max_params  = max_params
 
    # ── Weight extraction ─────────────────────────────────────────────────────
 
    def extract_weights(self, model) -> np.ndarray:
        """
        Extract a single flat weight vector from any supported model type.
        Returns 1-D np.ndarray of float32.
        Raises ValueError for unsupported model types.
        """
        weights: list[np.ndarray] = []
 
        if self._is_pytorch(model):
            weights = self._extract_pytorch(model)
        elif self._is_keras(model):
            weights = self._extract_keras(model)
        elif hasattr(model, "feature_importances_"):
            weights = [model.feature_importances_.flatten()]
        elif hasattr(model, "coef_"):
            weights = [model.coef_.flatten()]
            if hasattr(model, "intercept_"):
                weights.append(np.atleast_1d(model.intercept_).flatten())
 
        if not weights:
            raise ValueError(
                f"Cannot extract weights from {type(model).__name__}. "
                "Supported: PyTorch nn.Module, Keras Model, "
                "sklearn tree/linear estimators."
            )
 
        w = np.concatenate(weights).astype(np.float32)
        return self._maybe_subsample(w)
 
    # ── PyTorch helpers ───────────────────────────────────────────────────────
 
    @staticmethod
    def _is_pytorch(model) -> bool:
        try:
            import torch.nn as nn
            return isinstance(model, nn.Module)
        except ImportError:
            return False
 
    def _extract_pytorch(self, model) -> list[np.ndarray]:
        """
        Flatten state_dict tensors, filtering out:
          - Non-float dtypes (embedding index tables, LongTensor position ids)
          - Known non-parameter buffers (BatchNorm running stats)
          - Layers not matching self.layer_types (if specified)
        """
        weights = []
        sd          = model.state_dict()
        param_names = {n for n, _ in model.named_parameters()}
 
        for name, tensor in sd.items():
            if any(name.endswith(s) for s in _PYTORCH_NON_PARAM_SUFFIXES):
                continue
            if name not in param_names:
                continue
            if self.layer_types is not None:
                module_path = ".".join(name.split(".")[:-1])
                try:
                    module     = dict(model.named_modules())[module_path]
                    class_name = type(module).__name__
                    if not any(lt in class_name for lt in self.layer_types):
                        continue
                except KeyError:
                    pass
            if not tensor.is_floating_point():
                continue
            weights.append(tensor.cpu().detach().float().numpy().flatten())
 
        return weights
 
    # ── Keras helpers ─────────────────────────────────────────────────────────
 
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
 
    @staticmethod
    def _extract_keras(model) -> list[np.ndarray]:
        return [np.array(w).flatten().astype(np.float32)
                for w in model.trainable_weights]
 
    # ── Sub-sampling ──────────────────────────────────────────────────────────
 
    def _maybe_subsample(self, w: np.ndarray) -> np.ndarray:
        """Uniformly sub-sample the weight vector if it exceeds max_params."""
        if self.max_params is None or len(w) <= self.max_params:
            return w
        rng = np.random.default_rng(_SAMPLE_SEED)
        idx = rng.choice(len(w), size=self.max_params, replace=False)
        idx.sort()
        warnings.warn(
            f"PSC: weight vector length {len(w):,} exceeds "
            f"max_params={self.max_params:,}. "
            f"Sub-sampling to {self.max_params:,} elements (seed={_SAMPLE_SEED}). "
            "Set max_params=None to disable.",
            UserWarning, stacklevel=3,
        )
        return w[idx]
 
    # ── Alignment — Issue #3 fix ──────────────────────────────────────────────
 
    def _align(
        self,
        w_a: np.ndarray,
        w_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Align two weight vectors to the same length.
 
        Returns:
            (w_a_aligned, w_b_aligned, reliable)
 
        reliable=False when the size ratio exceeds _MAX_RELIABLE_RATIO,
        meaning truncation would discard the overwhelming majority of one
        model's parameters. The caller should return 0.5 in that case.
        """
        if len(w_a) == len(w_b):
            return w_a, w_b, True
 
        len_a, len_b = len(w_a), len(w_b)
        larger  = max(len_a, len_b)
        smaller = min(len_a, len_b)
        ratio   = larger / smaller   # always >= 1.0
 
        if ratio > _MAX_RELIABLE_RATIO:
            warnings.warn(
                f"PSC: weight vector size ratio is {ratio:.0f}x "
                f"({len_a:,} vs {len_b:,}). "
                f"Truncating the larger to {smaller:,} elements would discard "
                f"{100*(1 - 1/ratio):.0f}% of its parameters — PSC is unreliable "
                "for cross-architecture pairs with this size difference. "
                "Returning 0.5. Use FSC and RSC for cross-architecture comparison.",
                UserWarning, stacklevel=4,
            )
            return w_a, w_b, False   # caller must check reliable flag
 
        # Ratio is acceptable: truncate and warn at lower severity
        min_len = smaller
        warnings.warn(
            f"PSC: weight vectors have different lengths "
            f"({len_a:,} vs {len_b:,}, ratio={ratio:.1f}x). "
            f"Truncating both to {min_len:,}. "
            "This is an approximation; prefer comparing same-architecture models.",
            UserWarning, stacklevel=4,
        )
        return w_a[:min_len], w_b[:min_len], True
 
    # ── Similarity scores ─────────────────────────────────────────────────────
 
    def cosine_similarity_score(self, w_a: np.ndarray, w_b: np.ndarray) -> float:
        w_a, w_b, reliable = self._align(w_a, w_b)
        if not reliable:
            return 0.5
        cos_sim = cosine_similarity(w_a.reshape(1, -1), w_b.reshape(1, -1))[0, 0]
        return float((cos_sim + 1) / 2)
 
    def euclidean_similarity_score(self, w_a: np.ndarray, w_b: np.ndarray) -> float:
        w_a, w_b, reliable = self._align(w_a, w_b)
        if not reliable:
            return 0.5
        distance       = np.linalg.norm(w_a - w_b)
        mean_magnitude = (np.linalg.norm(w_a) + np.linalg.norm(w_b)) / 2
        if mean_magnitude < 1e-8:
            return 0.5
        return float(np.clip(1 / (1 + distance / mean_magnitude), 0, 1))
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def compute(self, model_a, model_b) -> float:
        """
        Compute PSC between two models.
 
        Returns float in [0, 1]:
          1.0 = identical weight direction
          0.5 = neutral / unreliable cross-arch / extraction error
          0.0 = opposite weight direction
        """
        try:
            w_a = self.extract_weights(model_a)
            w_b = self.extract_weights(model_b)
        except Exception as e:
            warnings.warn(
                f"PSC: weight extraction failed ({e}). Returning neutral 0.5.",
                UserWarning, stacklevel=2,
            )
            return 0.5
 
        if self.method == "cosine":
            return self.cosine_similarity_score(w_a, w_b)
        return self.euclidean_similarity_score(w_a, w_b)