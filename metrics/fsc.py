import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

class FSCCalculator:
    """
    Functional Similarity Compatibility (FSC)
    Measures output agreement between two models.
    
    Two strategies:
    1. Correlation: Are predictions linearly related?
    2. Agreement: Do they classify same samples the same way?
    """
    
    def __init__(self, strategy: str = 'correlation'):
        """
        Args:
            strategy: 'correlation' (for regression) or 'agreement' (for classification)
        """
        self.strategy = strategy
    
    def get_predictions(self, model, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from model.
        
        For classification: returns probabilities (not hard predictions)
        Why? More informative than binary labels.
        
        Example:
        Model predicts: [0.8, 0.3, 0.9] (probability of class 1)
        vs hard labels: [1, 0, 1]
        
        Soft predictions let us see confidence.
        Model A very confident (0.8) vs Model B not sure (0.51)
        → FSC catches this nuance
        """
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # For binary classification, return P(class=1)
            if proba.shape == 2:
                return proba[:, 1]
            # For multiclass, return max probability
            return np.max(proba, axis=1)
        else:
            # For regression models, return raw predictions
            return model.predict(X)
    
    def correlation_similarity(self, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
        """
        Pearson correlation: Do predictions trend together?
        
        Formula: r = cov(A, B) / (std(A) * std(B))
        
        Why Pearson?
        - Captures LINEAR relationship
        - Invariant to scaling
        - Interpretable (r=1 perfect correlation)
        
        Normalize to [0, 1]:
        (r + 1) / 2  (since r is in [-1, 1])
        
        Example:
        Model A predictions: [0.1, 0.5, 0.9]
        Model B predictions: [0.2, 0.4, 0.8]
        → Strong positive correlation → r ≈ 0.99
        → Normalized: ≈ 1.0 (excellent agreement)
        """
        try:
            r, _ = pearsonr(pred_a, pred_b)
        except:
            return 0.5
        
        # Normalize from [-1, 1] to [0, 1]
        return (r + 1) / 2
    
    def agreement_similarity(self, pred_a: np.ndarray, pred_b: np.ndarray, 
                            threshold: float = 0.5) -> float:
        """
        Classification agreement: % of samples classified the same way
        
        Why?
        For classification, we often care about same predicted class,
        not exact probability agreement.
        
        Example:
        Model A: [0.8, 0.3, 0.9] → labels [1, 0, 1]
        Model B: [0.75, 0.4, 0.85] → labels [1, 0, 1]
        → 100% agreement
        
        Formula: accuracy_score(binary_a, binary_b)
        """
        binary_a = (pred_a > threshold).astype(int)
        binary_b = (pred_b > threshold).astype(int)
        
        return accuracy_score(binary_a, binary_b)
    
    def compute(self, model_a, model_b, X: np.ndarray) -> float:
        """
        Compute FSC between two models on data X.
        
        Returns: float in [0, 1]
        - 1.0 = identical predictions
        - 0.5 = random agreement
        - 0.0 = opposite predictions
        """
        pred_a = self.get_predictions(model_a, X)
        pred_b = self.get_predictions(model_b, X)
        
        if self.strategy == 'correlation':
            return self.correlation_similarity(pred_a, pred_b)
        else:
            return self.agreement_similarity(pred_a, pred_b)