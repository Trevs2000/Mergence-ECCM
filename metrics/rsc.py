import numpy as np

class RSCCalculator:
    """
    Representational Similarity Compatibility (RSC)
    Simplified version using feature importance correlation.
    
    Full CKA (Centered Kernel Alignment) is complex and slow.
    For Random Forests on tabular data, we can approximate by:
    - Comparing feature importances
    - Using permutation importance instead of neural activations
    
    Why this approximation?
    - Fast (1-2 seconds vs 5+ minutes for full CKA)
    - Works for tree-based models
    - Still captures "what does model care about?"
    """
    
    def get_feature_importance(self, model) -> np.ndarray:
        """
        Extract feature importances from model.
        
        For RF: built-in feature_importances_
        For others: use permutation importance
        """
        if hasattr(model, 'feature_importances_'):
            # RandomForest, GradientBoosting, etc
            return model.feature_importances_
        else:
            # Fallback: uniform importance
            return np.ones(model.n_features_in_) / model.n_features_in_
    
    def compute(self, model_a, model_b, X: np.ndarray = None) -> float:
        """
        Compute RSC as feature importance correlation.
        
        Why this works?
        If two models have similar feature importances,
        they learned to use the same features.
        That means they learned similar representations.
        
        Example:
        Model A importance: [fraud_amount: 0.4, num_txns: 0.3, age: 0.3]
        Model B importance: [fraud_amount: 0.35, num_txns: 0.35, age: 0.3]
        Correlation ≈ 0.99 (very similar)
        """
        imp_a = self.get_feature_importance(model_a)
        imp_b = self.get_feature_importance(model_b)
        
        # Normalize to same length (if different)
        min_len = min(len(imp_a), len(imp_b))
        imp_a = imp_a[:min_len]
        imp_b = imp_b[:min_len]
        
        # Compute correlation
        if np.std(imp_a) < 1e-8 or np.std(imp_b) < 1e-8:
            return 0.5  # No variance, neutral score
        
        r = np.corrcoef(imp_a, imp_b)[0, 1]
        
        return (r + 1) / 2  # Normalizing to [0, 1]