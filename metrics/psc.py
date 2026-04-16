import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, Dict

class PSCCalculator:
    """
    Parameter Space Compatibility (PSC)
    Measures weight similarity between two models.

    Two methods:
    1. Cosine similarity: direction of weights (angle)
    2. Euclidean similarity: actual weight values

    Cosine is better for models trained differently.
    Euclidean is better for identically initialized models.
    """

    def __init__(self, method: str = 'cosine'):
        """
        Args:
            method: 'cosine' (default, robust) or 'euclidean' (exact)
        """
        if method not in ['cosine', 'euclidean']:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def extract_weights(self, model) -> np.ndarray:
        """
        Extract weights from any sklearn or torch model.

        For sklearn (Random Forest, LogisticRegression, etc.):
        - Uses feature_importances_ (tree-based)
        - Uses coef_ (linear models)

        For torch models:
        - Concatenates all parameter tensors

        Why flatten?
        We want ONE vector per model to compare.
        Think of it like comparing two people by their DNA sequence.
        """
        weights = []

        # PyTorch models, using state_dict because thats pytorches official weight container
        if hasattr(model, 'state_dict'):
            for param in model.state_dict().values(): #looping through all the weights
                w = param.cpu().detach().numpy() #Converting it into numeric values
                weights.append(w.flatten()) #Flattening it into a numeric value and storing it in the list

        # sklearn tree-based (RF, GB, etc) tree models dont have global vectors such as state_dict so feature_importances_ is used
        elif hasattr(model, 'feature_importances_'):
            weights.append(model.feature_importances_.flatten())

        # sklearn linear (LogReg, Ridge, etc)
        elif hasattr(model, 'coef_'): #linear models measure weights using coefficient
            weights.append(model.coef_.flatten())
            if hasattr(model, 'intercept_'): #measures biases by intercept_
                weights.append(model.intercept_.flatten())

        if not weights:
            raise ValueError("Cannot extract weights from this model type")

        return np.concatenate(weights) #concatenating to get a single vector for the different weights

    def cosine_similarity_score(self, w_a: np.ndarray, w_b: np.ndarray) -> float:
        """
        Cosine similarity: how parallel are the weight vectors?

        Formula: cos(θ) = (a · b) / (||a|| * ||b||)

        Why this?
        - Values in [-1, 1]
        - 1 = identical direction (good)
        - 0 = orthogonal (very different)
        - -1 = opposite direction (opposite)

        Normalize to [0, 1]:
        (cos_sim + 1) / 2

        Example:
        Model A weights: [0.1, 0.2, 0.3]
        Model B weights: [0.05, 0.1, 0.15]
        → Same direction (B is scaled A)
        → cos_sim ≈ 1.0
        → Score ≈ 1.0 (good match!)
        """
        cos_sim = cosine_similarity(
            w_a.reshape(1, -1),
            w_b.reshape(1, -1)
        )[0, 0]

        # Shift from [-1, 1] to [0, 1] to normalize
        return (cos_sim + 1) / 2

    def euclidean_similarity_score(self, w_a: np.ndarray, w_b: np.ndarray) -> float:
        """
        Euclidean similarity: how close are the actual weights?

        Formula:
        distance = ||w_a - w_b||
        similarity = 1 / (1 + normalized_distance)

        Why this?
        - Penalizes actual weight differences
        - More strict than cosine
        - Values in [0, 1]

        Example:
        Model A: [0.5, 0.5]
        Model B: [0.4, 0.6]
        distance = sqrt((0.1)^2 + (0.1)^2) ≈ 0.14
        similarity ≈ 0.88 (decent match, but not exact)
        """
        #Euclidean norm calculation (the actual distance) - did not chose manhattan distance because it doesnt penalize big errors enough
        distance = np.linalg.norm(w_a - w_b)

        # Normalize by average magnitude (so comparison is fair for small vs large weight ranges)
        mean_magnitude = (np.linalg.norm(w_a) + np.linalg.norm(w_b)) / 2
        if mean_magnitude < 1e-8: #0.00000001
            return 0.5  #Both models have near-zero weights so treating the nuetrally

        normalized_distance = distance / mean_magnitude
        similarity = 1 / (1 + normalized_distance)

        return np.clip(similarity, 0, 1) #ensures similarity is between 0 and 1

    def compute(self, model_a, model_b) -> float:
        """
        Compute PSC between two models.

        Returns: float in [0, 1]
        - 1.0 = identical weights
        - 0.5 = somewhat similar
        - 0.0 = completely different
        """
        try:
            w_a = self.extract_weights(model_a)
            w_b = self.extract_weights(model_b)
        except Exception as e:
            print(f"Warning: Could not extract weights: {e}")
            return 0.5  #Neutral score

        if self.method == 'cosine':
            return self.cosine_similarity_score(w_a, w_b)
        else:
            return self.euclidean_similarity_score(w_a, w_b)