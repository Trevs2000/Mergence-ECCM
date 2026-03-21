import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class EPCTrainer:
    """
    Evolutionary Pressure Compatibility (EPC)
    Trains a Random Forest on merge history.
    
    Input: PSC, RSC, FSC scores from past merges
    Output: Prediction of merge success
    
    Why Random Forest?
    - Fast to train (1 second for 80 merges)
    - Handles non-linear relationships
    - Interpretable (feature importance)
    - No hyperparameter tuning needed
    """
    
    def __init__(self):
        self.model = None
    
    def prepare_data(self, merge_history: pd.DataFrame):
        """
        Prepare features and targets from merge history.
        
        Expected columns:
        - psc: PSC score
        - fsc: FSC score
        - rsc: RSC score
        - improvement: Merged model AUC - best parent AUC
        """
        X = merge_history[['psc', 'fsc', 'rsc']].values
        y = merge_history['improvement'].values  # Continuous: AUC gain
        
        return X, y
    
    def train(self, merge_history: pd.DataFrame, n_trees: int = 100):
        """
        Train EPC model on historical merges.
        
        Example:
        merge_history has 80 rows
        For each merge: PSC=0.7, FSC=0.8, RSC=0.6, improvement=0.02
        
        RF learns: "High PSC + High FSC’ good merge"
        """
        X, y = self.prepare_data(merge_history)
        
        self.model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=8,  # Prevent overfitting
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        # Evaluate
        train_r2 = self.model.score(X, y)
        print(f" EPC trained on {len(merge_history)} merges")
        print(f" Train RÂ²: {train_r2:.4f}")
        
        return train_r2
    
    def predict(self, psc: float, fsc: float, rsc: float) -> float:
        """
        Predict merge success for new pair.
        
        Returns: Predicted AUC improvement
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = np.array([[psc, fsc, rsc]])
        pred = self.model.predict(X)
        
        return float(pred)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)