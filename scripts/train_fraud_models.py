import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib
import json
from datetime import datetime
from typing import Dict
import os


class FraudModelTrainer:
    """
    Training multiple Random Forest models with varied hyperparameters.
    Goal: 24 diverse models for merging experiments.
    """

    def __init__(self, data_path: str, output_dir: str = './models/fraud'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load preprocessed data
        self.df = pd.read_csv(data_path)
        self.X = self.df.drop('Class', axis=1).values
        self.y = self.df['Class'].values

        self.models = []
        self.metadata = []

    def train_model(self, variant_id: int,
                   n_estimators: int = 100,
                   max_depth: int = 15,
                   min_samples_split: int = 10) -> Dict:
        """
        Training one RF variant.

        Hyperparameters:
        - n_estimators: # of trees (more = better but slower)
        - max_depth: tree depth (deeper = overfit risk)
        - min_samples_split: min samples to split node (regularization)
        """

        # Train-test split (use variant_id as seed for different splits)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=variant_id,
            stratify=self.y  # Keep fraud ratio same
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=variant_id,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalance
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba_test)

        # Save
        model_path = f"{self.output_dir}/fraud_v{variant_id:02d}.pkl"
        joblib.dump(model, model_path)

        metadata = {
            'variant_id': variant_id,
            'hyperparams': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            },
            'auc': float(auc),
            'path': model_path,
            'timestamp': datetime.now().isoformat()
        }

        self.metadata.append(metadata)
        self.models.append((variant_id, model))

        print(f"v{variant_id:02d}: AUC={auc:.4f} "
              f"(trees={n_estimators}, depth={max_depth}, split={min_samples_split})")

        return metadata

    def train_batch(self, num_models: int = 24):
        """
        Training 24 models with varied hyperparameters.

        Strategy: Grid of variations
        - n_estimators: 50, 100, 150, 200
        - max_depth: 10, 15, 20
        - min_samples_split: 5, 10, 20

        Total: 4 × 3 × 3 = 36, picking first 24 for experiments.
        """

        variants = [
            (50, 10, 5),
            (50, 10, 10),
            (50, 15, 5),
            (50, 15, 10),
            (50, 20, 10),
            (100, 10, 5),
            (100, 10, 10),
            (100, 15, 5),
            (100, 15, 10),
            (100, 20, 10),
            (150, 10, 5),
            (150, 10, 20),
            (150, 15, 10),
            (150, 15, 20),
            (150, 20, 5),
            (200, 10, 10),
            (200, 10, 20),
            (200, 15, 5),
            (200, 15, 20),
            (200, 20, 10),
            (75, 12, 8),
            (125, 12, 15),
            (175, 18, 12),
            (225, 20, 15),
        ][:num_models]

        print(f"Training {num_models} fraud detection models...\n")

        for i, (n_est, max_d, min_samp) in enumerate(variants):
            self.train_model(i, n_estimators=n_est,
                           max_depth=max_d, min_samples_split=min_samp)

        # Save metadata
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\nTrained {num_models} models")
        print(f"Saved to {self.output_dir}/")

        return self.metadata

    def cross_validate_model(self, n_estimators, max_depth, min_samples_split, n_splits=5):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = cross_val_score(
        model, self.X, self.y,
        cv=cv,
        scoring='roc_auc'
        )
        print(f"CV AUC: mean={auc_scores.mean():.4f}, std={auc_scores.std():.4f}")
        return auc_scores

    def train_logistic_baseline(self, variant_id: int = 999):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=variant_id,
            stratify=self.y
       )
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs'
        )
        model.fit(X_train, y_train)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba_test)
        print(f"Logistic baseline: AUC= {auc:.4f}")
        return {
            "model_type": "logistic_regression",
            "auc": float(auc),
            "variant_id": variant_id
        }


    def train_benchmark_models(self, n_estimators=150, max_depth=10, min_samples_split=5, num_runs=5):
        """
        Training 5 models with same hyperparameters but different random seeds.
        Purpose: measure variance / stability of results while using the hyperparameters from the RF model 
        that has the best AUC score (v10: AUC=0.9922 (trees=150, depth=10, split=5)).
        """
        print(f"\nTraining {num_runs} benchmark models (same params, different seeds)...\n")

        for run in range(num_runs):
            seed = 1000 + run  #seeds: 1000, 1001, 1002, 1003, 1004
            variant_id = 100 + run  #IDs: 100, 101, 102, 103, 104

            self.train_model(
                variant_id=variant_id,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )

        print(f"\nDone. {num_runs} benchmark models saved.")


# Usage in Colab
if __name__ == "__main__":
    trainer = FraudModelTrainer(
        data_path='./data/fraud_preprocessed.csv',
        output_dir='./models/fraud'
    )
    trainer.train_batch(num_models=24)
    trainer.train_logistic_baseline()
    trainer.train_benchmark_models()
    trainer.cross_validate_model(
        n_estimators=150, max_depth=10, min_samples_split=5, n_splits=5
    )
