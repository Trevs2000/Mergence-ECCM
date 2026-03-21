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

# import shutil
# os.makedirs('/data', exist_ok=True)
# shutil.move('churn_train_with_churn_col.csv',
#             '/data/churn_train_with_churn_col.csv')
# shutil.move('churn_val_with_churn_col.csv',
#             '/data/churn_val_with_churn_col.csv')

class ChurnModelTrainer:

    def __init__(self, train_path: str, val_path: str, 
                 output_dir: str = './models/churn'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Loading processed train/val
        train_df = pd.read_csv(train_path)
        val_df   = pd.read_csv(val_path)

        self.X_train = train_df.drop('Churn', axis=1).values
        self.y_train = train_df['Churn'].values

        self.X_val   = val_df.drop('Churn', axis=1).values
        self.y_val   = val_df['Churn'].values

        self.models = []
        self.metadata = []

    def train_model(self, variant_id: int,
                    n_estimators: int = 100,
                    max_depth: int = 15,
                    min_samples_split: int = 10) -> Dict:

        X_train, y_train = self.X_train, self.y_train
        X_val,   y_val   = self.X_val,   self.y_val

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=variant_id,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)

        y_proba_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba_val)

        model_path = f"{self.output_dir}/churn_v{variant_id:02d}.pkl"
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

        print(f"Training {num_models} customer churn models...\n")

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
            model, self.X_train, self.y_train,
            cv=cv,
            scoring='roc_auc'
        )
        print(f"CV AUC: mean={auc_scores.mean():.4f}, std={auc_scores.std():.4f}")
        return auc_scores

    def train_logistic_baseline(self):
        X_train, y_train = self.X_train, self.y_train
        X_val,   y_val   = self.X_val,   self.y_val

        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs'
        )
        model.fit(X_train, y_train)

        y_proba_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba_val)
        print(f"Logistic baseline: AUC={auc:.4f}")
        return {
            "model_type": "logistic_regression",
            "auc": float(auc)
        }

    def train_benchmark_models(self, n_estimators=50, max_depth=15,
                               min_samples_split=10, num_runs=5):
        """
        Training 5 models with SAME hyperparameters but different random seeds.
        Purpose: measure variance / stability of results while using the hyperparameters from the RF model 
        that has the best AUC score (v03: AUC=0.9691 (trees=50, depth=15, split=10)).
        """
        print(f"\nTraining {num_runs} benchmark models (same params, different seeds)...\n")

        for run in range(num_runs):
            seed = 1000 + run  # seeds: 1000, 1001, 1002, 1003, 1004
            variant_id = 100 + run  # IDs: 100, 101, 102, 103, 104 (won't clash with 0–23)

            self.train_model(
                variant_id=variant_id,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )

        print(f"\nDone. {num_runs} benchmark models saved.")


if __name__ == "__main__":
    trainer = ChurnModelTrainer(
        train_path='./data/churn_train_with_churn_col.csv',
        val_path='./data/churn_val_with_churn_col.csv',
        output_dir='./models/churn'
    )
    trainer.train_batch(num_models=24)
    trainer.train_logistic_baseline()
    trainer.train_benchmark_models()
    trainer.cross_validate_model(
        n_estimators=150, max_depth=10, min_samples_split=5, n_splits=5
    )