"""
train_churn_models.py - Train all churn RF and ExtraTrees variants.
 
Model categories:
  v00–v23   RF main variants        (merge candidates)
  v100–v104 RF benchmark variants   (evaluation only, never merged)
  v200–v211 ExtraTrees variants     (merge candidates — expands EPC training data)
  Logistic  Baseline                (evaluation only, not saved as pkl)
"""
 
import json
import os
from datetime import datetime
 
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import torch
import torch.nn as nn
 
# ── Output path ───────────────────────────────────────────────────────────────
BASE = r"C:\Users\User\Desktop\ICTer\WordTemplate-1"

 
# 24 RF variants — original merge candidates (v00–v23)
RF_VARIANTS = [
    (50, 10, 5), (50, 10, 10), (50, 15, 5),  (50, 15, 10), (50, 20, 10),
    (100,10, 5), (100,10,10),  (100,15, 5),  (100,15,10),  (100,20,10),
    (150,10, 5), (150,10,20),  (150,15,10),  (150,15,20),  (150,20, 5),
    (200,10,10), (200,10,20),  (200,15, 5),  (200,15,20),  (200,20,10),
    (75, 12, 8), (125,12,15),  (175,18,12),  (225,20,15),
]
 
# 12 ExtraTrees variants — new merge candidates (v200–v211)
ET_VARIANTS = [
    (100, 15,  5), (100, 20, 10), (150, 15, 10),
    (150, 20,  5), (200, 10, 10), (200, 15, 20),
    ( 75, 12,  8), (125, 18,  5), (175, 10, 15),
    (225, 20, 10), ( 50, 20, 20), (100, 10, 20),
]

# 12 MLP variants — NN merge candidates (v300–v311)
#(hidden_size, num_layers, dropout, learning_rate)
NN_VARIANTS = [
    (64,  2, 0.2, 1e-3), (64,  2, 0.3, 1e-3), (64,  3, 0.2, 5e-4),
    (128, 2, 0.2, 1e-3), (128, 2, 0.3, 5e-4), (128, 3, 0.2, 1e-3),
    (256, 2, 0.2, 5e-4), (256, 2, 0.3, 1e-3), (256, 3, 0.3, 5e-4),
    ( 64, 3, 0.3, 5e-4), (128, 3, 0.3, 1e-3), (256, 3, 0.2, 1e-4),
]

class ChurnMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
 
class ChurnModelTrainer:
 
    def __init__(self, train_path: str, val_path: str, output_dir: str = f"{BASE}\\models\\churn"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
 
        train_df     = pd.read_csv(train_path)
        val_df       = pd.read_csv(val_path)
        self.X_train = train_df.drop("Churn", axis=1).values
        self.y_train = train_df["Churn"].values
        self.X_val   = val_df.drop("Churn", axis=1).values
        self.y_val   = val_df["Churn"].values
        self.metadata: list = []
 
    # ── Core training helper ──────────────────────────────────────────────────
 
    def _fit_and_save(
        self,
        variant_id: int,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        model_class=RandomForestClassifier,
    ) -> dict:
        """
        Fit one model, embed the FSC sample, save as pkl, return metadata.
 
        model_class lets us reuse this method for both RF and ExtraTrees
        without duplicating any logic.
        """
        model = model_class(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=variant_id,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(self.X_train, self.y_train)
 
        # Embed 200 stratified training rows for FSC fallback
        _, X_sample, _, _ = train_test_split(
            self.X_train, self.y_train,
            test_size=min(200, len(self.X_train)) / len(self.X_train),
            stratify=self.y_train, random_state=42,
        )
        model.X_train_sample_ = X_sample.astype(np.float32)
 
        auc = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
 
        family = "rf" if model_class is RandomForestClassifier else "et"
        path   = f"{self.output_dir}/churn_{family}_v{variant_id:03d}.pkl"
        joblib.dump(model, path)
 
        meta = {
            "variant_id":   variant_id,
            "model_family": family,
            "hyperparams": {
                "n_estimators":      n_estimators,
                "max_depth":         max_depth,
                "min_samples_split": min_samples_split,
            },
            "auc":       round(auc, 6),
            "path":      path,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  {family}_v{variant_id:03d}: AUC={auc:.4f} "
              f"(trees={n_estimators}, depth={max_depth}, split={min_samples_split})")
        return meta
 
    # ── Public methods ────────────────────────────────────────────────────────
 
    def train_main_variants(self) -> list:
        """Train v00–v23 RF models — merge candidates."""
        print(f"\nTraining {len(RF_VARIANTS)} main churn RF variants (v000–v023)...\n")
        for i, (n, d, s) in enumerate(RF_VARIANTS):
            self.metadata.append(self._fit_and_save(i, n, d, s, RandomForestClassifier))
        with open(f"{self.output_dir}/metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        return self.metadata
 
    def train_extratrees_variants(self) -> list:
        """
        Train v200–v211 ExtraTrees models — additional merge candidates.
 
        Purpose: increases EPC training corpus from ~72 records to 400+.
        C(12,2)=66 intra-ET pairs + 12×24=288 cross-family (RF vs ET) pairs
        = 354 new merge rows before blend-ratio expansion.
        """
        print(f"\nTraining {len(ET_VARIANTS)} ExtraTrees variants (v200–v211)...\n")
        et_meta = []
        for i, (n, d, s) in enumerate(ET_VARIANTS):
            et_meta.append(
                self._fit_and_save(200 + i, n, d, s, ExtraTreesClassifier)
            )
        with open(f"{self.output_dir}/metadata_et.json", "w") as f:
            json.dump(et_meta, f, indent=2)
        return et_meta

    def train_nn_variants(self, epochs: int = 30, batch_size: int = 256) -> list:
        """
        Train v300–v311 PyTorch MLP models — NN merge candidates.
        Uses self.X_train/self.y_train for training and self.X_val/self.y_val
        for evaluation, matching the existing churn RF/ET workflow.
        """
        print(f"\nTraining {len(NN_VARIANTS)} MLP variants (v300–v311)...\n")
        nn_meta = []
        n_features = self.X_train.shape[1]

        for i, (hidden_size, num_layers, dropout, lr) in enumerate(NN_VARIANTS):
            variant_id = 300 + i

            model = ChurnMLP(n_features, hidden_size, num_layers, dropout)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [sum(self.y_train == 0) / max(sum(self.y_train == 1), 1)],
                    dtype=torch.float32
                )
            )

            X_tr_t = torch.tensor(self.X_train.astype(np.float32), dtype=torch.float32)
            y_tr_t = torch.tensor(self.y_train.astype(np.float32), dtype=torch.float32).unsqueeze(1)

            model.train()
            for _ in range(epochs):
                for start in range(0, len(X_tr_t), batch_size):
                    xb = X_tr_t[start:start + batch_size]
                    yb = y_tr_t[start:start + batch_size]
                    optimiser.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimiser.step()

            # Evaluate on the existing churn validation set
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(self.X_val.astype(np.float32), dtype=torch.float32)).squeeze().numpy()
            proba = expit(logits)
            auc = roc_auc_score(self.y_val, proba)

            # Embed calibration sample for ECCM FSC/RSC
            _, X_sample, _, _ = train_test_split(
                self.X_train, self.y_train,
                test_size=min(200, len(self.X_train)) / len(self.X_train),
                stratify=self.y_train, random_state=42,
            )
            model.X_train_sample_ = X_sample.astype(np.float32)

            path = f"{self.output_dir}/churn_mlp_v{variant_id:03d}.pkl"
            joblib.dump(model, path)

            meta = {
                "variant_id": variant_id,
                "model_family": "mlp",
                "hyperparams": {
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                },
                "auc": round(auc, 6),
                "path": path,
                "timestamp": datetime.now().isoformat(),
            }
            print(
                f"  mlp_v{variant_id:03d}: AUC={auc:.4f} "
                f"(hidden={hidden_size}, layers={num_layers}, dropout={dropout}, lr={lr})"
            )
            nn_meta.append(meta)

        with open(f"{self.output_dir}/metadata_nn.json", "w") as f:
            json.dump(nn_meta, f, indent=2)

        return nn_meta

    def train_benchmark_variants(
        self,
        n_estimators: int = 50,
        max_depth: int = 15,
        min_samples_split: int = 10,
        num_runs: int = 5,
    ) -> list:
        """
        Train v100–v104: stability benchmarks.
        Evaluation only - not merge variants.
        """
        print(f"\nTraining {num_runs} benchmark variants (evaluation only)...\n")
        bench_meta = []
        for run in range(num_runs):
            bench_meta.append(
                self._fit_and_save(100 + run, n_estimators, max_depth, min_samples_split)
            )
        return bench_meta
 
    def evaluate_logistic_baseline(self) -> dict:
        """Fit logistic regression for evaluation. Not saved as pkl."""
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(self.X_train, self.y_train)
        auc = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
        print(f"  Logistic baseline AUC={auc:.4f}")
        return {"model_type": "logistic_regression", "auc": round(auc, 6)}
 
    def cross_validate(self, n_estimators=50, max_depth=15, min_samples_split=10, n_splits=5):
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="roc_auc")
        print(f"  CV AUC: mean={auc.mean():.4f}  std={auc.std():.4f}")
        return auc
 

if __name__ == "__main__":
    trainer = ChurnModelTrainer(
        train_path="./data/churn_train_with_churn_col.csv",
        val_path="./data/churn_val_with_churn_col.csv",
        output_dir=f"{BASE}\\models\\churn",
    )
    trainer.train_main_variants()
    trainer.train_extratrees_variants()
    trainer.train_nn_variants() 
    trainer.train_benchmark_variants()
    trainer.evaluate_logistic_baseline()
    trainer.cross_validate(n_estimators=50, max_depth=15, min_samples_split=10)