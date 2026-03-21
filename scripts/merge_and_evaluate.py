from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path
from itertools import combinations
from datetime import datetime
import os

from metrics.eccm import ECCMCalculator

class SimpleMerger:
    """
    Simple model merging: average predictions.

    Why simple?
    - Fast (instant)
    - Baseline quality
    - Gets data for EPC training

    Later (Phase 3): Swapping with M2N2 optimization.
    """

    @staticmethod
    def create_merged_model(model_a, model_b, blend_ratio: float = 0.5):
        """
        Create merged model that averages predictions.

        blend_ratio: weight for model_a
                    (1 - blend_ratio) for model_b
        """
        class BlendedModel:
            def __init__(self, m_a, m_b, ratio):
                self.m_a = m_a
                self.m_b = m_b
                self.ratio = ratio

            def predict_proba(self, X):
                p_a = self.m_a.predict_proba(X)
                p_b = self.m_b.predict_proba(X)
                return self.ratio * p_a + (1 - self.ratio) * p_b

            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)

        return BlendedModel(model_a, model_b, blend_ratio)

class MergePipeline:
    """
    Complete merge pipeline:
    1. Load models
    2. Merge all pairs
    3. Evaluate merged models
    4. Compute ECCM metrics
    5. Save results
    """

    def __init__(self, models_dir: str, X_val: np.ndarray, y_val: np.ndarray,
                 eccm_calc, output_dir: str = './results/merges'):
        self.models_dir = models_dir
        self.X_val = X_val
        self.y_val = y_val
        self.eccm_calc = eccm_calc
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = []

    def load_models(self) -> Dict[str, object]:
        """
        Load only main models v00–v23.
        Exclude benchmark models v100–v104.
        """
        models = {}
        for pkl_file in Path(self.models_dir).glob("*.pkl"):
            stem = pkl_file.stem          #e.g.: 'fraud_v00', 'fraud_v100'

            # Split on the last '_v'
            parts = stem.rsplit("_v", 1)
            if len(parts) != 2:
                continue

            variant_str = parts[1]       #'00', '15', '100', '101' ...
            try:
                variant_id = int(variant_str)
            except ValueError:
                continue

        # Keep only 0–23 and skip 100–104 
            if 0 <= variant_id < 24:
                models[stem] = joblib.load(pkl_file)

        print(f"Loaded {len(models)} models from {self.models_dir}")
        return models

    def evaluate_merge(self, model_a, model_b, merged_model,
                      model_a_id: str, model_b_id: str) -> Dict:
        """
        Evaluate merged model and compute all metrics.
        """
        # Get predictions
        y_proba_a = model_a.predict_proba(self.X_val)[:, 1]
        y_proba_b = model_b.predict_proba(self.X_val)[:, 1]
        y_proba_merged = merged_model.predict_proba(self.X_val)[:, 1]

        # Compute AUC
        auc_a = roc_auc_score(self.y_val, y_proba_a)
        auc_b = roc_auc_score(self.y_val, y_proba_b)
        auc_merged = roc_auc_score(self.y_val, y_proba_merged)

        # Determine success
        best_parent_auc = max(auc_a, auc_b)
        improvement = auc_merged - best_parent_auc
        success = 1 if improvement > 0 else 0

        # Compute ECCM metrics
        eccm_scores = self.eccm_calc.compute(
            model_a, model_b, self.X_val
        ) #took off , epc_pred=0.5 to compute eccm with data driven ratios

        result = {
            'model_a': model_a_id,
            'model_b': model_b_id,
            'auc_a': float(auc_a),
            'auc_b': float(auc_b),
            'auc_merged': float(auc_merged),
            'improvement': float(improvement),
            'success': int(success),
            'psc': eccm_scores['psc'],
            'fsc': eccm_scores['fsc'],
            'rsc': eccm_scores['rsc'],
            'eccm': eccm_scores['eccm'],
            'timestamp': datetime.now().isoformat()
        }

        return result

    def run(self, num_pairs: int = 276):
        """Run all merge experiments.
        - num_pairs: how many unique model pairs to evaluate (max C(24,2)=276)
        - For each pair, we test blend ratios in [0.3, 0.4, 0.5, 0.6, 0.7]
        """

        models = self.load_models()
        model_ids = sorted(list(models.keys()))

        # All unordered pairs of models
        from itertools import combinations
        all_pairs = list(combinations(model_ids, 2))
        pairs = all_pairs[:num_pairs]

        print(f"\nRunning {len(pairs)} pairs × 5 blend ratios...\n")

        for i, (mid_a, mid_b) in enumerate(pairs, 1):
            model_a = models[mid_a]
            model_b = models[mid_b]

            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]: #addind different blending ratios
                merged = SimpleMerger.create_merged_model(
                    model_a, model_b, blend_ratio=ratio
                )

                result = self.evaluate_merge(
                    model_a, model_b, merged, mid_a, mid_b
                )
                result["blend_ratio"] = ratio
                self.results.append(result)

            print(f"{i:3d}. {mid_a} + {mid_b}  (tested 5 ratios)") #276 pairs * 5 ratios = 1380 experiments

        # Save all results to CSV
        results_df = pd.DataFrame(self.results)
        csv_path = f"{self.output_dir}/merge_results_new_eccm.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(self.results)} results to {csv_path}")

        return results_df

from sklearn.model_selection import train_test_split

eccm_calc = ECCMCalculator()

fraud_df = pd.read_csv('./data/fraud_preprocessed.csv')
X = fraud_df.drop('Class', axis=1).values
y = fraud_df['Class'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

merge_pipeline = MergePipeline(
    models_dir='./models/fraud',
    X_val=X_val,
    y_val=y_val,
    eccm_calc=eccm_calc,
    output_dir='./results/merges/fraud'
)

results_df = merge_pipeline.run(num_pairs=276)

churn_val = pd.read_csv('./data/churn_val_with_churn_col.csv')

X_val = churn_val.drop('Churn', axis=1).values
y_val = churn_val['Churn'].values

merge_pipeline2 = MergePipeline(
    models_dir='./models/churn',
    X_val=X_val,
    y_val=y_val,
    eccm_calc=eccm_calc,
    output_dir='./results/merges/churn'
)

results_df2 = merge_pipeline2.run(num_pairs=276)