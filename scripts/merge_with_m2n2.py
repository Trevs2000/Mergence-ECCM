import numpy as np
import pandas as pd
import joblib
import cma
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os

from scipy.optimize import minimize_scalar


class DualEngineMerger:
    """
    Optimise blend weight for a model pair using:
      1) CMA-ES (M2N2-style evolutionary search)
      2) Fallback: 1D bounded optimisation (minimize_scalar)

    Returns the better of the two.
    """

    def __init__(
        self,
        cma_sigma0: float = 0.3,
        cma_max_iterations: int = 50,
        cma_population_size: int = 10,
        scalar_max_iter: int = 100,
    ):
        self.cma_sigma0 = cma_sigma0
        self.cma_max_iterations = cma_max_iterations
        self.cma_population_size = cma_population_size
        self.scalar_max_iter = scalar_max_iter

    def _precompute_proba(self, model_a, model_b, X_val):
        y_proba_a = model_a.predict_proba(X_val)[:, 1]
        y_proba_b = model_b.predict_proba(X_val)[:, 1]
        return y_proba_a, y_proba_b

    def _neg_auc(self, ratio, y_proba_a, y_proba_b, y_val):
        ratio = np.clip(ratio, 0.0, 1.0)
        blend = ratio * y_proba_a + (1 - ratio) * y_proba_b
        return -roc_auc_score(y_val, blend)

    def run_cmaes(self, y_proba_a, y_proba_b, y_val):
        """
        CMA-ES primary engine.

        CMA-ES requires dim >= 2; we pad with a dummy variable.
        Only x[0] is used as the blend ratio. For production,
        scipy.optimize.minimize_scalar would be cleaner, but CMA-ES
        is used here to align with the M2N2 evolutionary optimisation framework.
        """
        x0 = [0.5, 0.5]

        opts = cma.CMAOptions()
        opts["maxiter"] = self.cma_max_iterations
        opts["popsize"] = self.cma_population_size
        opts["bounds"] = [[0.0, 0.0], [1.0, 1.0]]
        opts["verbose"] = -9
        opts["seed"] = 42

        es = cma.CMAEvolutionStrategy(x0, self.cma_sigma0, opts)
        n_eval = 0

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for s in solutions:
                r = np.clip(s[0], 0.0, 1.0)
                neg_auc = self._neg_auc(r, y_proba_a, y_proba_b, y_val)
                fitnesses.append(neg_auc)
            es.tell(solutions, fitnesses)
            n_eval += len(solutions)

        res = es.result
        best_ratio = np.clip(res.xbest[0], 0.0, 1.0)
        best_auc = -res.fbest

        return {
            "engine": "cmaes",
            "best_ratio": float(best_ratio),
            "best_auc": float(best_auc),
            "n_evaluations": int(n_eval),
            "converged": True,
        }

    def run_scalar(self, y_proba_a, y_proba_b, y_val):
        """1D bounded optimisation via minimize_scalar."""
        def neg_auc(r):
            return self._neg_auc(r, y_proba_a, y_proba_b, y_val)

        res = minimize_scalar(
            neg_auc,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"maxiter": self.scalar_max_iter},
        )

        return {
            "engine": "scalar",
            "best_ratio": float(res.x),
            "best_auc": float(-res.fun),
            "n_evaluations": int(res.nfev),
            "converged": bool(res.success),
        }

    def optimise(self, model_a, model_b, X_val, y_val):
        """
        Run CMA-ES, then scalar optimisation, and return the better result.
        """
        y_proba_a, y_proba_b = self._precompute_proba(model_a, model_b, X_val)

        cma_res = self.run_cmaes(y_proba_a, y_proba_b, y_val)
        scalar_res = self.run_scalar(y_proba_a, y_proba_b, y_val)

        #Picks better AUC; if equal, prefer CMA-ES for thesis
        if scalar_res["best_auc"] > cma_res["best_auc"]:
            return {
                **scalar_res,
                "secondary_auc": cma_res["best_auc"],
                "secondary_engine": "cmaes",
            }
        else:
            return {
                **cma_res,
                "secondary_auc": scalar_res["best_auc"],
                "secondary_engine": "scalar",
            }


class M2N2Pipeline:
    """
    Full pipeline: load models, select top pairs by ECCM,
    run dual-engine optimisation, compare with fixed-ratio results.
    """

    def __init__(
        self,
        models_dir: str,
        X_val,
        y_val,
        output_dir: str,
        cma_sigma0: float = 0.3,
        cma_max_iter: int = 50,
        cma_pop_size: int = 10,
        scalar_max_iter: int = 100,
    ):
        self.models_dir = models_dir
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.merger = DualEngineMerger(
            cma_sigma0=cma_sigma0,
            cma_max_iterations=cma_max_iter,
            cma_population_size=cma_pop_size,
            scalar_max_iter=scalar_max_iter,
        )
        self.models_cache = {}
        self.results = []

    def load_model(self, model_id: str):
        if model_id not in self.models_cache:
            path = Path(self.models_dir) / f"{model_id}.pkl"
            self.models_cache[model_id] = joblib.load(path)
        return self.models_cache[model_id]

    def run(self, top_pairs_df: pd.DataFrame, fixed_results_csv: str):
        fixed_df = pd.read_csv(fixed_results_csv)

        for i, row in top_pairs_df.iterrows():
            mid_a = row["model_a"]
            mid_b = row["model_b"]

            model_a = self.load_model(mid_a)
            model_b = self.load_model(mid_b)

            opt_res = self.merger.optimise(
                model_a, model_b, self.X_val, self.y_val
            )

            pair_fixed = fixed_df[
                (fixed_df["model_a"] == mid_a)
                & (fixed_df["model_b"] == mid_b)
            ]
            best_fixed_auc = pair_fixed["auc_merged"].max()

            auc_a = roc_auc_score(
                self.y_val, model_a.predict_proba(self.X_val)[:, 1]
            )
            auc_b = roc_auc_score(
                self.y_val, model_b.predict_proba(self.X_val)[:, 1]
            )
            best_parent = max(auc_a, auc_b)

            result = {
                "model_a": mid_a,
                "model_b": mid_b,
                "eccm": row["eccm"],
                "auc_a": float(auc_a),
                "auc_b": float(auc_b),
                "best_parent_auc": float(best_parent),
                "fixed_best_auc": float(best_fixed_auc),
                "fixed_improvement": float(best_fixed_auc - best_parent),
                "best_engine": opt_res["engine"],
                "opt_best_ratio": opt_res["best_ratio"],
                "opt_best_auc": opt_res["best_auc"],
                "opt_improvement": float(opt_res["best_auc"] - best_parent),
                "opt_vs_fixed": float(opt_res["best_auc"] - best_fixed_auc),
                "opt_n_evals": opt_res["n_evaluations"],
                "secondary_engine": opt_res["secondary_engine"],
                "secondary_auc": opt_res["secondary_auc"],
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(result)

            delta = result["opt_vs_fixed"]
            marker = "+" if delta > 0 else " "
            print(
                f"{i+1:3d}. {mid_a} + {mid_b}  |  "
                f"Fixed: {best_fixed_auc:.6f}  |  "
                f"Opt({opt_res['engine']}): {opt_res['best_auc']:.6f} "
                f"(r={opt_res['best_ratio']:.4f})  |  "
                f"Δ={delta:+.6f} {marker}"
            )

        results_df = pd.DataFrame(self.results)
        csv_path = f"{self.output_dir}/m2n2_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(self.results)} results to {csv_path}")

        return results_df

