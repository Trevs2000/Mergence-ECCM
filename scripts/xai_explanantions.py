import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

#---------- 1. Loaders ----------

def load_fixed_results(task: str) -> pd.DataFrame:
    path = f"./results/merges/{task}/merge_results_new_eccm.csv"
    return pd.read_csv(path)


def load_m2n2_results(task: str) -> pd.DataFrame:
    path = f"./results/merges/{task}/m2n2_results.csv"
    return pd.read_csv(path)


def load_model(task: str, model_id: str):
    path = Path(f"./models/{task}") / f"{model_id}.pkl"
    return joblib.load(path)


#---------- 2. Natural language explanations ----------

def explain_pair_global(row_fixed: pd.Series,
                        row_m2n2: pd.Series,
                        task: str) -> str:
    """
    Explains, in text, why this pair was merged and
    what M2N2 added on top of fixed ratios.
    """

    a = row_fixed["model_a"]
    b = row_fixed["model_b"]

    psc = row_fixed["psc"]
    fsc = row_fixed["fsc"]
    rsc = row_fixed["rsc"]
    eccm = row_fixed["eccm_fixed"]
    base_impr = row_fixed["improvement"]

    opt_ratio = row_m2n2["opt_best_ratio"]
    opt_impr = row_m2n2["opt_improvement"]
    delta_vs_fixed = row_m2n2["opt_vs_fixed"]

    #template-style language
    text = []

    text.append(
        f"For {task}, models {a} and {b} were selected because "
        f"their overall ECCM compatibility score was {eccm:.3f}."
    )

    text.append(
        f"Structurally, PSC={psc:.3f}, FSC={fsc:.3f}, and RSC={rsc:.3f}, "
        f"which indicates that the models {'often agree' if psc > 0.9 else 'disagree on some cases'} "
        f"and have {'very similar' if rsc > 0.9 else 'moderately similar'} ranking behaviour."
    )

    if base_impr >= 0:
        text.append(
            f"Using the simple fixed ratios, the best merge improved AUC by "
            f"{base_impr:.6f} over the better parent."
        )
    else:
        text.append(
            f"Using the simple fixed ratios, the best merge slightly reduced AUC "
            f"by {abs(base_impr):.6f} compared to the better parent."
        )

    text.append(
        f"The M2N2-style optimiser then searched all blend ratios between 0 and 1 "
        f"and found an optimal weight of {opt_ratio:.3f} for model {a} "
        f"(and {1 - opt_ratio:.3f} for {b})."
    )

    if delta_vs_fixed > 0:
        text.append(
            f"This tuning increased AUC by an additional {delta_vs_fixed:.6f} "
            f"beyond the best fixed ratio."
        )
    elif delta_vs_fixed < 0:
        text.append(
            f"In this case, the optimiser did not improve on the fixed grid "
            f"(AUC decreased by {abs(delta_vs_fixed):.6f}), suggesting that the "
            f"simple ratios were already near-optimal."
        )
    else:
        text.append(
            "The optimiser matched, but did not exceed, the best fixed-ratio AUC, "
            "which is consistent with a fairly flat performance landscape."
        )

    return " ".join(text)


#---------- 3. Simple plots per pair ----------

def plot_pair_metrics(row_fixed: pd.Series,
                      row_m2n2: pd.Series,
                      task: str,
                      out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    a = row_fixed["model_a"]
    b = row_fixed["model_b"]

    #Bar plot for PSC/FSC/RSC
    metrics = ["psc", "fsc", "rsc"]
    values = [row_fixed[m] for m in metrics]

    plt.figure(figsize=(4, 3))
    plt.bar(metrics, values, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylim(0, 1)
    plt.title(f"{task} – {a} + {b} similarity metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{task}_{a}_{b}_metrics.png", dpi=150)
    plt.close()

    #Bar plot for AUC: best parent vs fixed vs M2N2
    best_parent = row_m2n2["best_parent_auc"]
    fixed = row_m2n2["fixed_best_auc"]
    opt = row_m2n2["opt_best_auc"]

    labels = ["Best parent", "Fixed merge", "Optimised merge"]
    vals = [best_parent, fixed, opt]

    plt.figure(figsize=(4, 3))
    plt.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    plt.ylim(min(vals) - 0.001, max(vals) + 0.001)
    plt.xticks(rotation=20)
    plt.title(f"{task} – {a} + {b} AUC comparison")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{task}_{a}_{b}_auc.png", dpi=150)
    plt.close()
