import pandas as pd
from scipy.stats import spearmanr

"""
ECCM ablation study.

Compares how well PSC, FSC, RSC, and full ECCM
correlate with actual merge improvement.

"""

def analyse_file(csvpath: str, name: str):
    df = pd.read_csv(csvpath)
    metrics = ["psc", "fsc", "rsc", "eccm"]
    print(f"\n{name}  {csvpath}")
    print("-" * 60)
    print(f"{'Metric':8}  {'Spearman r':11}  {'p-value':11}")
    print("-" * 60)
    for m in metrics:
        r, p = spearmanr(df[m], df["improvement"])
        print(f"{m:8}  {r:11.4f}  {p:11.4g}")

def analyse_m2n2(fixed_csv: str, m2n2_csv: str, name: str):
    """Correlate ECCM sub-metrics with CMA-ES optimised improvement."""
    fixed  = pd.read_csv(fixed_csv)
    m2n2   = pd.read_csv(m2n2_csv)
    joined = fixed.merge(m2n2[["model_a", "model_b", "opt_improvement"]],
                         on=["model_a", "model_b"], how="inner")
    metrics = ["psc", "fsc", "rsc", "eccm"]
    print(f"\n{name} - vs. opt_improvement (CMA-ES)")
    print("-" * 60)
    print(f"{'Metric':8}  {'Spearman r':11}  {'p-value':11}")
    print("-" * 60)
    for m in metrics:
        r, p = spearmanr(joined[m], joined["opt_improvement"])
        print(f"{m:8}  {r:11.4f}  {p:11.4g}")

def main():
    BASE = r"C:\Users\User\Desktop\ICTer\WordTemplate-1"

    fraud_fixed = fr"{BASE}\results\merges\fraud\merge_results_new_eccm.csv"
    churn_fixed = fr"{BASE}\results\merges\churn\merge_results_new_eccm.csv"
    fraud_m2n2  = fr"{BASE}\results\merges\fraud\m2n2_results.csv"
    churn_m2n2  = fr"{BASE}\results\merges\churn\m2n2_results.csv"

    # Original: fixed-ratio improvement
    analyse_file(fraud_fixed, "FRAUD")
    analyse_file(churn_fixed, "CHURN")

    # NEW: CMA-ES optimised improvement
    analyse_m2n2(fraud_fixed, fraud_m2n2, "FRAUD")
    analyse_m2n2(churn_fixed, churn_m2n2, "CHURN")


if __name__ == "__main__":
    main()