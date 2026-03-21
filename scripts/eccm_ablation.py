import pandas as pd
from scipy.stats import spearmanr

"""
ECCM ablation study.

Compares how well PSC, FSC, RSC, and full ECCM
correlate with actual merge improvement.

"""

def analyse_file(csv_path: str, name: str):
    df = pd.read_csv(csv_path)

    #Using the given columns from merge_and_evaluate.py: psc, fsc, rsc, eccm, improvement
    metrics = ["psc", "fsc", "rsc", "eccm"]

    print(f"\n{name} – {csv_path}")
    print("-" * 60)
    print(f"{'Metric':<8}  {'Spearman r':>11}  {'p-value':>11}")
    print("-" * 60)

    for m in metrics:
        r, p = spearmanr(df[m], df["improvement"])
        print(f"{m:<8}  {r:11.4f}  {p:11.4g}")


def main():
    fraud_csv = "./results/merges/fraud/merge_results_new_eccm.csv"
    churn_csv = "./results/merges/churn/merge_results_new_eccm.csv"

    analyse_file(fraud_csv, "FRAUD")
    analyse_file(churn_csv, "CHURN")


if __name__ == "__main__":
    main()