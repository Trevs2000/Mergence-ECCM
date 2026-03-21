import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.select_top_pairs import select_top_pairs
from scripts.merge_with_m2n2 import M2N2Pipeline


def run_fraud_experiments():
    #Loading fraud data and create validation split
    fraud_df = pd.read_csv("./data/fraud_preprocessed.csv")
    X = fraud_df.drop("Class", axis=1).values
    y = fraud_df["Class"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    #Top pairs by ECCM (from fixed-ratio runs)
    fixed_results_csv = "./results/merges/fraud/merge_results_new_eccm.csv"
    fraud_top = select_top_pairs(fixed_results_csv, top_n=50)

    #Run dual-engine optimisation (CMA-ES + scalar)
    fraud_pipeline = M2N2Pipeline(
        models_dir="./models/fraud",
        X_val=X_val,
        y_val=y_val,
        output_dir="./results/merges/fraud",
        cma_sigma0=0.3,
        cma_max_iter=50,
        cma_pop_size=10,
        scalar_max_iter=100,
    )

    fraud_results = fraud_pipeline.run(
        top_pairs_df=fraud_top,
        fixed_results_csv=fixed_results_csv,
    )
    return fraud_results


def run_churn_experiments():
    churn_val = pd.read_csv("./data/churn_val_with_churn_col.csv")
    X_val_churn = churn_val.drop("Churn", axis=1).values
    y_val_churn = churn_val["Churn"].values

    fixed_results_csv = "./results/merges/churn/merge_results_new_eccm.csv"
    churn_top = select_top_pairs(fixed_results_csv, top_n=50)

    churn_pipeline = M2N2Pipeline(
        models_dir="./models/churn",
        X_val=X_val_churn,
        y_val=y_val_churn,
        output_dir="./results/merges/churn",
        cma_sigma0=0.3,
        cma_max_iter=50,
        cma_pop_size=10,
        scalar_max_iter=100,
    )

    churn_results = churn_pipeline.run(
        top_pairs_df=churn_top,
        fixed_results_csv=fixed_results_csv,
    )
    return churn_results


if __name__ == "__main__":
    print("Running M2N2-style experiments on FRAUD...")
    fraud_results = run_fraud_experiments()

    print("\nRunning M2N2-style experiments on CHURN...")
    churn_results = run_churn_experiments()

    print("\nDone.")
