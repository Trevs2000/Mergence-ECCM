import pandas as pd


def select_top_pairs(results_csv: str, top_n: int = 50) -> pd.DataFrame:
    """
    Selecting top N model pairs by ECCM score from a fixed-ratio merge results CSV.

    Assumes the CSV has at least:
        - model_a
        - model_b
        - eccm

    ECCM is computed per pair, not per blend ratio, so we can
    safely drop duplicate (model_a, model_b) rows.
    """
    df = pd.read_csv(results_csv)

    #Deduplicating pairs; ECCM is the same across ratios for a given pair
    unique_pairs = df.drop_duplicates(subset=["model_a", "model_b"], keep="first")

    #Sorting by ECCM descending and take top N
    top_pairs = unique_pairs.nlargest(top_n, "eccm")

    print(f"Selected {len(top_pairs)} pairs from {len(unique_pairs)} total")
    print(
        f"ECCM range: {top_pairs['eccm'].min():.4f} – "
        f"{top_pairs['eccm'].max():.4f}"
    )

    #Only returning what the optimisation pipeline actually needs
    return top_pairs[["model_a", "model_b", "eccm"]].reset_index(drop=True)
