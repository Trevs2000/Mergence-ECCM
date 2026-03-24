from scripts.xai_explanantions import (
    load_fixed_results,
    load_m2n2_results,
    explain_pair_global,
    plot_pair_metrics,
)

def run_for_task(task: str, top_n: int = 5):
    fixed = load_fixed_results(task)
    m2n2 = load_m2n2_results(task)

    #Join on model_a, model_b
    merged = fixed.merge(
        m2n2, 
        on=["model_a", "model_b"],
        suffixes=("_fixed", "_opt"),
    )

    # print(merged.columns.tolist())

    #Take the top pairs by eccm
    top = merged.sort_values("eccm_fixed", ascending=False).head(top_n)

    for _, row in top.iterrows():
        text = explain_pair_global(row, row, task)
        print("\n---", task, row["model_a"], "+", row["model_b"], "---")
        print(text)

        plot_pair_metrics(
            row_fixed=row,
            row_m2n2=row,
            task=task,
            out_dir=f"./results/xai/{task}",
        )


if __name__ == "__main__":
    run_for_task("fraud", top_n=5)
    run_for_task("churn", top_n=5)