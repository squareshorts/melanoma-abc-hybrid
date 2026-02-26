import pandas as pd
import os


def load_internal_hybrid():
    return pd.read_csv(
        "results/tables/table_hybrid_internal_stratified_lesion.csv"
    )


def load_internal_deep():
    # Deep internal metrics are stored in summary.json
    import json
    with open("results/runs/deep_baseline/summary.json", "r") as f:
        s = json.load(f)

    d = s["test_thr_0.5"]

    row = {
        "model": "deep_baseline",
        "AUC": d["AUC"],
        "PR_AUC": d["PR_AUC"],
        "F1": d["F1"],
        "ACC": d["ACC"],
        "SENS": d["SENS"],
        "SPEC": d["SPEC"],
    }

    return pd.DataFrame([row])


def load_external():
    return pd.read_csv(
        "results/tables/table_external_isic_task3_snapshot.csv"
    )


if __name__ == "__main__":

    os.makedirs("results/tables", exist_ok=True)

    hybrid_int = load_internal_hybrid()
    deep_int = load_internal_deep()
    external = load_external()

    hybrid_int["dataset"] = "HAM_internal"
    deep_int["dataset"] = "HAM_internal"

    external_hybrid = external[external["model"].str.contains("hybrid")].copy()
    external_deep = external[external["model"].str.contains("deep")].copy()

    external_hybrid["dataset"] = "ISIC_external"
    external_deep["dataset"] = "ISIC_external"

    comparison = pd.concat(
        [
            hybrid_int[["model","dataset","AUC","PR_AUC","F1","SENS","SPEC"]],
            deep_int[["model","dataset","AUC","PR_AUC","F1","SENS","SPEC"]],
            external_hybrid[["model","dataset","AUC","PR_AUC","F1","SENS","SPEC"]],
            external_deep[["model","dataset","AUC","PR_AUC","F1","SENS","SPEC"]],
        ],
        ignore_index=True
    )

    comparison.to_csv(
        "results/tables/table_internal_external_comparison_stratified.csv",
        index=False
    )

    print(comparison)