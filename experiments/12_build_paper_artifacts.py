import os
import pandas as pd
from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.reporting.figures import plot_roc, plot_pr, plot_calibration

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    df_ham = load_ham_metadata(cfg["paths"]["ham_metadata"])
    split = read_json(cfg["derived"]["split_json"])

    t1 = pd.DataFrame([{
        "dataset": "HAM10000",
        "images": int(df_ham["image_id"].nunique()),
        "lesions": int(df_ham["lesion_id"].nunique()),
        "positive": int(df_ham["label"].sum()),
        "negative": int((df_ham["label"]==0).sum()),
        "split": "lesion_id (patient/lesion-level)"
    }])
    t1.to_csv("results/tables/table_I_dataset_summary.csv", index=False)

    # Curves: deep internal
    deep_pred = "results/runs/deep_baseline/ham_test_predictions_deep.csv"
    if os.path.exists(deep_pred):
        df = pd.read_csv(deep_pred)
        plot_roc(df["y_true"].values, df["y_prob"].values, "results/figures/fig_roc_internal_deep.png", "ROC (HAM test, deep)")
        plot_pr(df["y_true"].values, df["y_prob"].values, "results/figures/fig_pr_internal_deep.png", "PR (HAM test, deep)")
        plot_calibration(df["y_true"].values, df["y_prob"].values, "results/figures/fig_calibration_internal_deep.png", "Calibration (HAM test, deep)")

    # Curves: hybrid external
    ext_pred = "results/runs/hybrid/isic_task3_predictions_hybrid.csv"
    if os.path.exists(ext_pred):
        df = pd.read_csv(ext_pred)
        plot_roc(df["y_true"].values, df["y_prob"].values, "results/figures/fig_roc_external_hybrid.png", "ROC (ISIC Task3, hybrid)")
        plot_pr(df["y_true"].values, df["y_prob"].values, "results/figures/fig_pr_external_hybrid.png", "PR (ISIC Task3, hybrid)")
        plot_calibration(df["y_true"].values, df["y_prob"].values, "results/figures/fig_calibration_external_hybrid.png", "Calibration (ISIC Task3, hybrid)")

    print("Built paper artifacts in results/")
