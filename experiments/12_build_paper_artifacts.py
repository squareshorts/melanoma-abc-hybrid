import os
import pandas as pd
from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.data.isic import load_isic_task3_labels
from src.reporting.figures import plot_roc_multi, plot_pr_multi, plot_calibration_multi

def _subset_summary(df, name):
    n = int(len(df))
    pos = int(df["label"].sum())
    neg = n - pos
    lesions = int(df["lesion_id"].nunique()) if "lesion_id" in df.columns else None
    return {
        "subset": name,
        "images": n,
        "lesions": lesions,
        "positive": pos,
        "negative": neg,
        "melanoma_prevalence": pos / n if n else 0.0,
    }

def _read_pred(path, label):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return (label, df["y_true"].values, df["y_prob"].values)

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    df_ham = load_ham_metadata(cfg["paths"]["ham_metadata"])
    df_isic = load_isic_task3_labels(cfg["paths"]["isic_task3_labels"])
    split = read_json(cfg["derived"]["split_json"])
    train_ids, val_ids, test_ids = set(split["train"]), set(split["val"]), set(split["test"])
    split_checks = {
        "train_val_lesion_overlap": len(train_ids & val_ids),
        "train_test_lesion_overlap": len(train_ids & test_ids),
        "val_test_lesion_overlap": len(val_ids & test_ids),
        "split_group_key": split.get("group_key", "lesion_id"),
    }
    if any(split_checks[k] for k in ["train_val_lesion_overlap", "train_test_lesion_overlap", "val_test_lesion_overlap"]):
        raise RuntimeError(f"Split leakage detected: {split_checks}")
    pd.DataFrame([split_checks]).to_csv("results/tables/table_split_integrity.csv", index=False)

    t1 = pd.DataFrame([{
        "dataset": "HAM10000 / ISIC 2018 training images",
        "images": int(df_ham["image_id"].nunique()),
        "lesions": int(df_ham["lesion_id"].nunique()),
        "positive": int(df_ham["label"].sum()),
        "negative": int((df_ham["label"]==0).sum()),
        "endpoint": "melanoma vs all other diagnoses",
        "split": "lesion_id-based train/validation/test"
    }])
    t1.to_csv("results/tables/table_I_dataset_summary.csv", index=False)

    ham_ids = set(df_ham["image_id"])
    isic_ids = set(df_isic["image_id"])
    overlap = len(ham_ids & isic_ids)
    t_prov = pd.DataFrame([
        {
            "dataset_evaluation_source": "HAM10000",
            "images": int(df_ham["image_id"].nunique()),
            "lesions": int(df_ham["lesion_id"].nunique()),
            "positive": int(df_ham["label"].sum()),
            "negative": int((df_ham["label"] == 0).sum()),
            "melanoma_prevalence": float(df_ham["label"].mean()),
            "label_definition": "dx == mel vs all others",
            "image_overlap_with_ham10000": "Reference",
            "interpretation": "Primary within-corpus evaluation",
        },
        {
            "dataset_evaluation_source": "Local ISIC 2018 Task 3 files",
            "images": int(df_isic["image_id"].nunique()),
            "lesions": int(df_ham[df_ham["image_id"].isin(isic_ids)]["lesion_id"].nunique()),
            "positive": int(df_isic["label"].sum()),
            "negative": int((df_isic["label"] == 0).sum()),
            "melanoma_prevalence": float(df_isic["label"].mean()),
            "label_definition": "Official Task 3 melanoma label",
            "image_overlap_with_ham10000": f"{overlap}/{len(isic_ids)} = {overlap / len(isic_ids):.0%}",
            "interpretation": "Secondary label/evaluation partition only; not external",
        },
        {
            "dataset_evaluation_source": "Optional future cohort",
            "images": None,
            "lesions": None,
            "positive": None,
            "negative": None,
            "melanoma_prevalence": None,
            "label_definition": "melanoma vs all others",
            "image_overlap_with_ham10000": "0 required",
            "interpretation": "True external validation only if overlap audit passes",
        },
    ])
    t_prov.to_csv("results/tables/table_dataset_provenance.csv", index=False)

    split_rows = []
    for name, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        split_rows.append(_subset_summary(df_ham[df_ham["lesion_id"].isin(ids)], name))
    df_isic_with_lesions = df_isic.merge(df_ham[["image_id", "lesion_id"]], on="image_id", how="left")
    split_rows.append(_subset_summary(df_isic_with_lesions.rename(columns={"label": "label"}), "secondary_isic_task3_labels"))
    pd.DataFrame(split_rows).to_csv("results/tables/table_subset_prevalence.csv", index=False)

    internal_series = [
        _read_pred("results/runs/deep_baseline/ham_test_predictions_deep.csv", "Deep"),
        _read_pred("results/runs/handcrafted/ham_test_predictions_handcrafted.csv", "ABC"),
        _read_pred("results/runs/hybrid/ham_test_predictions_hybrid.csv", "Hybrid"),
    ]
    internal_series = [s for s in internal_series if s is not None]
    if internal_series:
        plot_roc_multi(internal_series, "results/figures/fig_curves_internal_roc.png", None)
        plot_pr_multi(internal_series, "results/figures/fig_curves_internal_pr.png", None)
        plot_calibration_multi(internal_series, "results/figures/fig_curves_internal_calibration.png", "HAM10000 test calibration")

    secondary_series = [
        _read_pred("results/runs/deep_baseline/isic_task3_predictions_deep.csv", "Deep"),
        _read_pred("results/runs/handcrafted/isic_task3_predictions_handcrafted.csv", "ABC"),
        _read_pred("results/runs/hybrid/isic_task3_predictions_hybrid.csv", "Hybrid"),
    ]
    secondary_series = [s for s in secondary_series if s is not None]
    if secondary_series:
        plot_roc_multi(secondary_series, "results/figures/fig_curves_secondary_roc.png", "ISIC Task 3-label ROC")
        plot_pr_multi(secondary_series, "results/figures/fig_curves_secondary_pr.png", "ISIC Task 3-label precision-recall")
        plot_calibration_multi(secondary_series, "results/figures/fig_curves_secondary_calibration.png", "ISIC Task 3-label calibration")

    print("Built paper artifacts in results/")
