import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau


FEAT_COLS_ABC = [
    'A_asymmetry', 'B_border_irreg',
    'C_color_mean_v', 'C_color_std_v',
    'C_color_mean_s', 'C_color_std_s',
    'C_glcm_contrast_d1', 'C_glcm_homogeneity_d1',
    'C_glcm_contrast_d2', 'C_glcm_homogeneity_d2'
]


def load_booster():
    booster = xgb.Booster()
    booster.load_model("results/runs/hybrid/hybrid_xgb_native.json")
    return booster


def build_hybrid_matrix(abc_path, emb_path, ids_path, pred_path):
    abc = pd.read_csv(abc_path)
    ids = pd.read_csv(ids_path)
    emb = np.load(emb_path)
    preds = pd.read_csv(pred_path)

    abc["image_id"] = abc["image_id"].astype(str)
    ids["image_id"] = ids["image_id"].astype(str)
    preds["image_id"] = preds["image_id"].astype(str)

    merged = ids.merge(abc, on="image_id", how="left")
    mask = merged["image_id"].isin(set(preds["image_id"]))

    X_emb = emb[mask.values]  # (n, 1280)
    X_abc = merged.loc[mask, FEAT_COLS_ABC].to_numpy()  # (n, 10)

    X = np.concatenate([X_emb, X_abc], axis=1)  # (n, 1290)
    return X


def shap_values_pred_contribs(booster, X):
    d = xgb.DMatrix(X)
    sv = booster.predict(d, pred_contribs=True)  # (n, 1291) includes bias
    sv = sv[:, :-1]  # drop bias
    return sv


def mean_abs_importance(sv):
    return np.mean(np.abs(sv), axis=0)


def main():
    booster = load_booster()

    # Internal (HAM test)
    X_ham = build_hybrid_matrix(
        "data/derived/features/ham_abc.csv",
        "data/derived/embeddings/ham_effb0.npy",
        "data/derived/embeddings/ham_effb0_ids.csv",
        "results/runs/hybrid/ham_test_predictions_hybrid.csv",
    )

    # External (ISIC eval set)
    X_isic = build_hybrid_matrix(
        "data/derived/features/isic_task3_abc.csv",
        "data/derived/embeddings/isic_task3_effb0.npy",
        "data/derived/embeddings/isic_task3_effb0_ids.csv",
        "results/runs/hybrid/isic_task3_predictions_hybrid.csv",
    )

    sv_ham = shap_values_pred_contribs(booster, X_ham)   # (1513, 1290)
    sv_isic = shap_values_pred_contribs(booster, X_isic) # (10015, 1290)

    imp_ham = mean_abs_importance(sv_ham)
    imp_isic = mean_abs_importance(sv_isic)

    # Block indices
    emb_idx = np.arange(0, 1280)
    abc_idx = np.arange(1280, 1290)

    # Block totals
    emb_total_ham = imp_ham[emb_idx].sum()
    abc_total_ham = imp_ham[abc_idx].sum()
    emb_total_isic = imp_isic[emb_idx].sum()
    abc_total_isic = imp_isic[abc_idx].sum()

    # Percent contribution
    ham_sum = emb_total_ham + abc_total_ham
    isic_sum = emb_total_isic + abc_total_isic

    block_summary = pd.DataFrame([
        {
            "domain": "HAM_test",
            "emb_total_mean_abs_shap": emb_total_ham,
            "abc_total_mean_abs_shap": abc_total_ham,
            "emb_percent": emb_total_ham / ham_sum if ham_sum > 0 else np.nan,
            "abc_percent": abc_total_ham / ham_sum if ham_sum > 0 else np.nan,
        },
        {
            "domain": "ISIC_external",
            "emb_total_mean_abs_shap": emb_total_isic,
            "abc_total_mean_abs_shap": abc_total_isic,
            "emb_percent": emb_total_isic / isic_sum if isic_sum > 0 else np.nan,
            "abc_percent": abc_total_isic / isic_sum if isic_sum > 0 else np.nan,
        }
    ])

    # ABC-only stability (rank correlation across the 10 ABC descriptors)
    imp_abc_ham = imp_ham[abc_idx]
    imp_abc_isic = imp_isic[abc_idx]

    spearman_abc, _ = spearmanr(imp_abc_ham, imp_abc_isic)
    kendall_abc, _ = kendalltau(imp_abc_ham, imp_abc_isic)

    abc_table = pd.DataFrame({
        "feature": FEAT_COLS_ABC,
        "mean_abs_shap_ham": imp_abc_ham,
        "mean_abs_shap_isic": imp_abc_isic,
        "rank_ham": pd.Series(imp_abc_ham).rank(ascending=False).to_numpy(),
        "rank_isic": pd.Series(imp_abc_isic).rank(ascending=False).to_numpy(),
    }).sort_values("rank_ham")

    # Save outputs
    block_out = "results/tables/shap_block_contribution_hybrid.csv"
    abc_out = "results/tables/shap_abc_only_stability_hybrid.csv"

    block_summary.to_csv(block_out, index=False)
    abc_table.to_csv(abc_out, index=False)

    print("Saved:", block_out)
    print("Saved:", abc_out)
    print("ABC-only Spearman:", spearman_abc)
    print("ABC-only Kendall tau:", kendall_abc)


if __name__ == "__main__":
    main()