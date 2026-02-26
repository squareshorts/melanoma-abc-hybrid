import numpy as np
import pandas as pd
import xgboost as xgb


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

    feat_cols = [
        'A_asymmetry', 'B_border_irreg',
        'C_color_mean_v', 'C_color_std_v',
        'C_color_mean_s', 'C_color_std_s',
        'C_glcm_contrast_d1', 'C_glcm_homogeneity_d1',
        'C_glcm_contrast_d2', 'C_glcm_homogeneity_d2'
    ]

    X_emb = emb[mask.values]
    X_abc = merged.loc[mask, feat_cols].to_numpy()

    X = np.concatenate([X_emb, X_abc], axis=1)

    return X


def compute_shap_mean_abs(booster, X):

    dmatrix = xgb.DMatrix(X)

    shap_values = booster.predict(dmatrix, pred_contribs=True)

    shap_values = shap_values[:, :-1]  # remove bias

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    return mean_abs


def main():

    booster = load_booster()

    # HAM internal test
    X_ham = build_hybrid_matrix(
        "data/derived/features/ham_abc.csv",
        "data/derived/embeddings/ham_effb0.npy",
        "data/derived/embeddings/ham_effb0_ids.csv",
        "results/runs/hybrid/ham_test_predictions_hybrid.csv",
    )

    # ISIC external
    X_isic = build_hybrid_matrix(
        "data/derived/features/isic_task3_abc.csv",
        "data/derived/embeddings/isic_task3_effb0.npy",
        "data/derived/embeddings/isic_task3_effb0_ids.csv",
        "results/runs/hybrid/isic_task3_predictions_hybrid.csv",
    )

    print("HAM matrix:", X_ham.shape)
    print("ISIC matrix:", X_isic.shape)

    shap_ham = compute_shap_mean_abs(booster, X_ham)
    shap_isic = compute_shap_mean_abs(booster, X_isic)

    # Rank correlation
    from scipy.stats import spearmanr, kendalltau

    spearman_corr, _ = spearmanr(shap_ham, shap_isic)
    kendall_corr, _ = kendalltau(shap_ham, shap_isic)

    print("Spearman rank correlation:", spearman_corr)
    print("Kendall tau:", kendall_corr)


if __name__ == "__main__":
    main()