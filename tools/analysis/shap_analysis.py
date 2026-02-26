import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau


def load_native_booster(model_path: str):
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def compute_shap_importance_booster(
    booster: xgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per feature using native XGBoost pred_contribs.
    """

    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    shap_values = booster.predict(dmatrix, pred_contribs=True)

    # Last column = bias term → remove
    shap_values = shap_values[:, :-1]

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df


def shap_stability_between_domains_booster(
    booster: xgb.Booster,
    X_internal: np.ndarray,
    X_external: np.ndarray,
    feature_names: list[str],
):

    imp_int = compute_shap_importance_booster(booster, X_internal, feature_names)
    imp_ext = compute_shap_importance_booster(booster, X_external, feature_names)

    merged = pd.merge(
        imp_int.rename(columns={"mean_abs_shap": "mean_abs_shap_internal"}),
        imp_ext.rename(columns={"mean_abs_shap": "mean_abs_shap_external"}),
        on="feature",
        how="inner",
    )

    merged["rank_internal"] = merged["mean_abs_shap_internal"].rank(ascending=False)
    merged["rank_external"] = merged["mean_abs_shap_external"].rank(ascending=False)

    spearman_corr, _ = spearmanr(merged["rank_internal"], merged["rank_external"])
    kendall_corr, _ = kendalltau(merged["rank_internal"], merged["rank_external"])

    merged = merged.sort_values("rank_internal").reset_index(drop=True)

    return spearman_corr, kendall_corr, merged