import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

SEED = 42
CAL_FRAC = 0.20  # calibration subset fraction


def _to_float01(y):
    # robust: handles 0/1, 0.0/1.0, strings
    y = pd.Series(y).astype(float).values
    return (y > 0.5).astype(int)


def calibration_slope_intercept(y, p):
    # logistic recalibration: logit(y) = a + b * logit(p)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit_p, y)

    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def evaluate(name, y, p):
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "model": name,
        "AUC": float(roc_auc_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def recalibrate_one(df, tag):
    # df columns expected: image_id, y_true, y_prob
    y = _to_float01(df["y_true"])
    p = df["y_prob"].astype(float).values

    idx = np.arange(len(df))

    # Stratified split indices
    idx_cal, idx_test = train_test_split(
        idx,
        test_size=(1.0 - CAL_FRAC),
        stratify=y,
        random_state=SEED,
    )

    y_cal, p_cal = y[idx_cal], p[idx_cal]
    y_test, p_test = y[idx_test], p[idx_test]

    rows = []

    # Raw baseline
    raw = evaluate(f"{tag}_raw", y_test, p_test)
    rows.append(raw)

    # Platt scaling
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p_cal.reshape(-1, 1), y_cal)
    p_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]
    platt_row = evaluate(f"{tag}_platt", y_test, p_platt)
    rows.append(platt_row)

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    p_iso = iso.transform(p_test)
    iso_row = evaluate(f"{tag}_isotonic", y_test, p_iso)
    rows.append(iso_row)

    out = pd.DataFrame(rows)

    # Add deltas vs raw (for paper-ready reporting)
    for col in ["AUC", "Brier", "calibration_slope", "calibration_intercept"]:
        out[f"d{col}_vs_raw"] = out[col] - raw[col]

    split_info = {
        "seed": SEED,
        "cal_frac": CAL_FRAC,
        "n_total": int(len(df)),
        "n_cal": int(len(idx_cal)),
        "n_test": int(len(idx_test)),
        "pos_total": int(y.sum()),
        "pos_cal": int(y_cal.sum()),
        "pos_test": int(y_test.sum()),
        "idx_cal": idx_cal.tolist(),
        "idx_test": idx_test.tolist(),
    }

    # Also return the exact test set predictions for each method (optional, but useful)
    preds = {
        "raw": (idx_test, y_test, p_test),
        "platt": (idx_test, y_test, p_platt),
        "isotonic": (idx_test, y_test, p_iso),
    }

    return out, split_info, preds


def write_preds_csv(df, tag, preds_dict):
    # Save per-image predictions for the test subset (so you can later do DCA or operating points)
    for method, (idx_test, y_test, p_hat) in preds_dict.items():
        out_df = df.iloc[idx_test][["image_id"]].copy()
        out_df["y_true"] = y_test.astype(int)
        out_df["y_prob"] = p_hat.astype(float)
        out_df.to_csv(f"results/audit/preds_isic_{tag}_{method}.csv", index=False)


def main():
    os.makedirs("results/audit", exist_ok=True)

    deep_path = "results/runs/deep_baseline/isic_task3_predictions_deep.csv"
    hyb_path = "results/runs/hybrid/isic_task3_predictions_hybrid.csv"

    deep_df = pd.read_csv(deep_path)
    hyb_df = pd.read_csv(hyb_path)

    deep_out, deep_split, deep_preds = recalibrate_one(deep_df, "deep")
    hyb_out, hyb_split, hyb_preds = recalibrate_one(hyb_df, "hybrid")

    deep_out.to_csv("results/audit/recalibration_isic_deep.csv", index=False)
    hyb_out.to_csv("results/audit/recalibration_isic_hybrid.csv", index=False)

    with open("results/audit/recalibration_isic_deep_split.json", "w", encoding="utf-8") as f:
        json.dump(deep_split, f, indent=2)

    with open("results/audit/recalibration_isic_hybrid_split.json", "w", encoding="utf-8") as f:
        json.dump(hyb_split, f, indent=2)

    write_preds_csv(deep_df, "deep", deep_preds)
    write_preds_csv(hyb_df, "hybrid", hyb_preds)

    print("\nDEEP\n", deep_out)
    print("\nHYBRID\n", hyb_out)
    print("\nWrote:")
    print("  results/audit/recalibration_isic_deep.csv")
    print("  results/audit/recalibration_isic_hybrid.csv")
    print("  results/audit/recalibration_isic_deep_split.json")
    print("  results/audit/recalibration_isic_hybrid_split.json")
    print("  results/audit/preds_isic_deep_{raw,platt,isotonic}.csv")
    print("  results/audit/preds_isic_hybrid_{raw,platt,isotonic}.csv")


if __name__ == "__main__":
    main()