import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, f1_score, confusion_matrix, precision_recall_curve, auc

from src.calibration.abc_conditioned import fit_abccalibrator
from src.evaluation.calibration import calibration_stats
from src.reporting.figures import _save_with_vector

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    sort_idx = np.argsort(y_prob)
    y_true_s = y_true[sort_idx]
    y_prob_s = y_prob[sort_idx]
    bin_edges = np.array_split(np.arange(len(y_prob)), n_bins)
    
    ece = 0.0
    for bin_indices in bin_edges:
        if len(bin_indices) == 0: continue
        acc = np.mean(y_true_s[bin_indices])
        conf = np.mean(y_prob_s[bin_indices])
        weight = len(bin_indices) / len(y_prob)
        ece += weight * np.abs(acc - conf)
    return ece

def safe_logit(p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def decision_curve(y, p, thresholds):
    n = len(y)
    net_benefits = []
    for thr in thresholds:
        tp = np.sum((p >= thr) & (y == 1))
        fp = np.sum((p >= thr) & (y == 0))
        nb = 0.0 if (tp + fp == 0) else (tp / n) - (fp / n) * (thr / (1 - thr))
        net_benefits.append(nb)
    return np.array(net_benefits)

def evaluate_predictions(y, p):
    auc_val = roc_auc_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = auc(rec, prec)
    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=10)
    ll = log_loss(y, p)
    
    pred_class = (p >= 0.5).astype(int)
    f1 = f1_score(y, pred_class)
    tn, fp, fn, tp = confusion_matrix(y, pred_class).ravel()
    sens = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    acc = (tp + tn) / len(y)
    
    s = safe_logit(p)
    lr = LogisticRegression()
    lr.fit(s.reshape(-1, 1), y)
    cal_slope = lr.coef_[0][0]
    cal_intercept = lr.intercept_[0]
    
    thr = np.linspace(0.02, 0.50, 50)
    dca = decision_curve(y, p, thr)
    mean_dca = np.mean(dca)
    
    return {
        "ROC-AUC": auc_val, "PR-AUC": pr_auc, "Brier score": brier, "ECE": ece,
        "calibration slope": cal_slope, "calibration intercept": cal_intercept, "log loss": ll,
        "F1": f1, "sensitivity": sens, "specificity": spec, "accuracy": acc, "mean DCA": mean_dca
    }

def prepare_data(pred_csv, abc_csv):
    df_p = pd.read_csv(pred_csv)
    df_abc = pd.read_csv(abc_csv)
    df = df_p.merge(df_abc, on="image_id", how="inner")
    
    abc_cols = [c for c in df.columns if c.startswith("A_") or c.startswith("B_") or c.startswith("C_") or c.startswith("seg_")]
    if len(abc_cols) == 0:
        raise ValueError("No ABC features found!")
    
    df["deep_logit"] = safe_logit(df["y_prob"].values)
    return df, abc_cols

def print_summary(res_df, dataset_name):
    print(f"\n=== SUMMARY: {dataset_name} ===")
    d = res_df.set_index("Model")
    brier_imp = d.loc["ABC-Conditioned", "Brier score"] < d.loc["Platt", "Brier score"] and d.loc["ABC-Conditioned", "Brier score"] < d.loc["Isotonic", "Brier score"]
    dca_imp = d.loc["ABC-Conditioned", "mean DCA"] > d.loc["Platt", "mean DCA"] and d.loc["ABC-Conditioned", "mean DCA"] > d.loc["Isotonic", "mean DCA"]
    
    print(f"ABC-Conditioned improved Brier score vs Platt/Isotonic: {'Yes' if brier_imp else 'No'}")
    print(f"ABC-Conditioned improved DCA vs Platt/Isotonic: {'Yes' if dca_imp else 'No'}")
    print(f"ABC-Conditioned Brier: {d.loc['ABC-Conditioned', 'Brier score']:.4f} | Platt: {d.loc['Platt', 'Brier score']:.4f} | Iso: {d.loc['Isotonic', 'Brier score']:.4f}")
    print(f"ABC-Conditioned DCA: {d.loc['ABC-Conditioned', 'mean DCA']:.4f} | Platt: {d.loc['Platt', 'mean DCA']:.4f} | Iso: {d.loc['Isotonic', 'mean DCA']:.4f}")
    print(f"ABC-Conditioned AUC: {d.loc['ABC-Conditioned', 'ROC-AUC']:.4f} | Raw AUC: {d.loc['Raw', 'ROC-AUC']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ham-val", type=str, default="results/runs/deep_baseline/ham_val_predictions_deep.csv")
    parser.add_argument("--ham-test", type=str, default="results/runs/deep_baseline/ham_test_predictions_deep.csv")
    parser.add_argument("--bcn", type=str, default="results/runs/deep_baseline/bcn20000_predictions_deep.csv")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(f"{args.outdir}/tables", exist_ok=True)
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)
    
    print("Preparing data...")
    df_val, abc_cols = prepare_data(args.ham_val, "data/derived/features/ham_abc.csv")
    df_test, _ = prepare_data(args.ham_test, "data/derived/features/ham_abc.csv")
    df_bcn, _ = prepare_data(args.bcn, "data/derived/features/bcn20000_abc.csv")
    
    scaler = StandardScaler()
    C_val = scaler.fit_transform(df_val[abc_cols].values)
    C_test = scaler.transform(df_test[abc_cols].values)
    C_bcn = scaler.transform(df_bcn[abc_cols].values)
    
    print("Fitting models...")
    # Platt
    platt = LogisticRegression()
    platt.fit(df_val["deep_logit"].values.reshape(-1, 1), df_val["y_true"].values)
    
    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df_val["y_prob"].values, df_val["y_true"].values)
    
    # ABC-Conditioned
    abc_cal = fit_abccalibrator(df_val["deep_logit"].values, C_val, df_val["y_true"].values, lambdas=[0, 1e-4, 1e-3, 1e-2, 1e-1])
    
    def generate_all_preds(df, C):
        p_raw = df["y_prob"].values
        s = df["deep_logit"].values
        y = df["y_true"].values
        
        p_platt = platt.predict_proba(s.reshape(-1, 1))[:, 1]
        p_iso = iso.predict(p_raw)
        
        with torch.no_grad():
            p_abc = abc_cal(torch.tensor(s, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)).numpy()
            
        return y, {"Raw": p_raw, "Platt": p_platt, "Isotonic": p_iso, "ABC-Conditioned": p_abc}

    datasets = {"HAM Test": (df_test, C_test, "table_abccal_ham.csv", "ham"), "BCN20000": (df_bcn, C_bcn, "table_abccal_bcn.csv", "bcn")}
    
    for ds_name, (df, C, tbl_out, prefix) in datasets.items():
        y, preds = generate_all_preds(df, C)
        
        rows = []
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        fig_dca, ax_dca = plt.subplots(figsize=(6, 5))
        
        thr_grid = np.linspace(0.02, 0.50, 50)
        prev = np.mean(y)
        treat_all = prev - (1 - prev) * (thr_grid / (1 - thr_grid))
        ax_dca.plot(thr_grid, treat_all, linestyle=":", color="black", label="Treat all")
        ax_dca.plot(thr_grid, np.zeros_like(thr_grid), linestyle="-", color="black", label="Treat none")
        
        for model_name, p in preds.items():
            metrics = evaluate_predictions(y, p)
            metrics["Model"] = model_name
            rows.append(metrics)
            
            # Calib Plot
            mean_p, frac_pos, brier = calibration_stats(y, p, n_bins=10)
            ax_cal.plot(mean_p, frac_pos, marker="o", label=f"{model_name} (Brier={brier:.3f})")
            
            # DCA Plot
            nb = decision_curve(y, p, thr_grid)
            ax_dca.plot(thr_grid, nb, label=model_name)
            
        res_df = pd.DataFrame(rows)
        # reorder columns to put Model first
        cols = ["Model"] + [c for c in res_df.columns if c != "Model"]
        res_df = res_df[cols]
        res_df.to_csv(f"{args.outdir}/tables/{tbl_out}", index=False)
        print_summary(res_df, ds_name)
        
        ax_cal.plot([0,1],[0,1], linestyle="--", color="gray")
        ax_cal.set_xlabel("Mean predicted probability")
        ax_cal.set_ylabel("Fraction of positives")
        ax_cal.legend()
        _save_with_vector(fig_cal, f"{args.outdir}/figures/fig_calibration_{prefix}_abccal.png")
        
        ax_dca.set_ylim(-0.05, 0.2)
        ax_dca.set_xlim(0, 0.5)
        ax_dca.set_xlabel("Threshold probability")
        ax_dca.set_ylabel("Net benefit")
        ax_dca.legend()
        _save_with_vector(fig_dca, f"{args.outdir}/figures/fig_dca_{prefix}_abccal.png")
        
    # Feature Importance Plot
    alpha_vals = abc_cal.alpha.detach().numpy()
    beta_vals = abc_cal.beta.detach().numpy()
    
    df_coef = pd.DataFrame({"Feature": abc_cols, "Alpha (Slope modifier)": alpha_vals, "Beta (Intercept modifier)": beta_vals})
    df_coef.to_csv(f"{args.outdir}/tables/table_abccal_coefficients.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(abc_cols))
    width = 0.35
    ax.bar(x - width/2, np.abs(alpha_vals), width, label='|Alpha|')
    ax.bar(x + width/2, np.abs(beta_vals), width, label='|Beta|')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in abc_cols], rotation=90)
    ax.set_ylabel("Absolute Standardized Coefficient")
    ax.legend()
    fig.tight_layout()
    _save_with_vector(fig, f"{args.outdir}/figures/fig_abccal_coefficients.png")
    
    print("\nABC-Conditioned Calibration pipeline finished.")
