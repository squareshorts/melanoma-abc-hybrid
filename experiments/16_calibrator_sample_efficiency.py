import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Load secondary ISIC Task 3-label predictions.
    df_deep = pd.read_csv("results/runs/deep_baseline/isic_task3_predictions_deep.csv")
    df_hybrid = pd.read_csv("results/runs/hybrid/isic_task3_predictions_hybrid.csv")
    
    models = {"Deep (EffB0)": df_deep, "Hybrid (ABC+Emb)": df_hybrid}
    sample_sizes = [50, 100, 200, 500, 1000]
    n_splits = 20
    
    results = []
    
    for m_name, df in models.items():
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values
        
        # Clip probabilities to avoid log-loss issues
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        
        for N in sample_sizes:
            briers_platt, briers_iso, briers_raw = [], [], []
            aucs = []
            
            for i in range(n_splits):
                np.random.seed(42 + i)
                idx = np.random.permutation(len(y_true))
                cal_idx, eval_idx = idx[:N], idx[N:]
                
                # Calibration subset
                y_cal = y_true[cal_idx]
                p_cal = y_prob[cal_idx]
                if len(np.unique(y_cal)) < 2:
                    continue # Need both classes
                
                # Eval subset
                y_ev = y_true[eval_idx]
                p_ev = y_prob[eval_idx]
                
                # Raw Brier
                briers_raw.append(brier_score_loss(y_ev, p_ev))
                aucs.append(roc_auc_score(y_ev, p_ev))
                
                # Platt Scaling
                lr = LogisticRegression(solver="lbfgs", C=1e5) # almost no regularization
                p_cal_logit = np.log(p_cal / (1 - p_cal)).reshape(-1, 1)
                lr.fit(p_cal_logit, y_cal)
                
                p_ev_logit = np.log(p_ev / (1 - p_ev)).reshape(-1, 1)
                p_platt = lr.predict_proba(p_ev_logit)[:, 1]
                briers_platt.append(brier_score_loss(y_ev, p_platt))
                
                # Isotonic Regression
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_cal, y_cal)
                p_iso = iso.predict(p_ev)
                briers_iso.append(brier_score_loss(y_ev, p_iso))
                
            results.append({
                "model": m_name,
                "N": N,
                "brier_raw_mean": np.mean(briers_raw),
                "brier_platt_mean": np.mean(briers_platt),
                "brier_iso_mean": np.mean(briers_iso),
                "brier_platt_std": np.std(briers_platt),
                "brier_iso_std": np.std(briers_iso)
            })
            
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/tables/table_calibrator_sample_efficiency.csv", index=False)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, m_name in enumerate(["Deep (EffB0)", "Hybrid (ABC+Emb)"]):
        ax = axes[i]
        d = df_res[df_res["model"] == m_name]
        
        ax.axhline(d["brier_raw_mean"].iloc[0], color="red", linestyle="--", label="Raw Predictions")
        ax.errorbar(d["N"], d["brier_platt_mean"], yerr=d["brier_platt_std"], fmt='-o', label="Platt Scaling")
        ax.errorbar(d["N"], d["brier_iso_mean"], yerr=d["brier_iso_std"], fmt='-s', label="Isotonic Regression")
        
        ax.set_title(f"{m_name} Task 3-label recalibration")
        ax.set_xlabel("Calibration samples (N)")
        ax.set_ylabel("Brier Score (Lower is Better)")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    plt.savefig("results/figures/fig_calibrator_sample_efficiency.png", dpi=200)
    print("Saved results/figures/fig_calibrator_sample_efficiency.png")
