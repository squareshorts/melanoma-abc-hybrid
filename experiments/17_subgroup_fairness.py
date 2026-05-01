import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from src.utils.io import load_config
from src.data.ham import load_ham_metadata

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Load metadata
    df_meta = load_ham_metadata(os.path.join(cfg["data_root"], cfg["raw"]["ham"]["metadata"]))
    # Clean age
    df_meta["age"] = pd.to_numeric(df_meta.get("age", np.nan), errors="coerce")
    
    # Load Predictions
    df_deep = pd.read_csv("results/runs/deep_baseline/ham_test_predictions_deep.csv")
    df_hybrid = pd.read_csv("results/runs/hybrid/ham_test_predictions_hybrid.csv")
    
    # Join
    df_meta = df_meta[["image_id", "sex", "age", "localization"]].drop_duplicates()
    df_deep = df_deep.merge(df_meta, on="image_id", how="left")
    df_hybrid = df_hybrid.merge(df_meta, on="image_id", how="left")
    
    subgroups = {
        "Sex: Male": df_deep["sex"] == "male",
        "Sex: Female": df_deep["sex"] == "female",
        "Age < 60": df_deep["age"] < 60,
        "Age >= 60": df_deep["age"] >= 60,
    }
    
    results = []
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, m_name in enumerate(["Deep (EffB0)", "Hybrid (ABC+Emb)"]):
        ax = axes[i]
        df = df_deep if m_name == "Deep (EffB0)" else df_hybrid
        
        ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        
        for sg_name, mask in subgroups.items():
            if m_name != "Deep (EffB0)": # Re-eval mask for hybrid just in case 
                # (since image order is the same, mask is same, but let's be safe)
                if sg_name == "Sex: Male": mask = df["sex"] == "male"
                elif sg_name == "Sex: Female": mask = df["sex"] == "female"
                elif sg_name == "Age < 60": mask = df["age"] < 60
                elif sg_name == "Age >= 60": mask = df["age"] >= 60
            
            sub_df = df[mask]
            brier = brier_score_loss(sub_df["y_true"], sub_df["y_prob"])
            
            # Calibration curve
            fop, mpv = calibration_curve(sub_df["y_true"], sub_df["y_prob"], n_bins=10, strategy="quantile")
            ax.plot(mpv, fop, "s-", label=f"{sg_name} (Brier: {brier:.3f})")
            
            results.append({
                "model": m_name,
                "subgroup": sg_name,
                "N": len(sub_df),
                "brier": brier,
                "roc_auc": np.nan if len(np.unique(sub_df["y_true"])) < 2 else __import__('sklearn.metrics').metrics.roc_auc_score(sub_df["y_true"], sub_df["y_prob"])
            })
            
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    pd.DataFrame(results).to_csv("results/tables/table_subgroup_fairness.csv", index=False)
    plt.tight_layout()
    plt.savefig("results/figures/fig_subgroup_fairness.png", dpi=200)
    print("Saved results/figures/fig_subgroup_fairness.png")
