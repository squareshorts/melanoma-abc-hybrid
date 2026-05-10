import os
import time
import numpy as np
import pandas as pd
import joblib

from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.models.xgb_models import train_xgb, predict_proba
from src.evaluation.metrics import compute_basic

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def load_embeddings(npy_path, ids_csv):
    E = np.load(npy_path)
    ids = pd.read_csv(ids_csv)["image_id"].tolist()
    return E, ids

def align(df_ids, E, ids):
    idx = {k:i for i,k in enumerate(ids)}
    keep = [idx[i] for i in df_ids["image_id"].tolist() if i in idx]
    df2 = df_ids[df_ids["image_id"].isin(idx.keys())].copy()
    E2 = E[keep]
    df2 = df2.set_index("image_id").loc[[ids[k] for k in keep]].reset_index()
    return df2, E2

if __name__ == "__main__":
    t0 = time.time()
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/analyses", exist_ok=True)
    
    print("Starting Additional Analyses for AIM Resubmission...")
    
    # ---------------------------------------------------------
    # 1. External Ablation (Embeddings-only on BCN20000)
    # ---------------------------------------------------------
    print("\n--- 1. External Ablation (Embeddings-only on BCN20000) ---")
    
    # Train the embeddings-only model on HAM10000
    split = read_json(cfg["derived"]["split_json"])
    df_meta = load_ham_metadata(cfg["paths"]["ham_metadata"])
    train_ids = set(split["train"])
    df_train = df_meta[df_meta["lesion_id"].isin(train_ids)][["image_id","label"]].drop_duplicates()
    
    E_ham, ids_ham = load_embeddings(cfg["derived"]["ham_emb_npy"], cfg["derived"]["ham_emb_ids"])
    df_train_al, Etr = align(df_train, E_ham, ids_ham)
    ytr = df_train_al["label"].values
    
    print("Training embeddings-only XGBoost on HAM10000...")
    m_noabc = train_xgb(Etr, ytr, cfg["xgb"])
    
    # Predict on BCN20000
    bcn_gt_path = "data/ISIC_2019_Training_GroundTruth.csv"
    df_gt = pd.read_csv(bcn_gt_path)
    df_gt["label"] = df_gt["MEL"].astype(int)
    valid_ids = pd.read_csv("data/splits/bcn20000_valid_ids.csv")["image_id"].tolist()
    df_bcn = df_gt[df_gt["image"].isin(valid_ids)].copy().rename(columns={"image": "image_id"})
    
    E_bcn, ids_bcn = load_embeddings("data/derived/embeddings/bcn20000_effb0.npy", "data/derived/embeddings/bcn20000_effb0_ids.csv")
    df_bcn_al, Ete_bcn = align(df_bcn[["image_id", "label"]], E_bcn, ids_bcn)
    yte_bcn = df_bcn_al["label"].values
    
    print("Evaluating on BCN20000...")
    p_noabc_bcn = predict_proba(m_noabc, Ete_bcn)
    stats_noabc_bcn = compute_basic(yte_bcn, p_noabc_bcn)
    
    print("Embeddings-only XGBoost on BCN20000:")
    print(f"AUC: {stats_noabc_bcn['AUC']:.4f}, PR-AUC: {stats_noabc_bcn['PR_AUC']:.4f}, Brier: {stats_noabc_bcn['Brier']:.4f}")
    
    # Compare with Full Hybrid (loaded from saved predictions)
    df_hybrid = pd.read_csv("results/runs/hybrid/bcn20000_predictions_hybrid.csv")
    stats_hybrid = compute_basic(df_hybrid["y_true"].values, df_hybrid["y_prob"].values)
    print("Full Hybrid XGBoost on BCN20000 (for reference):")
    print(f"AUC: {stats_hybrid['AUC']:.4f}, PR-AUC: {stats_hybrid['PR_AUC']:.4f}, Brier: {stats_hybrid['Brier']:.4f}")
    
    pd.DataFrame([
        {"variant": "embeddings_only", **stats_noabc_bcn},
        {"variant": "full_hybrid", **stats_hybrid}
    ]).to_csv("results/analyses/external_ablation.csv", index=False)
    
    # ---------------------------------------------------------
    # 2. Clinical Error Profiling
    # ---------------------------------------------------------
    print("\n--- 2. Clinical Error Profiling (Hybrid on BCN20000) ---")
    df_abc = pd.read_csv("data/derived/features/bcn20000_abc.csv")
    df_err = df_hybrid.merge(df_abc, on="image_id", how="inner")
    
    # TPs, TNs, FNs, FPs (using 0.5 threshold for profiling, though specificities are often used)
    thr = 0.5
    df_err["pred"] = (df_err["y_prob"] >= thr).astype(int)
    
    fns = df_err[(df_err["y_true"] == 1) & (df_err["pred"] == 0)].sort_values("y_prob")
    fps = df_err[(df_err["y_true"] == 0) & (df_err["pred"] == 1)].sort_values("y_prob", ascending=False)
    tps = df_err[(df_err["y_true"] == 1) & (df_err["pred"] == 1)]
    tns = df_err[(df_err["y_true"] == 0) & (df_err["pred"] == 0)]
    
    print(f"Profiling {len(fns)} False Negatives and {len(fps)} False Positives.")
    
    profile_cols = ["A_asymmetry", "B_border_irreg", "C_color_mean_s", "C_color_mean_v"]
    profile_res = []
    for group_name, group_df in [("False Negatives (Missed Mel)", fns.head(50)), 
                                 ("False Positives (Overcalled)", fps.head(50)),
                                 ("True Positives (Found Mel)", tps),
                                 ("True Negatives (Correct Non-Mel)", tns)]:
        row = {"Group": group_name, "N": len(group_df)}
        for c in profile_cols:
            row[f"{c}_mean"] = group_df[c].mean()
        profile_res.append(row)
        
    df_profile = pd.DataFrame(profile_res)
    print(df_profile.to_string(index=False))
    df_profile.to_csv("results/analyses/error_profiling.csv", index=False)

    # ---------------------------------------------------------
    # 3. Out-of-Distribution (OOD) Rejection
    # ---------------------------------------------------------
    print("\n--- 3. Out-of-Distribution Rejection via Entropy ---")
    p = df_hybrid["y_prob"].values
    # Avoid log(0)
    eps = 1e-9
    p_safe = np.clip(p, eps, 1 - eps)
    entropy = -p_safe * np.log2(p_safe) - (1 - p_safe) * np.log2(1 - p_safe)
    df_hybrid["entropy"] = entropy
    
    # Reject top 5% highest entropy
    threshold_95 = np.percentile(entropy, 95)
    df_accepted = df_hybrid[df_hybrid["entropy"] <= threshold_95]
    
    stats_acc = compute_basic(df_accepted["y_true"].values, df_accepted["y_prob"].values)
    print(f"Base AUC (100% data): {stats_hybrid['AUC']:.4f}, Brier: {stats_hybrid['Brier']:.4f}")
    print(f"Rejecting top 5% most uncertain (Entropy > {threshold_95:.4f}):")
    print(f"Remaining AUC (95% data): {stats_acc['AUC']:.4f}, Brier: {stats_acc['Brier']:.4f}")
    
    pd.DataFrame([
        {"fraction": "100%", **stats_hybrid},
        {"fraction": "95%", **stats_acc}
    ]).to_csv("results/analyses/ood_rejection.csv", index=False)

    # ---------------------------------------------------------
    # 4. Fairness Disparity Metrics
    # ---------------------------------------------------------
    print("\n--- 4. Fairness Disparity Metrics (FNR) ---")
    df_meta_bcn = pd.read_csv("data/ISIC_2019_Training_Metadata.csv")
    df_fair = df_hybrid.merge(df_meta_bcn[["image", "age_approx", "sex"]], left_on="image_id", right_on="image", how="inner")
    
    def compute_fnr(df):
        mel = df[df["y_true"] == 1]
        if len(mel) == 0: return np.nan
        # FNR at threshold 0.5
        fn = np.sum(mel["y_prob"] < 0.5)
        return fn / len(mel)
        
    print("FNR by Sex:")
    sex_res = []
    for s in df_fair["sex"].dropna().unique():
        sub = df_fair[df_fair["sex"] == s]
        fnr = compute_fnr(sub)
        sex_res.append({"sex": s, "N_melanoma": len(sub[sub["y_true"]==1]), "FNR": fnr})
        print(f"  {s}: FNR = {fnr:.4f} (n_mel = {len(sub[sub['y_true']==1])})")
        
    pd.DataFrame(sex_res).to_csv("results/analyses/fairness_sex.csv", index=False)
    
    print("FNR by Age Group:")
    df_fair["age_group"] = pd.cut(df_fair["age_approx"], bins=[0, 40, 60, 100], labels=["<40", "40-60", ">60"])
    age_res = []
    for a in df_fair["age_group"].dropna().unique():
        sub = df_fair[df_fair["age_group"] == a]
        fnr = compute_fnr(sub)
        age_res.append({"age_group": a, "N_melanoma": len(sub[sub["y_true"]==1]), "FNR": fnr})
        print(f"  {a}: FNR = {fnr:.4f} (n_mel = {len(sub[sub['y_true']==1])})")
        
    pd.DataFrame(age_res).to_csv("results/analyses/fairness_age.csv", index=False)
    
    t1 = time.time()
    print(f"\nAll analyses completed in {t1 - t0:.1f} seconds. Results saved to 'results/analyses/'.")
