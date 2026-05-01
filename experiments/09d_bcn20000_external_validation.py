import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.io import load_config
from src.models.xgb_models import predict_proba
from src.evaluation.metrics import compute_basic
from src.evaluation.bootstrap import bootstrap_ci
from src.reporting.figures import plot_roc_multi, plot_pr_multi, plot_calibration_multi

def decision_curve(y, p, thresholds):
    n = len(y)
    net_benefits = []
    for thr in thresholds:
        tp = np.sum((p >= thr) & (y == 1))
        fp = np.sum((p >= thr) & (y == 0))
        if tp + fp == 0:
            nb = 0.0
        else:
            nb = (tp / n) - (fp / n) * (thr / (1 - thr))
        net_benefits.append(nb)
    return np.array(net_benefits)

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load Labels
    gt_path = "c:/work/melanoma-abc-hybrid/data/ISIC_2019_Training_GroundTruth.csv"
    if not os.path.exists(gt_path):
        raise SystemExit("Please download ISIC_2019_Training_GroundTruth.csv first.")
    
    df_gt = pd.read_csv(gt_path)
    # The endpoint is melanoma versus all other diagnoses
    df_gt["label"] = df_gt["MEL"].astype(int)
    
    valid_csv = "data/splits/bcn20000_valid_ids.csv"
    if os.path.exists(valid_csv):
        valid_ids = pd.read_csv(valid_csv)["image_id"].tolist()
        df_bcn = df_gt[df_gt["image"].isin(valid_ids)].copy()
    else:
        # Fallback if audit didn't run properly
        valid_ids = [f.replace(".jpg", "") for f in os.listdir("C:/work/datasets/BCN20000/images") if f.endswith(".jpg")]
        df_bcn = df_gt[df_gt["image"].isin(valid_ids)].copy()
        
    df_bcn = df_bcn.rename(columns={"image": "image_id"})
    
    rows = []

    # ---------------------------
    # ABC-ONLY EVALUATION
    # ---------------------------
    stats_abc = None
    handcrafted_path = "results/runs/handcrafted/handcrafted_xgb.joblib"
    if os.path.exists(handcrafted_path) and os.path.exists("data/derived/features/bcn20000_abc.csv"):
        pack_abc = joblib.load(handcrafted_path)
        model_abc = pack_abc["model"]
        feat_cols_abc = pack_abc["feat_cols"]

        feats_bcn = pd.read_csv("data/derived/features/bcn20000_abc.csv")
        df_abc = df_bcn.merge(feats_bcn, on="image_id", how="inner")
        
        if len(df_abc) > 0:
            Xabc_only = df_abc[feat_cols_abc].values
            y_abc = df_abc["label"].values
            p_abc = predict_proba(model_abc, Xabc_only)
            stats_abc = compute_basic(y_abc, p_abc, thr=0.5)

            pd.DataFrame({
                "image_id": df_abc["image_id"],
                "y_true": y_abc,
                "y_prob": p_abc
            }).to_csv("results/runs/handcrafted/bcn20000_predictions_handcrafted.csv", index=False)

    # ---------------------------
    # HYBRID EVALUATION
    # ---------------------------
    stats_hybrid = None
    hybrid_path = "results/runs/hybrid/hybrid_xgb.joblib"
    if os.path.exists(hybrid_path) and os.path.exists("data/derived/features/bcn20000_abc.csv") and os.path.exists("data/derived/embeddings/bcn20000_effb0.npy"):
        pack = joblib.load(hybrid_path)
        model_hybrid = pack["model"]
        feat_cols = pack["feat_cols"]

        feats_bcn = pd.read_csv("data/derived/features/bcn20000_abc.csv")
        df = df_bcn.merge(feats_bcn, on="image_id", how="inner")

        E = np.load("data/derived/embeddings/bcn20000_effb0.npy")
        ids_emb = pd.read_csv("data/derived/embeddings/bcn20000_effb0_ids.csv")["image_id"].tolist()
        idx = {k: i for i, k in enumerate(ids_emb)}
        keep = [idx[i] for i in df["image_id"].tolist() if i in idx]
        df = df[df["image_id"].isin(idx.keys())].reset_index(drop=True)
        E = E[keep]

        if len(df) > 0:
            Xabc = df[feat_cols].values
            X = np.concatenate([Xabc, E], axis=1)
            y = df["label"].values
            p_hybrid = predict_proba(model_hybrid, X)

            stats_hybrid = compute_basic(y, p_hybrid, thr=0.5)

            pd.DataFrame({
                "image_id": df["image_id"],
                "y_true": y,
                "y_prob": p_hybrid
            }).to_csv("results/runs/hybrid/bcn20000_predictions_hybrid.csv", index=False)

    # ---------------------------
    # DEEP BASELINE EVALUATION (USING PRE-EXTRACTED EMBEDDINGS + LOGITS)
    # Actually, the deep model classifier is just a linear layer.
    # We can reconstruct it, or use a prediction script. 
    # But since we just need the Deep probabilities, let's load the model fully.
    # ---------------------------
    stats_deep = None
    deep_ckpt_path = cfg["derived"]["deep_ckpt"]
    if os.path.exists(deep_ckpt_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import timm
        import importlib.util
        from src.data.transforms import build_transforms
        
        # Dynamically load the module since it starts with a number
        spec = importlib.util.spec_from_file_location("ext_val", "experiments/09_external_validation.py")
        ext_val = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext_val)
        
        ISICDataset = ext_val.ISICDataset
        _predict = ext_val._predict
        
        model_deep = timm.create_model(
            cfg["training"]["deep"]["backbone"],
            pretrained=False,
            num_classes=1
        ).to(device)

        ckpt = torch.load(deep_ckpt_path, map_location=device)
        model_deep.load_state_dict(ckpt["model"])
        
        tfm_eval = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)
        ds = ISICDataset(df_bcn, "C:/work/datasets/BCN20000/images", tfm_eval)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        
        ids_deep, y_deep, p_deep = _predict(model_deep, dl, device)
        if len(y_deep) > 0:
            stats_deep = compute_basic(y_deep, p_deep, thr=0.5)
            
            pd.DataFrame({
                "image_id": ids_deep,
                "y_true": y_deep,
                "y_prob": p_deep
            }).to_csv("results/runs/deep_baseline/bcn20000_predictions_deep.csv", index=False)

    # ---------------------------
    # SAVE SUMMARY TABLE AND PLOTS
    # ---------------------------
    model_stats = []
    if stats_abc is not None:
        model_stats.append(("abc_bcn20000", stats_abc, y_abc, p_abc))
    if stats_hybrid is not None:
        model_stats.append(("hybrid_bcn20000", stats_hybrid, y, p_hybrid))
    if stats_deep is not None:
        model_stats.append(("deep_bcn20000", stats_deep, y_deep, p_deep))

    for name, stats, yy, pp in model_stats:
        row = {"model": name, **stats}
        # Lesion-level clustering not available for BCN20000, so groups=None
        for k in ["AUC","PR_AUC","Brier","F1","SENS","SPEC","ACC"]:
            lo, hi = bootstrap_ci(
                yy, pp, k, n=1000, seed=cfg["seed"], thr=0.5, groups=None
            )
            row[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)

    if len(rows) > 0:
        out = pd.DataFrame(rows)
        out.to_csv("results/tables/table_external_bcn20000.csv", index=False)

        if stats_abc is not None: print("ABC:", stats_abc)
        if stats_hybrid is not None: print("Hybrid:", stats_hybrid)
        if stats_deep is not None: print("Deep:", stats_deep)
        
        # Plotting (multi-curves with NO titles)
        series = []
        if stats_abc is not None: series.append(("ABC", y_abc, p_abc))
        if stats_hybrid is not None: series.append(("Hybrid", y, p_hybrid))
        if stats_deep is not None: series.append(("Deep", y_deep, p_deep))
        
        plot_roc_multi(series, "results/figures/fig_curves_bcn_roc.png", title="")
        plot_pr_multi(series, "results/figures/fig_curves_bcn_pr.png", title="")
        plot_calibration_multi(series, "results/figures/fig_curves_bcn_calibration.png", title="")
        
        # Plot DCA
        fig, ax = plt.subplots(figsize=(6, 5))
        thr = np.linspace(0.01, 0.5, 50)
        
        # "Treat all" line
        # Use first label array since all have same prevalence
        y_dca = series[0][1]
        prev = np.mean(y_dca)
        treat_all = prev - (1 - prev) * (thr / (1 - thr))
        ax.plot(thr, treat_all, linestyle=":", color="black", label="Treat all")
        ax.plot(thr, np.zeros_like(thr), linestyle="-", color="black", label="Treat none")
        
        for label, y_true, y_prob in series:
            nb = decision_curve(y_true, y_prob, thr)
            ax.plot(thr, nb, label=label, linewidth=2)
            
        ax.set_ylim(-0.05, 0.2)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("results/figures/fig_dca_bcn.png", dpi=600)
        fig.savefig("results/figures/fig_dca_bcn.pdf")
        plt.close(fig)
        
        print("Generated metrics and title-free plots.")
    else:
        print("No models were evaluated. Please run feature extraction first.")
