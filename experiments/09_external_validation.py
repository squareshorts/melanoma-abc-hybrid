import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import joblib

from src.utils.io import load_config
from src.data.isic import load_isic_task3_labels
from src.models.xgb_models import predict_proba
from src.evaluation.metrics import compute_basic
from src.evaluation.bootstrap import bootstrap_ci
from src.data.transforms import build_transforms
from src.data.ham import image_path


class ISICDataset(Dataset):
    def __init__(self, df, images_dir, tfm):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        y = int(row["label"])
        p = image_path(self.images_dir, img_id)
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, torch.tensor([y], dtype=torch.float32), img_id


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    ids, ys, ps = [], [], []
    for x, y, img_id in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy().reshape(-1))
        ids += list(img_id)
    return ids, np.concatenate(ys), np.concatenate(ps)


if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)

    df_isic = load_isic_task3_labels(cfg["paths"]["isic_task3_labels"])

    # ---------------------------
    # HYBRID EVALUATION
    # ---------------------------
    pack = joblib.load("results/runs/hybrid/hybrid_xgb.joblib")
    model_hybrid = pack["model"]
    feat_cols = pack["feat_cols"]

    feats_isic = pd.read_csv(cfg["derived"]["isic_task3_abc_csv"])
    df = df_isic.merge(feats_isic, on="image_id", how="inner")

    E = np.load(cfg["derived"]["isic_task3_emb_npy"])
    ids_emb = pd.read_csv(cfg["derived"]["isic_task3_emb_ids"])["image_id"].tolist()
    idx = {k: i for i, k in enumerate(ids_emb)}
    keep = [idx[i] for i in df["image_id"].tolist() if i in idx]
    df = df[df["image_id"].isin(idx.keys())].reset_index(drop=True)
    E = E[keep]

    Xabc = df[feat_cols].values
    X = np.concatenate([Xabc, E], axis=1)
    y = df["label"].values
    p_hybrid = predict_proba(model_hybrid, X)

    stats_hybrid = compute_basic(y, p_hybrid, thr=0.5)

    pd.DataFrame({
        "image_id": df["image_id"],
        "y_true": y,
        "y_prob": p_hybrid
    }).to_csv("results/runs/hybrid/isic_task3_predictions_hybrid.csv", index=False)

    # ---------------------------
    # DEEP BASELINE EVALUATION
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_deep = timm.create_model(
        cfg["training"]["deep"]["backbone"],
        pretrained=False,
        num_classes=1
    ).to(device)

    ckpt = torch.load(cfg["derived"]["deep_ckpt"], map_location=device)
    model_deep.load_state_dict(ckpt["model"])

    tfm_eval = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    ds = ISICDataset(df_isic, cfg["paths"]["isic_task3_images"], tfm_eval)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    ids_deep, y_deep, p_deep = _predict(model_deep, dl, device)

    stats_deep = compute_basic(y_deep, p_deep, thr=0.5)

    pd.DataFrame({
        "image_id": ids_deep,
        "y_true": y_deep,
        "y_prob": p_deep
    }).to_csv("results/runs/deep_baseline/isic_task3_predictions_deep.csv", index=False)

    # ---------------------------
    # SAVE SUMMARY TABLE
    # ---------------------------
    rows = []

    for name, stats in [
        ("hybrid_external_isic_task3", stats_hybrid),
        ("deep_external_isic_task3", stats_deep)
    ]:
        row = {"model": name, **stats}
        for k in ["AUC","PR_AUC","F1","SENS","SPEC","ACC"]:
            lo, hi = bootstrap_ci(
                y_deep if "deep" in name else y,
                p_deep if "deep" in name else p_hybrid,
                k,
                n=1000,
                seed=cfg["seed"],
                thr=0.5
            )
            row[f"{k}_CI95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)

    pd.DataFrame(rows).to_csv("results/tables/table_external_isic_task3.csv", index=False)

    print("Hybrid:", stats_hybrid)
    print("Deep:", stats_deep)