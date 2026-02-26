import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import joblib

from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata, image_path
from src.data.transforms import build_transforms


SEED = 42


def _to_float01(y):
    y = pd.Series(y).astype(float).values
    return (y > 0.5).astype(int)


def calibration_slope_intercept(y, p):
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


def write_preds_csv(image_ids, y, p, out_path):
    df = pd.DataFrame({"image_id": image_ids, "y_true": y.astype(int), "y_prob": p.astype(float)})
    df.to_csv(out_path, index=False)


# -------------------------
# Deep: predict on images
# -------------------------

class ImageDataset(Dataset):
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
def predict_deep(model, loader, device):
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


def recalibrate_from_val_to_test(tag, ids_val, y_val, p_val, ids_test, y_test, p_test):
    rows = []

    raw = evaluate(f"{tag}_raw", y_test, p_test)
    rows.append(raw)

    # Platt
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p_val.reshape(-1, 1), y_val)
    p_test_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]
    rows.append(evaluate(f"{tag}_platt", y_test, p_test_platt))

    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    p_test_iso = iso.transform(p_test)
    rows.append(evaluate(f"{tag}_isotonic", y_test, p_test_iso))

    out = pd.DataFrame(rows)
    for col in ["AUC", "Brier", "calibration_slope", "calibration_intercept"]:
        out[f"d{col}_vs_raw"] = out[col] - raw[col]

    preds = {
        "raw": (ids_test, y_test, p_test),
        "platt": (ids_test, y_test, p_test_platt),
        "isotonic": (ids_test, y_test, p_test_iso),
    }
    return out, preds


# -------------------------
# Hybrid: build X from ABC + embeddings
# -------------------------

def load_embeddings(npy_path, ids_csv):
    E = np.load(npy_path)
    ids = pd.read_csv(ids_csv)["image_id"].tolist()
    return E, ids


def align_ids(df_ids, E, ids):
    idx = {k: i for i, k in enumerate(ids)}
    keep = [idx[i] for i in df_ids["image_id"].tolist() if i in idx]
    df2 = df_ids[df_ids["image_id"].isin(idx.keys())].copy()
    E2 = E[keep]
    # enforce exact same order as E2
    df2 = df2.set_index("image_id").loc[[ids[k] for k in keep]].reset_index()
    return df2, E2


def main():
    cfg = load_config()
    os.makedirs("results/audit", exist_ok=True)

    df_meta = load_ham_metadata(cfg["paths"]["ham_metadata"])
    split = read_json(cfg["derived"]["split_json"])

    val_lesions = set(split["val"])
    test_lesions = set(split["test"])

    df_val = df_meta[df_meta["lesion_id"].isin(val_lesions)][["image_id", "label"]].drop_duplicates().copy()
    df_test = df_meta[df_meta["lesion_id"].isin(test_lesions)][["image_id", "label"]].drop_duplicates().copy()

    # -------------------------
    # DEEP
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm_eval = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    val_loader = DataLoader(ImageDataset(df_val, cfg["paths"]["ham_images"], tfm_eval),
                            batch_size=32, shuffle=False, num_workers=int(cfg["training"]["deep"].get("num_workers", 2)))
    test_loader = DataLoader(ImageDataset(df_test, cfg["paths"]["ham_images"], tfm_eval),
                             batch_size=32, shuffle=False, num_workers=int(cfg["training"]["deep"].get("num_workers", 2)))

    deep = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=False, num_classes=1).to(device)
    ckpt = torch.load(cfg["derived"]["deep_ckpt"], map_location=device)
    deep.load_state_dict(ckpt["model"])

    deep_ids_val, deep_y_val, deep_p_val = predict_deep(deep, val_loader, device)
    deep_ids_test, deep_y_test, deep_p_test = predict_deep(deep, test_loader, device)

    deep_out, deep_preds = recalibrate_from_val_to_test(
        "deep",
        deep_ids_val, _to_float01(deep_y_val), deep_p_val,
        deep_ids_test, _to_float01(deep_y_test), deep_p_test
    )
    deep_out.to_csv("results/audit/recalibration_ham_deep.csv", index=False)
    for method, (ids, y, p) in deep_preds.items():
        write_preds_csv(ids, y, p, f"results/audit/preds_ham_deep_{method}.csv")

    # -------------------------
    # HYBRID
    # -------------------------
    # Load hybrid model
    blob = joblib.load("results/runs/hybrid/hybrid_xgb.joblib")
    model = blob["model"]
    feat_cols = blob["feat_cols"]

    feats = pd.read_csv(cfg["derived"]["ham_abc_csv"])

    # Join labels + ABC
    df_val_h = df_val.merge(feats, on="image_id", how="inner")
    df_test_h = df_test.merge(feats, on="image_id", how="inner")

    # Embeddings and alignment
    E_ham, ids_ham = load_embeddings(cfg["derived"]["ham_emb_npy"], cfg["derived"]["ham_emb_ids"])

    df_val_al, E_val = align_ids(df_val_h[["image_id"]], E_ham, ids_ham)
    df_test_al, E_test = align_ids(df_test_h[["image_id"]], E_ham, ids_ham)

    df_val_h = df_val_h.set_index("image_id").loc[df_val_al["image_id"]].reset_index()
    df_test_h = df_test_h.set_index("image_id").loc[df_test_al["image_id"]].reset_index()

    X_val_abc = df_val_h[feat_cols].values
    y_val_h = df_val_h["label"].values.astype(int)

    X_test_abc = df_test_h[feat_cols].values
    y_test_h = df_test_h["label"].values.astype(int)

    X_val = np.concatenate([X_val_abc, E_val], axis=1)
    X_test = np.concatenate([X_test_abc, E_test], axis=1)

    # Predict (XGBoost)
    p_val_h = model.predict_proba(X_val)[:, 1]
    p_test_h = model.predict_proba(X_test)[:, 1]

    hyb_ids_val = df_val_h["image_id"].tolist()
    hyb_ids_test = df_test_h["image_id"].tolist()

    hyb_out, hyb_preds = recalibrate_from_val_to_test(
        "hybrid",
        hyb_ids_val, y_val_h, p_val_h,
        hyb_ids_test, y_test_h, p_test_h
    )
    hyb_out.to_csv("results/audit/recalibration_ham_hybrid.csv", index=False)
    for method, (ids, y, p) in hyb_preds.items():
        write_preds_csv(ids, y, p, f"results/audit/preds_ham_hybrid_{method}.csv")

    # Print to console
    print("\nDEEP (HAM)\n", deep_out)
    print("\nHYBRID (HAM)\n", hyb_out)
    print("\nWrote:")
    print("  results/audit/recalibration_ham_deep.csv")
    print("  results/audit/recalibration_ham_hybrid.csv")
    print("  results/audit/preds_ham_deep_{raw,platt,isotonic}.csv")
    print("  results/audit/preds_ham_hybrid_{raw,platt,isotonic}.csv")


if __name__ == "__main__":
    main()