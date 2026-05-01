import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import brier_score_loss

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, read_json
from src.data.ham import load_ham_metadata
from src.data.transforms import build_transforms
from src.evaluation.metrics import compute_basic

from src.models.deep_train import ImageDataset, _predict

def train_ablation(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = "results/runs/deep_ablation"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    df = load_ham_metadata(os.path.join(cfg["data_root"], cfg["raw"]["ham"]["metadata"]))
    split = read_json(cfg["derived"]["split_json"])
    train_ids, val_ids, test_ids = set(split["train"]), set(split["val"]), set(split["test"])

    df_train = df[df["lesion_id"].isin(train_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()
    df_val   = df[df["lesion_id"].isin(val_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()
    df_test  = df[df["lesion_id"].isin(test_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()

    tfm_train = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=True)
    tfm_eval  = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    bs = int(cfg["training"]["deep"]["batch_size"])
    nw = int(cfg["training"]["deep"].get("num_workers", 2))

    ham_img_dir = os.path.join(cfg["data_root"], cfg["raw"]["ham"]["images"])
    tr = DataLoader(ImageDataset(df_train, ham_img_dir, tfm_train), batch_size=bs, shuffle=True, num_workers=nw)
    va = DataLoader(ImageDataset(df_val,   ham_img_dir, tfm_eval),  batch_size=bs, shuffle=False, num_workers=nw)
    te = DataLoader(ImageDataset(df_test,  ham_img_dir, tfm_eval),  batch_size=bs, shuffle=False, num_workers=nw)

    model = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=True, num_classes=1).to(device)

    # ABLATION: No pos_weight
    criterion = nn.BCEWithLogitsLoss()
    
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["deep"]["lr"]), weight_decay=float(cfg["training"]["deep"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["deep"].get("amp", True)) and device=="cuda")

    best_auc = -1.0
    best_path = os.path.join(run_dir, "ablation_best.pt")
    
    epochs = int(cfg["training"]["deep"]["epochs"])
    print(f"Training Unweighted Deep Baseline for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        _, yv, pv = _predict(model, va, device)
        auc = compute_basic(yv, pv, thr=0.5)["AUC"]
        if np.isfinite(auc) and auc > best_auc:
            best_auc = float(auc)
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)

        print(f"Epoch {epoch+1}: loss={np.mean(losses):.4f} val_auc={auc:.4f} best={best_auc:.4f}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    ids_te, yt, pt = _predict(model, te, device)
    
    df_pred = pd.DataFrame({
        "image_id": ids_te, 
        "y_true": yt.astype(int), 
        "y_prob": pt.astype(float)
    })
    df_pred.to_csv(os.path.join(run_dir, "ham_test_predictions_ablation.csv"), index=False)
    
    from sklearn.linear_model import LogisticRegression
    df_base = pd.read_csv("results/runs/deep_baseline/ham_test_predictions_deep.csv")
    
    def get_calib_params(y_true, y_prob):
        y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
        logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", C=1e5)
        lr.fit(logit, y_true)
        return lr.intercept_[0], lr.coef_[0][0]
        
    a_base, b_base = get_calib_params(df_base["y_true"], df_base["y_prob"])
    a_abl, b_abl = get_calib_params(df_pred["y_true"], df_pred["y_prob"])
    
    print("\nCalibration Comparison:")
    print(f"Weighted Baseline:   intercept={a_base:.3f}, slope={b_base:.3f}, Brier={brier_score_loss(df_base['y_true'], df_base['y_prob']):.3f}")
    print(f"Unweighted Ablation: intercept={a_abl:.3f}, slope={b_abl:.3f}, Brier={brier_score_loss(df_pred['y_true'], df_pred['y_prob']):.3f}")
    
    pd.DataFrame([{
        "model": "Weighted Baseline", "intercept": a_base, "slope": b_base, "brier": brier_score_loss(df_base["y_true"], df_base["y_prob"])
    }, {
        "model": "Unweighted Ablation", "intercept": a_abl, "slope": b_abl, "brier": brier_score_loss(df_pred["y_true"], df_pred["y_prob"])
    }]).to_csv("results/tables/table_loss_ablation.csv", index=False)

if __name__ == "__main__":
    from src.utils.io import load_config
    train_ablation(load_config())
