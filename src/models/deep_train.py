import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

from src.utils.seed import set_seed
from src.utils.io import ensure_dir, read_json
from src.data.ham import load_ham_metadata, image_path
from src.data.transforms import build_transforms
from src.evaluation.metrics import compute_basic, threshold_at_specificity
from src.evaluation.bootstrap import bootstrap_ci

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

def train_deep(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = "results/runs/deep_baseline"
    ensure_dir(run_dir)

    df = load_ham_metadata(cfg["paths"]["ham_metadata"])
    split = read_json(cfg["derived"]["split_json"])
    train_ids, val_ids, test_ids = set(split["train"]), set(split["val"]), set(split["test"])

    df_train = df[df["lesion_id"].isin(train_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()
    df_val   = df[df["lesion_id"].isin(val_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()
    df_test  = df[df["lesion_id"].isin(test_ids)][["image_id","label","lesion_id"]].drop_duplicates().copy()

    tfm_train = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=True)
    tfm_eval  = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    bs = int(cfg["training"]["deep"]["batch_size"])
    nw = int(cfg["training"]["deep"].get("num_workers", 2))

    tr = DataLoader(ImageDataset(df_train, cfg["paths"]["ham_images"], tfm_train), batch_size=bs, shuffle=True, num_workers=nw)
    va = DataLoader(ImageDataset(df_val,   cfg["paths"]["ham_images"], tfm_eval),  batch_size=bs, shuffle=False, num_workers=nw)
    te = DataLoader(ImageDataset(df_test,  cfg["paths"]["ham_images"], tfm_eval),  batch_size=bs, shuffle=False, num_workers=nw)

    model = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=True, num_classes=1).to(device)

    pos = df_train["label"].sum()
    neg = len(df_train) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-8)], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["deep"]["lr"]), weight_decay=float(cfg["training"]["deep"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"]["deep"].get("amp", True)) and device=="cuda")

    best_auc = -1.0
    best_path = cfg["derived"]["deep_ckpt"]
    ensure_dir(os.path.dirname(best_path))

    for epoch in range(int(cfg["training"]["deep"]["epochs"])):
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
    thr90 = threshold_at_specificity(yt, pt, target_spec=0.90)

    stats05 = compute_basic(yt, pt, thr=0.5)
    stats90 = compute_basic(yt, pt, thr=thr90)

    lesions_aligned = df_test.set_index("image_id").loc[ids_te]["lesion_id"].values
    ci = {}
    for k in ["AUC","PR_AUC","Brier","F1","SENS","SPEC","ACC"]:
        lo, hi = bootstrap_ci(yt, pt, k, n=1000, seed=cfg["seed"], thr=0.5, groups=lesions_aligned)
        ci[k] = (lo, hi)

    pd.DataFrame({"image_id": ids_te, "lesion_id": lesions_aligned, "y_true": yt.astype(int), "y_prob": pt.astype(float)}).to_csv(
        os.path.join(run_dir, "ham_test_predictions_deep.csv"), index=False
    )

    summary = {
        "model": cfg["training"]["deep"]["backbone"],
        "best_val_auc": best_auc,
        "test_thr_0.5": stats05,
        "test_thr_spec90": stats90,
        "bootstrap_ci_thr_0.5": {k: {"lo": ci[k][0], "hi": ci[k][1]} for k in ci},
        "checkpoint": best_path
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", os.path.join(run_dir, "summary.json"))
