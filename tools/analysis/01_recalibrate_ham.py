import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import timm
from PIL import Image

from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata, image_path
from src.data.transforms import build_transforms


# -------------------------
# Dataset
# -------------------------

class ImageDataset(torch.utils.data.Dataset):
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


# -------------------------
# Prediction
# -------------------------

@torch.no_grad()
def predict_deep(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()
        ps.append(prob)
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


# -------------------------
# Calibration slope & intercept
# -------------------------

def calibration_slope_intercept(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit, y)

    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return slope, intercept


# -------------------------
# Main
# -------------------------

def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_meta = load_ham_metadata(cfg["paths"]["ham_metadata"])
    split = read_json(cfg["derived"]["split_json"])

    val_ids = set(split["val"])
    test_ids = set(split["test"])

    df_val = df_meta[df_meta["lesion_id"].isin(val_ids)][["image_id", "label"]].drop_duplicates()
    df_test = df_meta[df_meta["lesion_id"].isin(test_ids)][["image_id", "label"]].drop_duplicates()

    tfm_eval = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    val_loader = DataLoader(
        ImageDataset(df_val, cfg["paths"]["ham_images"], tfm_eval),
        batch_size=32, shuffle=False, num_workers=2
    )

    test_loader = DataLoader(
        ImageDataset(df_test, cfg["paths"]["ham_images"], tfm_eval),
        batch_size=32, shuffle=False, num_workers=2
    )

    # Load trained deep model
    model = timm.create_model(
        cfg["training"]["deep"]["backbone"],
        pretrained=False,
        num_classes=1
    ).to(device)

    ckpt = torch.load(cfg["derived"]["deep_ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"])

    # Predictions
    y_val, p_val = predict_deep(model, val_loader, device)
    y_test, p_test = predict_deep(model, test_loader, device)

    # -------------------------
    # Platt scaling
    # -------------------------
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p_val.reshape(-1, 1), y_val)
    p_test_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]

    # -------------------------
    # Isotonic regression
    # -------------------------
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    p_test_iso = iso.transform(p_test)

    rows = []

    def record(name, y, p):
        slope, intercept = calibration_slope_intercept(y, p)

        rows.append({
            "model": name,
            "AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
            "calibration_slope": slope,
            "calibration_intercept": intercept
        })

    record("deep_raw", y_test, p_test)
    record("deep_platt", y_test, p_test_platt)
    record("deep_isotonic", y_test, p_test_iso)

    out = pd.DataFrame(rows)

    os.makedirs("results/audit", exist_ok=True)
    out.to_csv("results/audit/recalibration_ham.csv", index=False)

    print(out)


if __name__ == "__main__":
    main()