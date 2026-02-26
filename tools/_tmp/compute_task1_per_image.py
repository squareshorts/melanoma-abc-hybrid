import pandas as pd
import numpy as np
from pathlib import Path
import cv2

gt_dir = Path(r"data/raw/ISIC2018/Task1/masks")

pred_dirs = {
    "LevelSet": Path(r"data\derived\segmentations\isic_task1_levelset"),
    "OtsuBaseline": Path(r"data\derived\segmentations\isic_task1_baseline"),
}

def bin_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)

def dice_iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a,b).sum()
    sa = a.sum(); sb = b.sum()
    dice = (2*inter) / (sa + sb + 1e-9)
    union = np.logical_or(a,b).sum()
    iou = inter / (union + 1e-9)
    return float(dice), float(iou)

rows = []
gt_files = sorted(gt_dir.glob("*.png"))
gt_index = {f.stem: f for f in gt_files}

for method, pdir in pred_dirs.items():
    if not pdir.exists():
        raise SystemExit(f"Pred dir missing: {pdir}")
    for stem, gt_path in gt_index.items():
        pred_path = pdir / f"{stem}.png"
        if not pred_path.exists():
            rows.append({"image_id": stem, "method": method, "dice": np.nan, "iou": np.nan, "missing_pred": 1})
            continue
        gt = bin_mask(gt_path)
        pr = bin_mask(pred_path)
        if gt is None or pr is None:
            rows.append({"image_id": stem, "method": method, "dice": np.nan, "iou": np.nan, "missing_pred": 1})
            continue
        d, j = dice_iou(pr, gt)
        rows.append({"image_id": stem, "method": method, "dice": d, "iou": j, "missing_pred": 0})

df = pd.DataFrame(rows)
out = Path("results/audit/isic_task1_per_image.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("Wrote", out, "rows=", len(df), "missing_pred=", int(df["missing_pred"].sum()))
