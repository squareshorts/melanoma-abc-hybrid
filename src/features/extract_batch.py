import os
from glob import glob
import cv2
import pandas as pd
from tqdm import tqdm
from src.features.abc import extract_abc
from src.utils.io import ensure_dir

def extract_csv(image_dir, mask_dir, out_csv):
    ensure_dir(os.path.dirname(out_csv))
    rows = []
    for p in tqdm(sorted(glob(os.path.join(image_dir, "*.*"))), desc=f"ABC -> {out_csv}"):
        image_id = os.path.splitext(os.path.basename(p))[0]
        mp = os.path.join(mask_dir, f"{image_id}.png")
        if not os.path.exists(mp):
            continue
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        feats = extract_abc(rgb, mask)
        feats["image_id"] = image_id
        rows.append(feats)
    df = pd.DataFrame(rows).sort_values("image_id")
    df.to_csv(out_csv, index=False)
    return df
