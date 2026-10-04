"""Generate BCN20000 prediction files for the three fitted model families.

This script only generates predictions. Current external inference, lesion-clustered
confidence intervals, and source-derived operating thresholds are implemented in
`23_final_external_audit.py`.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.xgb_models import predict_proba
from src.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]


def load_external_labels():
    gt_path = ROOT / "data" / "ISIC_2019_Training_GroundTruth.csv"
    valid_path = ROOT / "data" / "splits" / "bcn20000_valid_ids.csv"
    if not gt_path.exists():
        raise FileNotFoundError(
            "Missing data/ISIC_2019_Training_GroundTruth.csv required to reproduce the evaluated BCN subset."
        )
    if not valid_path.exists():
        raise FileNotFoundError(
            "Missing data/splits/bcn20000_valid_ids.csv. Run the overlap audit first."
        )

    gt = pd.read_csv(gt_path)
    gt["label"] = gt["MEL"].astype(int)
    valid = set(pd.read_csv(valid_path)["image_id"].astype(str))
    return gt[gt["image"].astype(str).isin(valid)].rename(columns={"image": "image_id"})


def main():
    cfg = load_config()
    df_bcn = load_external_labels()
    image_dir = Path(cfg["data_root"]) / cfg["raw"]["bcn20000"]["images"]

    out_hand = ROOT / "results/runs/handcrafted"
    out_hyb = ROOT / "results/runs/hybrid"
    out_deep = ROOT / "results/runs/deep_baseline"
    for p in (out_hand, out_hyb, out_deep):
        p.mkdir(parents=True, exist_ok=True)

    feat_path = ROOT / cfg["derived"]["bcn20000_abc_csv"]
    emb_path = ROOT / cfg["derived"]["bcn20000_emb_npy"]
    emb_ids_path = ROOT / cfg["derived"]["bcn20000_emb_ids"]

    feats = pd.read_csv(feat_path)

    # Handcrafted XGBoost.
    pack = joblib.load(ROOT / "results/runs/handcrafted/handcrafted_xgb.joblib")
    model = pack["model"]
    feat_cols = pack["feat_cols"]
    d = df_bcn.merge(feats, on="image_id", how="inner")
    p = predict_proba(model, d[feat_cols].to_numpy())
    pd.DataFrame({
        "image_id": d["image_id"],
        "y_true": d["label"].astype(int),
        "y_prob": p,
    }).to_csv(out_hand / "bcn20000_predictions_handcrafted.csv", index=False)

    # Hybrid XGBoost.
    pack = joblib.load(ROOT / "results/runs/hybrid/hybrid_xgb.joblib")
    model = pack["model"]
    feat_cols = pack["feat_cols"]
    d = df_bcn.merge(feats, on="image_id", how="inner")
    emb = np.load(emb_path)
    emb_ids = pd.read_csv(emb_ids_path)["image_id"].astype(str).tolist()
    emap = {k: i for i, k in enumerate(emb_ids)}
    d = d[d["image_id"].astype(str).isin(emap)].reset_index(drop=True)
    e = emb[[emap[str(i)] for i in d["image_id"]]]
    x = np.concatenate([d[feat_cols].to_numpy(), e], axis=1)
    p = predict_proba(model, x)
    pd.DataFrame({
        "image_id": d["image_id"],
        "y_true": d["label"].astype(int),
        "y_prob": p,
    }).to_csv(out_hyb / "bcn20000_predictions_hybrid.csv", index=False)

    # End-to-end EfficientNet-B0.
    import timm
    from src.data.transforms import build_transforms

    spec = importlib.util.spec_from_file_location(
        "external_validation", ROOT / "experiments/09_external_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = timm.create_model(
        cfg["training"]["deep"]["backbone"], pretrained=False, num_classes=1
    ).to(device)
    ckpt = torch.load(ROOT / cfg["derived"]["deep_ckpt"], map_location=device)
    net.load_state_dict(ckpt["model"])

    tfm = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)
    ds = module.ISICDataset(df_bcn, str(image_dir), tfm)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    ids, y, p = module._predict(net, dl, device)

    pd.DataFrame({
        "image_id": ids,
        "y_true": y,
        "y_prob": p,
    }).to_csv(out_deep / "bcn20000_predictions_deep.csv", index=False)

    print(f"Generated BCN20000 predictions for {len(df_bcn):,} evaluated images.")
    print("Run experiments/23_final_external_audit.py for current external inference.")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
