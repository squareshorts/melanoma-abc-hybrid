import os
import numpy as np
import pandas as pd
import cv2
import joblib
import torch
from PIL import Image
import timm

from src.utils.io import load_config
from src.data.transforms import build_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.shap_xgb import shap_values_tree

def _find_image(images_dir, image_id):
    for ext in [".jpg",".jpeg",".png"]:
        p = os.path.join(images_dir, image_id + ext)
        if os.path.exists(p):
            return p
    return None

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/figures", exist_ok=True)

    # SHAP for ABC-only model (clean interpretability)
    import xgboost as xgb
    feats = pd.read_csv(cfg["derived"]["ham_abc_csv"])
    pred = pd.read_csv("results/runs/deep_baseline/ham_test_predictions_deep.csv")
    df = pred.merge(feats, on="image_id", how="inner")
    feat_cols = [c for c in df.columns if c.startswith("A_") or c.startswith("B_") or c.startswith("C_")]
    X = df[feat_cols].values
    y = df["y_true"].values

    m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                          eval_metric="logloss", n_jobs=-1)
    m.fit(X, y)

    sv, _ = shap_values_tree(m, X)
    import shap
    shap.summary_plot(sv, features=X, feature_names=feat_cols, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig("results/figures/fig_shap_summary.png", dpi=200)
    plt.close()

    # Grad-CAM for deep baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deep = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=False, num_classes=1).to(device)
    ckpt = torch.load(cfg["derived"]["deep_ckpt"], map_location=device)
    deep.load_state_dict(ckpt["model"])
    deep.eval()

    target_layer = None
    for name, mod in deep.named_modules():
        if "conv_head" in name:
            target_layer = mod
    if target_layer is None:
        target_layer = list(deep.modules())[-1]

    cam_fn = GradCAM(deep, target_layer)
    tfm = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)

    dfp = pd.read_csv("results/runs/deep_baseline/ham_test_predictions_deep.csv")
    dfp["y_pred"] = (dfp["y_prob"] >= 0.5).astype(int)
    tp = dfp[(dfp.y_true==1)&(dfp.y_pred==1)].sort_values("y_prob", ascending=False).head(2)
    tn = dfp[(dfp.y_true==0)&(dfp.y_pred==0)].sort_values("y_prob", ascending=True).head(2)
    fp = dfp[(dfp.y_true==0)&(dfp.y_pred==1)].sort_values("y_prob", ascending=False).head(1)
    fn = dfp[(dfp.y_true==1)&(dfp.y_pred==0)].sort_values("y_prob", ascending=True).head(1)
    sel = pd.concat([tp, tn, fp, fn]).drop_duplicates("image_id").head(6)

    fig, axes = plt.subplots(len(sel), 3, figsize=(10, 3*len(sel)))
    if len(sel) == 1:
        axes = np.expand_dims(axes, 0)

    for i, r in enumerate(sel.itertuples(index=False)):
        img_id = r.image_id
        p = _find_image(cfg["paths"]["ham_images"], img_id)
        if p is None:
            continue
        rgb = np.array(Image.open(p).convert("RGB").resize((cfg["image"]["size"], cfg["image"]["size"])))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        cam = cam_fn(x)

        axes[i,0].imshow(rgb); axes[i,0].axis("off"); axes[i,0].set_title("Image")
        axes[i,1].imshow(cam, cmap="jet"); axes[i,1].axis("off"); axes[i,1].set_title("Grad-CAM")
        axes[i,2].axis("off"); axes[i,2].text(0, 0.9, f"GT={r.y_true}\nP={r.y_prob:.3f}", fontsize=12, va="top")

    plt.tight_layout()
    plt.savefig("results/figures/fig_gradcam_cases.png", dpi=200)
    plt.close()

    print("Wrote: results/figures/fig_shap_summary.png and fig_gradcam_cases.png")
