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

def _overlay_cam(rgb, cam, alpha=0.45):
    heat = np.uint8(255 * np.clip(cam, 0, 1))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * rgb + alpha * heat)

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

    handcrafted_pkg = "results/runs/handcrafted/hybrid_xgb.joblib" # Note: shared package name or use the handcrafted one
    if not os.path.exists(handcrafted_pkg):
        handcrafted_pkg = "results/runs/handcrafted/handcrafted_xgb.joblib"
    
    if os.path.exists(handcrafted_pkg):
        pack = joblib.load(handcrafted_pkg)
        m = pack["model"]
        feat_cols = pack["feat_cols"]
    else:
        # Fallback to re-fit if not found (though less ideal)
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                              eval_metric="logloss", n_jobs=-1)
        m.fit(X, y)
        feat_cols = [c for c in df.columns if c.startswith("A_") or c.startswith("B_") or c.startswith("C_")]

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
    candidates = pd.concat([
        dfp[(dfp.y_true == 1) & (dfp.y_pred == 1)].sort_values("y_prob", ascending=False),
        dfp[(dfp.y_true == 0) & (dfp.y_pred == 1)].sort_values("y_prob", ascending=False),
    ]).drop_duplicates("image_id")

    selected = []
    for r in candidates.itertuples(index=False):
        img_id = r.image_id
        p = _find_image(cfg["paths"]["ham_images"], img_id)
        if p is None:
            continue
        rgb = np.array(Image.open(p).convert("RGB").resize((cfg["image"]["size"], cfg["image"]["size"])))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        cam = cam_fn(x)
        if float(np.ptp(cam)) < 0.05:
            continue
        selected.append((r, rgb, _overlay_cam(rgb, cam)))
        if len(selected) == 4:
            break

    if not selected:
        print("No non-degenerate Grad-CAM overlays passed the variance check; no Grad-CAM figure written.")
    else:
        fig, axes = plt.subplots(len(selected), 2, figsize=(7.5, 3.2 * len(selected)))
        if len(selected) == 1:
            axes = np.expand_dims(axes, 0)

        for i, (r, rgb, overlay) in enumerate(selected):
            axes[i, 0].imshow(rgb)
            axes[i, 0].axis("off")
            axes[i, 0].set_title(f"Image (GT={int(r.y_true)}, P={r.y_prob:.3f})")
            axes[i, 1].imshow(overlay)
            axes[i, 1].axis("off")
            axes[i, 1].set_title("Grad-CAM overlay")

        plt.tight_layout()
        plt.savefig("results/figures/fig_gradcam_cases.png", dpi=600)
        plt.close()

    print("Wrote explainability artifacts.")
