import os
import glob
import numpy as np
import pandas as pd
import joblib
import cv2
import warnings
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.utils.io import load_config
from src.features.abc import extract_abc
from src.models.embed_extract import extract_embeddings
from src.explainability.shap_xgb import shap_values_tree
from src.segmentation.metrics import dice

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    task1_img_dir = os.path.join(cfg["data_root"], cfg["raw"]["isic_task1"]["images"])
    task1_gt_dir = os.path.join(cfg["data_root"], cfg["raw"]["isic_task1"]["masks"])
    task1_ls_dir = cfg["derived"]["isic_task1_levelset_dir"]

    # Grab 50 ground truth masks
    np.random.seed(42)
    gt_paths = sorted(glob.glob(os.path.join(task1_gt_dir, "*.png")))
    subset_indices = np.random.choice(len(gt_paths), size=min(50, len(gt_paths)), replace=False)
    gt_paths = [gt_paths[i] for i in subset_indices]

    records = []
    abc_rows = []
    image_ids = []

    print("Step 1: Computing Dice & Extracting ABC features...")
    for gt_path in tqdm(gt_paths):
        base = os.path.splitext(os.path.basename(gt_path))[0] # E.g., ISIC_000001 or ISIC_000001_segmentation
        name = base.replace("_segmentation", "")
        
        # In the pipeline, the levelset masks are usually named identical to the image (e.g., ISIC_0000000.png)
        # However, 02_segment_isic_task1.py implies GT name = Levelset name. We check both.
        pr_path1 = os.path.join(task1_ls_dir, f"{name}.png")
        pr_path2 = os.path.join(task1_ls_dir, f"{base}.png")
        
        img_path = os.path.join(task1_img_dir, f"{name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(task1_img_dir, f"{base}.jpg")
            if not os.path.exists(img_path): continue
            
        gt = (cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        
        if os.path.exists(pr_path1): pr = (cv2.imread(pr_path1, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        elif os.path.exists(pr_path2): pr = (cv2.imread(pr_path2, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        else:
            from src.segmentation.levelset import segment_levelset
            rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            rgb_small = cv2.resize(rgb, (256, 256))
            pr_small = segment_levelset(rgb_small, **cfg["segmentation"]["levelset"])
            pr = cv2.resize(pr_small, (w, h), interpolation=cv2.INTER_NEAREST)
            
        d = dice(pr, gt)
        
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = extract_abc(rgb, pr)
        feats["image_id"] = name
        abc_rows.append(feats)
        records.append({"image_id": name, "dice": d})
        image_ids.append(name)
        
    df_dice = pd.DataFrame(records)
    df_abc = pd.DataFrame(abc_rows)

    if df_dice.empty:
        raise ValueError("No matching images found for subset!")

    print(f"Computed for {len(df_dice)} images. Mean Dice: {df_dice['dice'].mean():.3f}")

    print("\nStep 2: Extracting Embeddings...")
    npy_out = "data/derived/embeddings/isic_task1_subset_effb0.npy"
    csv_out = "data/derived/embeddings/isic_task1_subset_effb0_ids.csv"
    extract_embeddings(
        cfg["training"]["deep"]["backbone"],
        pd.DataFrame({"image_id": image_ids}),
        task1_img_dir, npy_out, csv_out,
        image_size=cfg["image"]["size"], normalize=cfg["image"]["normalize"],
        batch_size=32, num_workers=0, kind="task1_subset"
    )

    print("\nStep 3: Creating Hybrid Features and Predicting...")
    E = np.load(npy_out)
    ids_emb = pd.read_csv(csv_out)["image_id"].tolist()
    
    idx = {k: i for i, k in enumerate(ids_emb)}
    keep = [idx[i] for i in df_dice["image_id"].tolist() if i in idx]
    df_abc_al = df_abc[df_abc["image_id"].isin(idx.keys())].set_index("image_id").loc[[ids_emb[k] for k in keep]].reset_index()
    df_dice_al = df_dice[df_dice["image_id"].isin(idx.keys())].set_index("image_id").loc[[ids_emb[k] for k in keep]].reset_index()
    E_al = E[keep]

    pack = joblib.load("results/runs/hybrid/hybrid_xgb.joblib")
    model = pack["model"]
    feat_cols = pack["feat_cols"]

    X_abc = df_abc_al[feat_cols].values
    X = np.concatenate([X_abc, E_al], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = model.predict_proba(X)[:, 1]
    
    df_dice_al["y_prob"] = p

    print("\nStep 4: Computing SHAP Values...")
    sv, _ = shap_values_tree(model, X)
    
    hap_abc_sum = np.abs(sv[:, :len(feat_cols)]).sum(axis=1)
    df_dice_al["shap_abc_magnitude"] = hap_abc_sum

    corr_abc, p_abc = spearmanr(df_dice_al["dice"], df_dice_al["shap_abc_magnitude"])
    corr_prob, p_prob = spearmanr(df_dice_al["dice"], df_dice_al["y_prob"])

    print(f"\nResults:")
    print(f"Spearman correlation (Dice vs ABC SHAP magnitude): {corr_abc:.3f} (p={p_abc:.3e})")
    print(f"Spearman correlation (Dice vs y_prob): {corr_prob:.3f} (p={p_prob:.3e})")
    
    pd.DataFrame([{
        "metric": "Spearman_Dice_vs_SHAP_ABC", "r": corr_abc, "p": p_abc
    }, {
        "metric": "Spearman_Dice_vs_Prob", "r": corr_prob, "p": p_prob
    }]).to_csv("results/tables/table_segmentation_impact.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(df_dice_al["dice"], df_dice_al["shap_abc_magnitude"], alpha=0.5, color='blue')
    plt.xlabel("Segmentation Dice Score")
    plt.ylabel("Total ABS SHAP Magnitude\n(ABC Features)")

    plt.subplot(1, 2, 2)
    plt.scatter(df_dice_al["dice"], df_dice_al["y_prob"], alpha=0.5, color='red')
    plt.xlabel("Segmentation Dice Score")
    plt.ylabel("Hybrid Model y_prob")
    plt.title(f"Dice vs Predicted Probability\nSpearman $\\rho$ = {corr_prob:.3f}")

    plt.tight_layout()
    plt.savefig("results/figures/fig_segmentation_impact.png", dpi=200)
    print("Saved results/figures/fig_segmentation_impact.png")
