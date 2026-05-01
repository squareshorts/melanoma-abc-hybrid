import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torchvision import transforms
from PIL import Image

from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.features.abc import extract_abc
from src.segmentation.baseline import segment_otsu

class SimpleImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    split = read_json(cfg["derived"]["split_json"])
    # We will just pick 150 train and 50 test images dynamically
    np.random.seed(42)
    train_lesions = np.random.choice(split["train"], 150, replace=False)
    test_lesions = np.random.choice(split["test"], 50, replace=False)
    
    df_meta = load_ham_metadata(os.path.join(cfg["data_root"], cfg["raw"]["ham"]["metadata"]))
    
    df_tr = df_meta[df_meta["lesion_id"].isin(train_lesions)].copy().drop_duplicates(subset=["lesion_id"])
    df_te = df_meta[df_meta["lesion_id"].isin(test_lesions)].copy().drop_duplicates(subset=["lesion_id"])
    
    ham_img_dir = os.path.join(cfg["data_root"], cfg["raw"]["ham"]["images"])
    
    def get_feats_lbls(df_sub):
        E = []
        A = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=True, num_classes=0).to(device)
        model.eval()
        
        paths = [os.path.join(ham_img_dir, f"{name}.jpg") for name in df_sub["image_id"].tolist()]
        
        # 1. Embeddings
        tfm = transforms.Compose([
            transforms.Resize((cfg["image"]["size"], cfg["image"]["size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        loader = DataLoader(SimpleImageDataset(paths, tfm), batch_size=32, shuffle=False)
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(device)
                out = model(bx)
                E.append(out.cpu().numpy())
        E = np.vstack(E)
        
        # 2. ABC features
        for p in paths:
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            mask = segment_otsu(rgb)
            feats = extract_abc(rgb, mask)
            # Remove image_id from features to leave only numbers
            feats.pop("image_id", None)
            A.append(feats)
            
        A_df = pd.DataFrame(A)
        return E, A_df

    print("Extracting for Train subset...")
    Etr, Atr_df = get_feats_lbls(df_tr)
    print("Extracting for Test subset...")
    Ete, Ate_df = get_feats_lbls(df_te)
    
    abc_cols = Atr_df.columns.tolist()
    
    scaler = StandardScaler()
    Etr_s = scaler.fit_transform(Etr)
    Ete_s = scaler.transform(Ete)
    
    results = []
    
    for c in abc_cols:
        y_tr = Atr_df[c].values
        y_te = Ate_df[c].values
        
        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
        y_te_s = y_scaler.transform(y_te.reshape(-1, 1)).flatten()
        
        ridge = Ridge(alpha=10.0)
        ridge.fit(Etr_s, y_tr_s)
        
        pred_te = ridge.predict(Ete_s)
        r2 = r2_score(y_te_s, pred_te)
        
        results.append({"feature": c, "R2": r2})
        print(f" {c:25s} | Test R2: {r2:.3f}")
        
    df_res = pd.DataFrame(results).sort_values("R2", ascending=False)
    df_res.to_csv("results/tables/table_implicit_concept_learning.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_res["feature"][::-1], np.maximum(0, df_res["R2"][::-1]), color="purple")
    plt.xlabel("Test $R^2$ Score")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/figures/fig_implicit_concept_learning.png", dpi=200)
    print("Saved results/figures/fig_implicit_concept_learning.png")
