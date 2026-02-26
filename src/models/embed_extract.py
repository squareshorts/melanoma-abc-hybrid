import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

from src.data.transforms import build_transforms
from src.data.ham import image_path as ham_image_path
from src.data.isic import image_path as isic_image_path
from src.utils.io import ensure_dir

class SimpleImageDataset(Dataset):
    def __init__(self, df_ids, images_dir, tfm, kind="ham"):
        self.df = df_ids.reset_index(drop=True)
        self.images_dir = images_dir
        self.tfm = tfm
        self.kind = kind

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image_id"]
        p = ham_image_path(self.images_dir, img_id) if self.kind=="ham" else isic_image_path(self.images_dir, img_id)
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, img_id

def build_embed_model(backbone: str):
    return timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")

@torch.no_grad()
def extract_embeddings(backbone, df_ids, images_dir, out_npy, out_ids_csv, image_size=224, normalize="imagenet",
                       batch_size=64, num_workers=2, kind="ham"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = build_transforms(image_size, normalize, train=False)
    ds = SimpleImageDataset(df_ids, images_dir, tfm, kind=kind)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(num_workers))

    model = build_embed_model(backbone).to(device).eval()

    embs, ids = [], []
    for x, img_id in ld:
        x = x.to(device)
        z = model(x).detach().cpu().numpy()
        embs.append(z)
        ids += list(img_id)

    E = np.concatenate(embs, axis=0)
    ensure_dir(os.path.dirname(out_npy))
    np.save(out_npy, E)
    pd.DataFrame({"image_id": ids}).to_csv(out_ids_csv, index=False)
    return E, ids
