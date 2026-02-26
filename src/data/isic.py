import os
import pandas as pd

def load_isic_task3_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    required = {"image_id", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError("labels.csv must contain columns: image_id,label")
    df["label"] = df["label"].astype(int)
    return df

def image_path(images_dir: str, image_id: str) -> str:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(images_dir, image_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find image file for {image_id} in {images_dir}")
