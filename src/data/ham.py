import os
import pandas as pd

MALIGNANT_DX = {"mel", "bcc", "akiec"}

def load_ham_metadata(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"image_id", "lesion_id"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"metadata.csv must contain columns: {required}")
    if "dx" not in df.columns:
        raise ValueError("metadata.csv must contain dx column")
    df["label"] = df["dx"].astype(str).str.lower().isin(MALIGNANT_DX).astype(int)
    return df

def image_path(images_dir: str, image_id: str) -> str:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(images_dir, image_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find image file for {image_id} in {images_dir}")
