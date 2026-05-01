import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src.utils.io import load_config
from src.segmentation.levelset import segment_levelset
from src.features.abc import extract_abc
from src.features.extract_batch import extract_csv
import numpy as np
import torchvision.transforms as T
import timm
import cv2
import multiprocessing
from functools import partial

def process_single_image(args):
    img_path, seg_dir, cfg_levelset = args
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(seg_dir, f"{image_id}.png")
    
    # Check if already processed
    if os.path.exists(mask_path):
        return True
        
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return False
            
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Segment
        mask = (segment_levelset(rgb, **cfg_levelset) > 0).astype("uint8") * 255
        cv2.imwrite(mask_path, mask)
        
        return True
    except Exception as e:
        # Catch errors silently to not break parallel loop
        return False

def extract_single_abc(args):
    img_path, mask_dir = args
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f"{image_id}.png")
    
    if not os.path.exists(mask_path):
        return None
        
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
            
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype("uint8")
        
        feats = extract_abc(rgb, mask)
        feats["image_id"] = image_id
        return feats
    except Exception as e:
        return None

def extract_embeddings_for_bcn(cfg):
    valid_csv = "data/splits/bcn20000_valid_ids.csv"
    if not os.path.exists(valid_csv):
        print("Warning: valid IDs not found, running on all BCN20000 images")
        valid_ids = [f.replace(".jpg", "") for f in os.listdir("C:/work/datasets/BCN20000/images") if f.endswith(".jpg")]
    else:
        valid_ids = pd.read_csv(valid_csv)["image_id"].tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = timm.create_model(
        cfg["training"]["deep"]["backbone"],
        pretrained=False,
        num_classes=1
    )
    ckpt = torch.load(cfg["derived"]["deep_ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    # We remove the classifier head to get embeddings
    if hasattr(model, 'classifier'):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Identity()

    transform = T.Compose([
        T.Resize((cfg["image"]["size"], cfg["image"]["size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    embeddings = []
    ids = []
    
    for img_id in tqdm(valid_ids, desc="Extracting Deep Embeddings"):
        img_path = f"C:/work/datasets/BCN20000/images/{img_id}.jpg"
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            ids.append(img_id)
        except Exception as e:
            print(f"Failed on {img_id}: {e}")
            
    os.makedirs("data/derived/embeddings", exist_ok=True)
    np.save("data/derived/embeddings/bcn20000_effb0.npy", np.array(embeddings))
    pd.DataFrame({"image_id": ids}).to_csv("data/derived/embeddings/bcn20000_effb0_ids.csv", index=False)
    print("Saved embeddings.")

if __name__ == "__main__":
    cfg = load_config()
    bcn_img_dir = "C:/work/datasets/BCN20000/images"
    bcn_seg_dir = "data/derived/segmentations/bcn20000_levelset"
    bcn_abc_csv = "data/derived/features/bcn20000_abc.csv"
    
    os.makedirs(bcn_seg_dir, exist_ok=True)
    os.makedirs(os.path.dirname(bcn_abc_csv), exist_ok=True)
    
    import glob
    img_paths = sorted(glob.glob(os.path.join(bcn_img_dir, "*.jpg")))
    
    print("Running LevelSet Segmentation in Parallel...")
    seg_args = [(p, bcn_seg_dir, cfg["segmentation"]["levelset"]) for p in img_paths]
    
    cores = min(8, multiprocessing.cpu_count())
    print(f"Using {cores} CPU cores to avoid memory overload...")
    
    with multiprocessing.Pool(cores) as pool:
        list(tqdm(pool.imap_unordered(process_single_image, seg_args), total=len(seg_args), desc="Parallel Segmentation"))
    
    print("Extracting ABC features in Parallel...")
    abc_args = [(p, bcn_seg_dir) for p in img_paths]
    
    valid_feats = []
    with multiprocessing.Pool(cores) as pool:
        for res in tqdm(pool.imap_unordered(extract_single_abc, abc_args), total=len(abc_args), desc="Parallel ABC"):
            if res is not None:
                valid_feats.append(res)
                
    if valid_feats:
        df = pd.DataFrame(valid_feats).sort_values("image_id")
        df.to_csv(bcn_abc_csv, index=False)
        print(f"Saved ABC features to {bcn_abc_csv}")
    
    print("Extracting Deep Embeddings (CUDA)...")
    extract_embeddings_for_bcn(cfg)
    
    print("Feature extraction complete.")
