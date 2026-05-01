import os
import glob
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import sys

def compute_phash(img_path):
    try:
        with Image.open(img_path) as img:
            return os.path.basename(img_path).replace(".jpg", ""), str(imagehash.phash(img))
    except Exception as e:
        return None

def get_all_hashes(img_dir, num_workers=4):
    paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    results = {}
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(compute_phash, paths), total=len(paths), desc=f"Hashing {os.path.basename(img_dir)}"):
            if res:
                img_id, h = res
                results[img_id] = h
    return results

if __name__ == "__main__":
    ham_dir = "C:/work/datasets/HAM10000/images"
    bcn_dir = "C:/work/datasets/BCN20000/images"

    if not os.path.exists(ham_dir):
        raise SystemExit(f"HAM10000 directory missing: {ham_dir}")
    if not os.path.exists(bcn_dir):
        raise SystemExit(f"BCN20000 directory missing: {bcn_dir}")

    # Compute hashes
    print("Computing perceptual hashes for HAM10000...")
    ham_hashes = get_all_hashes(ham_dir, num_workers=8)
    
    print("Computing perceptual hashes for BCN20000...")
    bcn_hashes = get_all_hashes(bcn_dir, num_workers=8)

    # 1. Exact ID overlap
    ham_ids = set(ham_hashes.keys())
    bcn_ids = set(bcn_hashes.keys())
    id_overlap = ham_ids.intersection(bcn_ids)
    print(f"ISIC ID Overlap count: {len(id_overlap)}")

    # 2. Perceptual Hash overlap
    ham_hash_set = set(ham_hashes.values())
    bcn_overlap_ids = []
    
    for bcn_id, bcn_h in bcn_hashes.items():
        if bcn_h in ham_hash_set:
            bcn_overlap_ids.append(bcn_id)
            
    print(f"Perceptual Hash Overlap count: {len(bcn_overlap_ids)}")
    
    total_overlap = set(list(id_overlap) + bcn_overlap_ids)
    
    valid_bcn_ids = bcn_ids - total_overlap
    print(f"Valid BCN20000 independent images: {len(valid_bcn_ids)} out of {len(bcn_ids)}")
    
    df = pd.DataFrame({"image_id": list(valid_bcn_ids)})
    os.makedirs("data/splits", exist_ok=True)
    df.to_csv("data/splits/bcn20000_valid_ids.csv", index=False)
    print("Saved valid independent IDs to data/splits/bcn20000_valid_ids.csv")
    
    assert len(total_overlap) == 0, f"Overlap detected! {len(total_overlap)} images are not independent."
    print("Zero-overlap audit PASSED!")
