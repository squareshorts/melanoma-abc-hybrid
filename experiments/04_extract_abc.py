import os
import pandas as pd
from src.utils.io import load_config
from src.data.isic import load_isic_task3_labels
from src.segmentation.levelset import segment_levelset
from src.segmentation.run_batch import run_segmentation
from src.features.extract_batch import extract_csv

if __name__ == "__main__":
    cfg = load_config()

    if (not os.path.exists(cfg["derived"]["ham_seg_dir"])) or (len(os.listdir(cfg["derived"]["ham_seg_dir"])) == 0):
        raise SystemExit("HAM segmentations are missing. Run experiments/03_segment_ham.py first.")

    extract_csv(cfg["paths"]["ham_images"], cfg["derived"]["ham_seg_dir"], cfg["derived"]["ham_abc_csv"])

    ham_feats = pd.read_csv(cfg["derived"]["ham_abc_csv"])
    task3_ids = load_isic_task3_labels(cfg["paths"]["isic_task3_labels"])[["image_id"]].drop_duplicates()
    task3_overlap = task3_ids["image_id"].isin(set(ham_feats["image_id"]))

    if bool(task3_overlap.all()):
        task3_feats = task3_ids.merge(ham_feats, on="image_id", how="left")
        task3_feats.to_csv(cfg["derived"]["isic_task3_abc_csv"], index=False)
    else:
        # Fallback for a genuinely non-overlapping Task 3 directory.
        if (not os.path.exists(cfg["derived"]["isic_task3_seg_dir"])) or (len(os.listdir(cfg["derived"]["isic_task3_seg_dir"])) == 0):
            run_segmentation(cfg["paths"]["isic_task3_images"], cfg["derived"]["isic_task3_seg_dir"],
                             lambda rgb: segment_levelset(rgb, **cfg["segmentation"]["levelset"]))
        extract_csv(cfg["paths"]["isic_task3_images"], cfg["derived"]["isic_task3_seg_dir"], cfg["derived"]["isic_task3_abc_csv"])

    print("Wrote:", cfg["derived"]["ham_abc_csv"])
    print("Wrote:", cfg["derived"]["isic_task3_abc_csv"])
