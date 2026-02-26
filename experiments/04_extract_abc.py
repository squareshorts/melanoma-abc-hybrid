import os
from src.utils.io import load_config
from src.segmentation.levelset import segment_levelset
from src.segmentation.run_batch import run_segmentation
from src.features.extract_batch import extract_csv

if __name__ == "__main__":
    cfg = load_config()

    # Ensure ISIC Task 3 segmentation exists
    if (not os.path.exists(cfg["derived"]["isic_task3_seg_dir"])) or (len(os.listdir(cfg["derived"]["isic_task3_seg_dir"])) == 0):
        run_segmentation(cfg["paths"]["isic_task3_images"], cfg["derived"]["isic_task3_seg_dir"],
                         lambda rgb: segment_levelset(rgb, **cfg["segmentation"]["levelset"]))

    extract_csv(cfg["paths"]["ham_images"], cfg["derived"]["ham_seg_dir"], cfg["derived"]["ham_abc_csv"])
    extract_csv(cfg["paths"]["isic_task3_images"], cfg["derived"]["isic_task3_seg_dir"], cfg["derived"]["isic_task3_abc_csv"])

    print("Wrote:", cfg["derived"]["ham_abc_csv"])
    print("Wrote:", cfg["derived"]["isic_task3_abc_csv"])
