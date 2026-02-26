import os
from glob import glob
import cv2
from tqdm import tqdm
from src.utils.io import ensure_dir

def run_segmentation(image_dir, out_dir, segment_fn, limit=0, start=0, pattern="*.*"):
    ensure_dir(out_dir)

    paths = sorted(glob(os.path.join(image_dir, pattern)))

    start = max(0, int(start))
    if limit and limit > 0:
        paths = paths[start:start + limit]
    else:
        paths = paths[start:]

    for p in tqdm(paths, desc=f"Seg -> {out_dir}"):
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = (segment_fn(rgb) > 0).astype("uint8") * 255
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), mask)