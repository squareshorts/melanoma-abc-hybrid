import os
from glob import glob
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
from src.utils.io import ensure_dir

def _process_one(p, out_dir, segment_fn, skip_existing):
    name = os.path.splitext(os.path.basename(p))[0]
    out_path = os.path.join(out_dir, f"{name}.png")

    if skip_existing and os.path.exists(out_path):
        return True

    img_bgr = cv2.imread(p)
    if img_bgr is None:
        return False
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = (segment_fn(rgb) > 0).astype("uint8") * 255
    cv2.imwrite(out_path, mask)
    return True

def run_segmentation_parallel(image_dir, out_dir, segment_fn, limit=0, start=0, pattern="*.*", n_jobs=4, skip_existing=True):
    ensure_dir(out_dir)

    paths = sorted(glob(os.path.join(image_dir, pattern)))
    start = max(0, int(start))
    if limit and limit > 0:
        paths = paths[start:start + limit]
    else:
        paths = paths[start:]

    results = Parallel(n_jobs=int(n_jobs), prefer="processes")(
        delayed(_process_one)(p, out_dir, segment_fn, skip_existing) for p in tqdm(paths, desc=f"Seg(par) -> {out_dir}")
    )
    return sum(bool(r) for r in results)