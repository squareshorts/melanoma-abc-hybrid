import numpy as np
from pathlib import Path

SEED = 42
N_SUB = 300

image_dir = Path("data/raw/ISIC2018/Task1/images")

if not image_dir.exists():
    raise RuntimeError(f"Directory not found: {image_dir.resolve()}")

image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

if len(image_paths) == 0:
    raise RuntimeError("No images found in Task1/images.")

ids = np.array([p.stem for p in image_paths])

print(f"Detected {len(ids)} Task1 images.")

if len(ids) < N_SUB:
    raise RuntimeError(f"Only {len(ids)} images found; cannot sample {N_SUB}.")

rng = np.random.default_rng(SEED)
subset_ids = rng.choice(ids, size=N_SUB, replace=False)

out_dir = Path("results/segmentation")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "task1_subset_300_seed42.txt"
out_path.write_text("\n".join(subset_ids) + "\n")

print(f"Saved subset to: {out_path}")