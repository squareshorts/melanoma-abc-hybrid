import os, json
from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get("data_root")
    raw = cfg.get("raw", {})
    paths = cfg.setdefault("paths", {})

    def raw_path(section, key):
        value = raw.get(section, {}).get(key)
        if value is None:
            return None
        return os.path.join(data_root, value) if data_root else value

    defaults = {
        "ham_images": raw_path("ham", "images"),
        "ham_metadata": raw_path("ham", "metadata"),
        "isic_task1_images": raw_path("isic_task1", "images"),
        "isic_task1_masks": raw_path("isic_task1", "masks"),
        "isic_task3_images": raw_path("isic_task3", "images"),
        "isic_task3_labels": raw_path("isic_task3", "labels"),
    }
    for key, value in defaults.items():
        if value is not None:
            paths.setdefault(key, value)

    return cfg

def ensure_dir(p: str):
    if not p:
        return
    Path(p).mkdir(parents=True, exist_ok=True)

def read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj, p: str):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
