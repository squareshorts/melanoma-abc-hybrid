import os, json
from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
