from src.utils.io import load_config
from src.models.deep_train import train_deep

if __name__ == "__main__":
    cfg = load_config()
    train_deep(cfg)
