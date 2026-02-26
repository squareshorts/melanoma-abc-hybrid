from src.utils.io import load_config
from src.utils.seed import set_seed
from src.data.split_patient import make_patient_split

if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    make_patient_split(cfg["paths"]["ham_metadata"], cfg["derived"]["split_json"], seed=cfg["seed"])
    print("Wrote:", cfg["derived"]["split_json"])
