import pandas as pd
from src.utils.io import load_config, read_json
from src.data.ham import load_ham_metadata
from src.data.isic import load_isic_task3_labels
from src.models.embed_extract import extract_embeddings

if __name__ == "__main__":
    cfg = load_config()
    split = read_json(cfg["derived"]["split_json"])

    df_ham = load_ham_metadata(cfg["paths"]["ham_metadata"])
    all_lesions = set(split["train"]) | set(split["val"]) | set(split["test"])
    df_ham_all = df_ham[df_ham["lesion_id"].isin(all_lesions)][["image_id"]].drop_duplicates().copy()

    df_isic = load_isic_task3_labels(cfg["paths"]["isic_task3_labels"])[["image_id"]].drop_duplicates().copy()

    extract_embeddings(cfg["training"]["deep"]["backbone"], df_ham_all, cfg["paths"]["ham_images"],
                       cfg["derived"]["ham_emb_npy"], cfg["derived"]["ham_emb_ids"],
                       image_size=cfg["image"]["size"], normalize=cfg["image"]["normalize"],
                       batch_size=64, num_workers=cfg["training"]["deep"].get("num_workers", 2), kind="ham")

    extract_embeddings(cfg["training"]["deep"]["backbone"], df_isic, cfg["paths"]["isic_task3_images"],
                       cfg["derived"]["isic_task3_emb_npy"], cfg["derived"]["isic_task3_emb_ids"],
                       image_size=cfg["image"]["size"], normalize=cfg["image"]["normalize"],
                       batch_size=64, num_workers=cfg["training"]["deep"].get("num_workers", 2), kind="isic")

    print("Wrote embeddings.")
