import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import write_json


def _infer_group_key(df: pd.DataFrame):
    # Prefer patient_id if present and non-null, else fall back to lesion_id.
    if "patient_id" in df.columns and df["patient_id"].notnull().sum() > 0:
        return "patient_id"
    if "lesion_id" in df.columns:
        return "lesion_id"
    raise ValueError("No valid grouping column found (patient_id or lesion_id).")


def _derive_binary_label_from_dx(df: pd.DataFrame):
    if "dx" not in df.columns:
        raise ValueError("metadata.csv must contain 'dx' column to derive label.")
    # Binary target used in your pipeline: melanoma vs rest
    return (df["dx"].astype(str).str.lower() == "mel").astype(int)


def make_patient_split(
    metadata_csv: str,
    out_json: str,
    seed: int = 42,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
):
    df = pd.read_csv(metadata_csv)

    group_key = _infer_group_key(df)
    df["label"] = _derive_binary_label_from_dx(df)

    # Group-level label: group is positive if ANY sample in group is positive
    group_labels = df.groupby(group_key)["label"].max().reset_index()

    groups = group_labels[group_key].values
    y_group = group_labels["label"].values

    # train vs temp
    g_train, g_temp, y_train, y_temp = train_test_split(
        groups,
        y_group,
        test_size=(val + test),
        stratify=y_group,
        random_state=seed,
    )

    # val vs test
    rel_test = test / (val + test)
    g_val, g_test, _, _ = train_test_split(
        g_temp,
        y_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=seed,
    )

    split = {
        "train": g_train.tolist(),
        "val": g_val.tolist(),
        "test": g_test.tolist(),
        "group_key": group_key,
        "label_definition": "label = 1 if dx == 'mel' else 0; group label = max(label) within group",
        "seed": seed,
        "fractions": {"train": train, "val": val, "test": test},
    }

    write_json(split, out_json)

    print(f"Split created using group_key = {group_key}")
    print("Train groups:", len(g_train))
    print("Val groups:", len(g_val))
    print("Test groups:", len(g_test))

    return split