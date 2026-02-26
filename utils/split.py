import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _infer_group_key(metadata_df: pd.DataFrame):
    """
    Determine grouping key:
    Prefer patient_id if exists and not mostly null.
    Otherwise fallback to lesion_id.
    """
    if "patient_id" in metadata_df.columns:
        if metadata_df["patient_id"].notnull().sum() > 0:
            return "patient_id"
    if "lesion_id" in metadata_df.columns:
        return "lesion_id"
    raise ValueError("Neither patient_id nor lesion_id found in metadata.")


def split_by_patient(
    metadata_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    """
    Perform strict group-level split.
    All samples from same patient (or lesion fallback)
    remain in same split.

    Returns:
        train_df, val_df, test_df
    """

    rng = np.random.RandomState(seed)

    df = metadata_df.copy()

    group_key = _infer_group_key(df)

    # Group-level label definition:
    # group is positive if ANY lesion in group is malignant (label=1)
    group_labels = (
        df.groupby(group_key)["label"]
        .max()
        .reset_index()
    )

    groups = group_labels[group_key].values
    y_group = group_labels["label"].values

    # First split: train vs temp (val+test)
    g_train, g_temp, y_train, y_temp = train_test_split(
        groups,
        y_group,
        test_size=(test_size + val_size),
        stratify=y_group,
        random_state=seed,
    )

    # Second split: val vs test from temp
    relative_test_size = test_size / (test_size + val_size)

    g_val, g_test, y_val, y_test = train_test_split(
        g_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=seed,
    )

    train_df = df[df[group_key].isin(g_train)].reset_index(drop=True)
    val_df   = df[df[group_key].isin(g_val)].reset_index(drop=True)
    test_df  = df[df[group_key].isin(g_test)].reset_index(drop=True)

    # Sanity checks
    assert set(train_df[group_key]).isdisjoint(val_df[group_key])
    assert set(train_df[group_key]).isdisjoint(test_df[group_key])
    assert set(val_df[group_key]).isdisjoint(test_df[group_key])

    return train_df, val_df, test_df