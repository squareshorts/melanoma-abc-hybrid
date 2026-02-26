import pandas as pd
from utils.split import _infer_group_key


def inspect_group_distribution(metadata_path: str):
    df = pd.read_csv(metadata_path)

    group_key = _infer_group_key(df)
    group_counts = df.groupby(group_key).size()

    print("Grouping key used:", group_key)
    print("Total groups:", group_counts.shape[0])
    print("Groups with >1 image:", (group_counts > 1).sum())
    print("Mean images per group:", group_counts.mean())
    print("Max images per group:", group_counts.max())


if __name__ == "__main__":
    metadata_path = "data/raw/HAM10000/metadata.csv"
    inspect_group_distribution(metadata_path)