import numpy as np
import pandas as pd


def main():

    # Load ONE feature matrix to detect dimension
    X = np.load("data/derived/embeddings/effb0_embeddings_test.npy")

    n_features = X.shape[1]
    print("Detected feature dimension:", n_features)

    # ASSUMPTION:
    # Last 4 columns = GLCM metrics
    # Everything before that = ResNet embeddings

    if n_features < 5:
        raise ValueError("Feature dimension too small. Check pipeline.")

    n_resnet = n_features - 4

    resnet_names = [f"resnet_{i}" for i in range(n_resnet)]
    glcm_names = [
        "glcm_corr",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_contrast",
    ]

    feature_names = resnet_names + glcm_names

    df = pd.DataFrame({"feature": feature_names})

    df.to_csv("data/derived/features/feature_names.csv", index=False)

    print("feature_names.csv created successfully.")
    print("Total features written:", len(feature_names))


if __name__ == "__main__":
    main()