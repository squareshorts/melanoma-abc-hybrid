import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imagehash
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.transforms import build_transforms  # noqa: E402
from src.models.xgb_models import predict_proba  # noqa: E402
from src.utils.io import load_config  # noqa: E402


RESULT_TABLES = ROOT / "results" / "tables"
RESULT_FIGURES = ROOT / "results" / "figures"
RESULT_AUDIT = ROOT / "results" / "audit"
DCA_GRID = np.linspace(0.02, 0.50, 50)
EPS = 1e-7


def ensure_dirs():
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULT_AUDIT.mkdir(parents=True, exist_ok=True)


def save_figure(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    chunks = np.array_split(order, min(n_bins, len(order)))
    ece = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(y)) * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return float(ece)


def mean_net_benefit(y_true, y_prob, thresholds=DCA_GRID):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(y) == 0:
        return float("nan")
    out = []
    n = len(y)
    for thr in thresholds:
        pred = p >= thr
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append((tp / n) - (fp / n) * (thr / (1.0 - thr)))
    return float(np.mean(out))


def safe_auc(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_prob))


def safe_pr_auc(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, y_prob)
    return float(np.trapz(precision, recall) * -1.0)


def safe_average_precision(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, y_prob))


def confusion_metrics(y_true, y_prob, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def calibration_slope_intercept(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(y) < 3 or len(np.unique(y)) < 2 or len(np.unique(np.round(p, 12))) < 2:
        return float("nan"), float("nan")
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(safe_logit(p).reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def model_metrics(y_true, y_prob, threshold=0.5, include_calibration=True):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    out = {
        "N": int(len(y)),
        "positives": int(y.sum()) if len(y) else 0,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "AUC": safe_auc(y, p),
        "PR-AUC": safe_average_precision(y, p),
        "Brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "ECE": expected_calibration_error(y, p),
        "mean_net_benefit_0.02_0.50": mean_net_benefit(y, p),
    }
    out.update(confusion_metrics(y, p, threshold=threshold))
    if include_calibration:
        slope, intercept = calibration_slope_intercept(y, p)
        out["calibration_slope"] = slope
        out["calibration_intercept"] = intercept
    return out


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ham_metadata(cfg):
    meta = pd.read_csv(cfg["paths"]["ham_metadata"])
    meta["label"] = (meta["dx"] == "mel").astype(int)
    return meta


def image_path_for_id(images_dir, image_id):
    images_dir = Path(images_dir)
    for ext in [".jpg", ".jpeg", ".png"]:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def split_assignments(cfg):
    split = read_json(ROOT / cfg["derived"]["split_json"])
    rows = []
    for name in ["train", "val", "test"]:
        for lesion_id in split[name]:
            rows.append({"lesion_id": lesion_id, "split": name})
    return pd.DataFrame(rows), split


def compute_phash_hex(path):
    try:
        with Image.open(path) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception:
        return None


BYTE_COUNTS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_uint64_matrix(a, b):
    xor = np.bitwise_xor(a[:, None], b[None, :])
    bytes_view = np.ascontiguousarray(xor).view(np.uint8).reshape(xor.shape + (8,))
    return BYTE_COUNTS[bytes_view].sum(axis=2)


def count_cross_split_phash_pairs(phash_df, threshold=4, max_examples=12):
    split_pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    total_exact = 0
    total_near_nonexact = 0
    min_distance = None
    exact_examples = []
    near_examples = []

    for split_a, split_b in split_pairs:
        da = phash_df[(phash_df["split"] == split_a) & phash_df["phash_int"].notna()]
        db = phash_df[(phash_df["split"] == split_b) & phash_df["phash_int"].notna()]
        a = da["phash_int"].astype(np.uint64).to_numpy()
        b = db["phash_int"].astype(np.uint64).to_numpy()
        ids_a = da["image_id"].to_numpy()
        ids_b = db["image_id"].to_numpy()
        for start in range(0, len(a), 256):
            aa = a[start : start + 256]
            if len(aa) == 0 or len(b) == 0:
                continue
            dist = hamming_uint64_matrix(aa, b)
            local_min = int(dist.min()) if dist.size else None
            if local_min is not None:
                min_distance = local_min if min_distance is None else min(min_distance, local_min)
            exact_mask = dist == 0
            total_exact += int(exact_mask.sum())
            near_mask = (dist > 0) & (dist <= threshold)
            total_near_nonexact += int(near_mask.sum())
            if len(exact_examples) < max_examples and np.any(exact_mask):
                coords = np.argwhere(exact_mask)
                for ia, ib in coords[: max_examples - len(exact_examples)]:
                    exact_examples.append(
                        f"{ids_a[start + int(ia)]}:{split_a} vs {ids_b[int(ib)]}:{split_b} d={int(dist[ia, ib])}"
                    )
                    if len(exact_examples) >= max_examples:
                        break
            if len(near_examples) < max_examples and np.any(near_mask):
                coords = np.argwhere(near_mask)
                for ia, ib in coords[: max_examples - len(near_examples)]:
                    near_examples.append(
                        f"{ids_a[start + int(ia)]}:{split_a} vs {ids_b[int(ib)]}:{split_b} d={int(dist[ia, ib])}"
                    )
                    if len(near_examples) >= max_examples:
                        break
    return total_exact, total_near_nonexact, min_distance, "; ".join(exact_examples), "; ".join(near_examples)


def run_split_integrity(cfg):
    print("Running split integrity audit...")
    ham = load_ham_metadata(cfg)
    assign, split = split_assignments(cfg)
    ham_split = ham.merge(assign, on="lesion_id", how="left")
    rows = []

    split_sets = {name: set(split[name]) for name in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_sets[a] & split_sets[b]
        rows.append(
            {
                "analysis": "split_integrity",
                "check": f"lesion_id_overlap_{a}_{b}",
                "value": len(overlap),
                "status": "pass" if len(overlap) == 0 else "fail",
                "details": ";".join(sorted(list(overlap))[:10]),
            }
        )

    image_cross = (
        ham_split.dropna(subset=["split"])
        .groupby("image_id")["split"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    cross = image_cross[image_cross["split"].map(len) > 1]
    rows.append(
        {
            "analysis": "split_integrity",
            "check": "duplicate_image_id_cross_split",
            "value": int(len(cross)),
            "status": "pass" if len(cross) == 0 else "fail",
            "details": ";".join(cross["image_id"].head(10).tolist()),
        }
    )

    image_dir = Path(cfg["paths"]["ham_images"])
    cache = RESULT_AUDIT / "ham_split_phashes.csv"
    expected_ids = set(ham_split["image_id"].dropna().unique())
    if cache.exists():
        phash_df = pd.read_csv(cache)
        if set(phash_df["image_id"]) != expected_ids:
            phash_df = None
    else:
        phash_df = None

    if phash_df is None:
        out = []
        for i, row in enumerate(ham_split[["image_id", "lesion_id", "split"]].drop_duplicates().itertuples(index=False), 1):
            path = image_path_for_id(image_dir, row.image_id)
            phash = compute_phash_hex(path) if path is not None else None
            out.append(
                {
                    "image_id": row.image_id,
                    "lesion_id": row.lesion_id,
                    "split": row.split,
                    "path_available": path is not None,
                    "phash": phash,
                }
            )
            if i % 1000 == 0:
                print(f"  hashed {i} HAM images")
        phash_df = pd.DataFrame(out)
        phash_df.to_csv(cache, index=False)

    phash_df["phash_int"] = phash_df["phash"].map(lambda x: int(str(x), 16) if pd.notna(x) else np.nan)
    n_missing = int(phash_df["phash"].isna().sum())
    rows.append(
        {
            "analysis": "split_integrity",
            "check": "phash_images_missing_or_unreadable",
            "value": n_missing,
            "status": "pass" if n_missing == 0 else "warning",
            "details": "perceptual hashes computed with imagehash.phash where images were available",
        }
    )

    exact, near, min_dist, exact_examples, near_examples = count_cross_split_phash_pairs(phash_df, threshold=4)
    rows += [
        {
            "analysis": "split_integrity",
            "check": "exact_phash_cross_split_pairs",
            "value": exact,
            "status": "pass" if exact == 0 else "warning",
            "details": exact_examples if exact else "",
        },
        {
            "analysis": "split_integrity",
            "check": "near_phash_cross_split_pairs_hamming_1_to_4",
            "value": near,
            "status": "pass" if near == 0 else "warning",
            "details": near_examples,
        },
        {
            "analysis": "split_integrity",
            "check": "minimum_cross_split_phash_hamming_distance",
            "value": min_dist if min_dist is not None else np.nan,
            "status": "pass" if (min_dist is not None and min_dist > 4) else "warning",
            "details": "near-duplicate screening threshold was Hamming distance <= 4",
        },
    ]

    pd.DataFrame(rows).to_csv(RESULT_TABLES / "table_split_integrity.csv", index=False)


def load_bcn_metadata_with_predictions():
    meta = pd.read_csv("C:/work/datasets/BCN20000/images/metadata.csv").rename(columns={"isic_id": "image_id"})
    preds = {
        "ABC-only": pd.read_csv(ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv"),
        "Deep": pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv"),
        "Hybrid": pd.read_csv(ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv"),
    }
    for name, df in preds.items():
        preds[name] = df.rename(columns={"y_prob": f"{name}_prob"})[["image_id", "y_true", f"{name}_prob"]]
    base = preds["Deep"].merge(meta, on="image_id", how="left")
    for name in ["ABC-only", "Hybrid"]:
        base = base.merge(preds[name][["image_id", f"{name}_prob"]], on="image_id", how="inner")
    base["age_group"] = pd.cut(
        pd.to_numeric(base["age_approx"], errors="coerce"),
        bins=[-np.inf, 39, 59, 74, np.inf],
        labels=["<=39", "40-59", "60-74", ">=75"],
    ).astype("object")
    subgroup_cols = ["age_group", "sex", "anatom_site_general", "diagnosis_1", "diagnosis_2", "diagnosis_3"]
    for col in subgroup_cols:
        base[col] = base[col].fillna("Missing").astype(str)
    return base


def subgroup_rows(df, subgroup_cols, models):
    rows = []
    for model_name, prob_col in models.items():
        overall = model_metrics(df["y_true"], df[prob_col], include_calibration=False)
        rows.append(
            {
                "dataset": "BCN20000",
                "model": model_name,
                "subgroup_variable": "overall",
                "subgroup_level": "overall",
                **{k: overall[k] for k in ["N", "positives", "prevalence", "AUC", "PR-AUC", "Brier", "sensitivity", "specificity", "accuracy"]},
            }
        )
        for col in subgroup_cols:
            for level, g in df.groupby(col, dropna=False):
                metrics = model_metrics(g["y_true"], g[prob_col], include_calibration=False)
                rows.append(
                    {
                        "dataset": "BCN20000",
                        "model": model_name,
                        "subgroup_variable": col,
                        "subgroup_level": level,
                        **{
                            k: metrics[k]
                            for k in [
                                "N",
                                "positives",
                                "prevalence",
                                "AUC",
                                "PR-AUC",
                                "Brier",
                                "sensitivity",
                                "specificity",
                                "accuracy",
                            ]
                        },
                    }
                )
    return pd.DataFrame(rows)


def run_bcn_subgroup_analysis():
    print("Running BCN20000 subgroup analysis...")
    df = load_bcn_metadata_with_predictions()
    models = {"ABC-only": "ABC-only_prob", "Deep": "Deep_prob", "Hybrid": "Hybrid_prob"}
    subgroup_cols = ["age_group", "sex", "anatom_site_general", "diagnosis_1", "diagnosis_2", "diagnosis_3"]
    out = subgroup_rows(df, subgroup_cols, models)
    out.to_csv(RESULT_TABLES / "table_bcn_subgroup_performance.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    plot_cols = ["age_group", "sex", "anatom_site_general"]
    markers = {"age_group": "o", "sex": "s", "anatom_site_general": "^"}
    for ax, (model_name, prob_col) in zip(axes, models.items()):
        for col in plot_cols:
            for level, g in df.groupby(col):
                if len(g) < 30:
                    continue
                ax.scatter(
                    g[prob_col].mean(),
                    g["y_true"].mean(),
                    s=max(20, min(220, len(g) / 25)),
                    alpha=0.65,
                    marker=markers[col],
                    label=col if (model_name == "ABC-only" and level == sorted(df[col].unique())[0]) else None,
                )
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1)
        ax.set_xlabel(f"{model_name} mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed melanoma prevalence")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=8)
    save_figure(fig, RESULT_FIGURES / "fig_bcn_subgroup_calibration.png")


def load_model_prediction_frames():
    ham_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
    bcn_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv")
    frames = {
        "ABC-only": (
            pd.read_csv(ROOT / "results/runs/handcrafted/ham_test_predictions_handcrafted.csv"),
            pd.read_csv(ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv"),
        ),
        "Deep": (ham_deep, bcn_deep),
        "Hybrid": (
            pd.read_csv(ROOT / "results/runs/hybrid/ham_test_predictions_hybrid.csv"),
            pd.read_csv(ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv"),
        ),
    }
    return frames


def run_domain_shift():
    print("Running domain-shift quantification...")
    ham_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")
    bcn_deep = pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv")
    ham_abc = pd.read_csv(ROOT / "data/derived/features/ham_abc.csv")
    bcn_abc = pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv")
    ham = ham_deep.merge(ham_abc, on="image_id", how="inner")
    bcn = bcn_deep.merge(bcn_abc, on="image_id", how="inner")
    ham["deep_probability"] = ham["y_prob"]
    bcn["deep_probability"] = bcn["y_prob"]
    ham["deep_logit"] = safe_logit(ham["deep_probability"])
    bcn["deep_logit"] = safe_logit(bcn["deep_probability"])
    variables = [c for c in ham.columns if c.startswith(("A_", "B_", "C_"))] + ["deep_probability", "deep_logit"]
    rows = []
    for var in variables:
        x = pd.to_numeric(ham[var], errors="coerce").dropna().to_numpy()
        z = pd.to_numeric(bcn[var], errors="coerce").dropna().to_numpy()
        pooled = math.sqrt((np.var(x, ddof=1) + np.var(z, ddof=1)) / 2.0)
        smd = (float(np.mean(z)) - float(np.mean(x))) / pooled if pooled > 0 else float("nan")
        ks = ks_2samp(x, z)
        rows.append(
            {
                "variable": var,
                "ham_test_N": int(len(x)),
                "bcn_N": int(len(z)),
                "ham_test_mean": float(np.mean(x)),
                "bcn_mean": float(np.mean(z)),
                "ham_test_sd": float(np.std(x, ddof=1)),
                "bcn_sd": float(np.std(z, ddof=1)),
                "standardized_mean_difference_bcn_minus_ham": smd,
                "absolute_standardized_mean_difference": abs(smd) if np.isfinite(smd) else float("nan"),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT_TABLES / "table_domain_shift_abc.csv", index=False)

    frames = load_model_prediction_frames()
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    bins = np.linspace(0, 1, 50)
    for ax, (model_name, (ham_df, bcn_df)) in zip(axes, frames.items()):
        ax.hist(ham_df["y_prob"], bins=bins, density=True, histtype="step", linewidth=2, label=f"{model_name} HAM test")
        ax.hist(bcn_df["y_prob"], bins=bins, density=True, histtype="step", linewidth=2, label=f"{model_name} BCN20000")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Predicted melanoma probability")
    save_figure(fig, RESULT_FIGURES / "fig_domain_shift_scores.png")

    ham_emb_path = ROOT / "data/derived/embeddings/ham_effb0.npy"
    bcn_emb_path = ROOT / "data/derived/embeddings/bcn20000_effb0.npy"
    if ham_emb_path.exists() and bcn_emb_path.exists():
        ham_ids = pd.read_csv(ROOT / "data/derived/embeddings/ham_effb0_ids.csv")["image_id"].tolist()
        bcn_ids = pd.read_csv(ROOT / "data/derived/embeddings/bcn20000_effb0_ids.csv")["image_id"].tolist()
        ham_idx = {image_id: i for i, image_id in enumerate(ham_ids)}
        bcn_idx = {image_id: i for i, image_id in enumerate(bcn_ids)}
        ham_keep_ids = [i for i in ham_deep["image_id"].tolist() if i in ham_idx]
        bcn_keep_ids = [i for i in bcn_deep["image_id"].tolist() if i in bcn_idx]
        ham_E = np.load(ham_emb_path, mmap_mode="r")[[ham_idx[i] for i in ham_keep_ids]]
        bcn_E = np.load(bcn_emb_path, mmap_mode="r")[[bcn_idx[i] for i in bcn_keep_ids]]
        X = np.vstack([ham_E, bcn_E]).astype(np.float32)
        X = StandardScaler().fit_transform(X)
        coords = PCA(n_components=2, random_state=42, svd_solver="randomized").fit_transform(X)
        labels = np.array(["HAM test"] * len(ham_E) + ["BCN20000"] * len(bcn_E))
        fig, ax = plt.subplots(figsize=(6, 5))
        for label, color, alpha in [("HAM test", "tab:blue", 0.65), ("BCN20000", "tab:orange", 0.22)]:
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=alpha, color=color, label=label, linewidths=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
        save_figure(fig, RESULT_FIGURES / "fig_domain_shift_embedding_pca.png")
    else:
        pd.DataFrame(
            [
                {
                    "analysis": "domain_shift_embedding_pca",
                    "status": "skipped",
                    "note": "Embedding files were not available.",
                }
            ]
        ).to_csv(RESULT_AUDIT / "domain_shift_embedding_pca_skipped.csv", index=False)


def candidate_otsu_files():
    candidates = {
        "ham": [
            ROOT / "data/derived/features/ham_abc_otsu.csv",
            ROOT / "data/derived/features/ham_otsu_abc.csv",
            ROOT / "data/derived/features/ham_abc_baseline.csv",
        ],
        "bcn": [
            ROOT / "data/derived/features/bcn20000_abc_otsu.csv",
            ROOT / "data/derived/features/bcn20000_otsu_abc.csv",
            ROOT / "data/derived/features/bcn20000_abc_baseline.csv",
        ],
    }
    return {k: next((p for p in paths if p.exists()), None) for k, paths in candidates.items()}


def align_embeddings(ids, emb_npy, emb_ids_csv):
    E = np.load(emb_npy, mmap_mode="r")
    emb_ids = pd.read_csv(emb_ids_csv)["image_id"].tolist()
    idx = {image_id: i for i, image_id in enumerate(emb_ids)}
    keep_ids = [image_id for image_id in ids if image_id in idx]
    return keep_ids, E[[idx[i] for i in keep_ids]]


def evaluate_segmentation_feature_set(dataset, feature_path, abc_model, hybrid_model, hybrid_feat_cols):
    feats = pd.read_csv(feature_path)
    if dataset == "HAM test":
        ydf = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv")[["image_id", "y_true"]]
        emb_npy = ROOT / "data/derived/embeddings/ham_effb0.npy"
        emb_ids = ROOT / "data/derived/embeddings/ham_effb0_ids.csv"
    else:
        ydf = pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv")[["image_id", "y_true"]]
        emb_npy = ROOT / "data/derived/embeddings/bcn20000_effb0.npy"
        emb_ids = ROOT / "data/derived/embeddings/bcn20000_effb0_ids.csv"

    df = ydf.merge(feats, on="image_id", how="inner")
    abc_cols = abc_model["feat_cols"]
    rows = []
    p_abc = predict_proba(abc_model["model"], df[abc_cols].values)
    rows.append({"dataset": dataset, "model": "ABC-only", **model_metrics(df["y_true"], p_abc, include_calibration=False)})

    keep_ids, E = align_embeddings(df["image_id"].tolist(), emb_npy, emb_ids)
    df_h = df.set_index("image_id").loc[keep_ids].reset_index()
    X = np.concatenate([df_h[hybrid_feat_cols].values, E], axis=1)
    p_h = predict_proba(hybrid_model["model"], X)
    rows.append({"dataset": dataset, "model": "Hybrid", **model_metrics(df_h["y_true"], p_h, include_calibration=False)})
    return rows


def run_segmentation_sensitivity():
    print("Running segmentation sensitivity check...")
    otsu = candidate_otsu_files()
    levelset = {
        "ham": ROOT / "data/derived/features/ham_abc.csv",
        "bcn": ROOT / "data/derived/features/bcn20000_abc.csv",
    }
    if not all(p.exists() for p in levelset.values()) or otsu["ham"] is None or otsu["bcn"] is None:
        pd.DataFrame(
            [
                {
                    "analysis": "segmentation_sensitivity",
                    "status": "skipped",
                    "levelset_ham_features": str(levelset["ham"]) if levelset["ham"].exists() else "missing",
                    "levelset_bcn_features": str(levelset["bcn"]) if levelset["bcn"].exists() else "missing",
                    "otsu_ham_features": str(otsu["ham"]) if otsu["ham"] else "missing",
                    "otsu_bcn_features": str(otsu["bcn"]) if otsu["bcn"] else "missing",
                    "note": "Otsu-derived HAM and BCN feature files were not found; no classifier was refit and no segmentation-sensitivity result was fabricated.",
                }
            ]
        ).to_csv(RESULT_TABLES / "table_segmentation_sensitivity.csv", index=False)
        return

    abc_model = joblib.load(ROOT / "results/runs/handcrafted/handcrafted_xgb.joblib")
    hybrid_model = joblib.load(ROOT / "results/runs/hybrid/hybrid_xgb.joblib")
    rows = []
    for feature_source, paths in [
        ("LevelSet", levelset),
        ("Otsu", otsu),
    ]:
        for dataset, key in [("HAM test", "ham"), ("BCN20000", "bcn")]:
            part = evaluate_segmentation_feature_set(dataset, paths[key], abc_model, hybrid_model, hybrid_model["feat_cols"])
            for row in part:
                row["feature_source"] = feature_source
                rows.append(row)
    pd.DataFrame(rows).to_csv(RESULT_TABLES / "table_segmentation_sensitivity.csv", index=False)


class Task1MaskDataset(Dataset):
    def __init__(self, image_ids, image_dir, mask_dir, transform, image_size):
        self.image_ids = image_ids
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        x = self.transform(img)
        mask = Image.open(self.mask_dir / f"{image_id}.png").convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
        return x, torch.from_numpy(mask_arr), image_id


class BatchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        torch.sigmoid(logits).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        mins = flat.min(dim=1).values[:, None, None]
        maxs = flat.max(dim=1).values[:, None, None]
        cam = (cam - mins) / (maxs - mins + 1e-8)
        return cam.detach().cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_effnet_target_layer(model):
    target = None
    for name, module in model.named_modules():
        if "conv_head" in name:
            target = module
    if target is not None:
        return target
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return list(model.modules())[-1]


def gradcam_metrics(cam, mask):
    cam = np.asarray(cam, dtype=float)
    mask = np.asarray(mask).astype(bool)
    flat = cam.ravel()
    mask_flat = mask.ravel()
    k = max(1, int(math.ceil(0.20 * flat.size)))
    top_idx = np.argpartition(flat, -k)[-k:]
    top_inside = float(mask_flat[top_idx].mean())
    inside = cam[mask]
    outside = cam[~mask]
    inside_mean = float(inside.mean()) if inside.size else float("nan")
    outside_mean = float(outside.mean()) if outside.size else float("nan")
    ratio = inside_mean / (outside_mean + EPS) if np.isfinite(inside_mean) and np.isfinite(outside_mean) else float("nan")
    return {
        "top20_saliency_fraction_inside_mask": top_inside,
        "mean_saliency_inside_mask": inside_mean,
        "mean_saliency_outside_mask": outside_mean,
        "inside_outside_saliency_ratio": ratio,
        "mask_area_fraction": float(mask.mean()),
    }


def run_gradcam_localization(cfg, batch_size=16, limit=None):
    print("Running Grad-CAM localization sanity check...")
    import timm

    image_dir = Path(cfg["paths"]["isic_task1_images"])
    mask_dir = Path(cfg["paths"]["isic_task1_masks"])
    ids = sorted(
        p.stem
        for p in image_dir.glob("*.jpg")
        if (mask_dir / f"{p.stem}.png").exists()
    )
    if limit is not None:
        ids = ids[: int(limit)]
    if not ids:
        pd.DataFrame(
            [{"analysis": "gradcam_localization_sanity", "status": "skipped", "note": "No image-mask pairs were available."}]
        ).to_csv(RESULT_TABLES / "table_gradcam_localization_sanity.csv", index=False)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(cfg["training"]["deep"]["backbone"], pretrained=False, num_classes=1).to(device)
    ckpt = torch_load_checkpoint(ROOT / cfg["derived"]["deep_ckpt"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    target_layer = find_effnet_target_layer(model)
    cam_fn = BatchGradCAM(model, target_layer)
    transform = build_transforms(cfg["image"]["size"], cfg["image"]["normalize"], train=False)
    ds = Task1MaskDataset(ids, image_dir, mask_dir, transform, cfg["image"]["size"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    rows = []
    seen = 0
    for x, masks, batch_ids in dl:
        x = x.to(device)
        cams, probs = cam_fn(x)
        masks_np = masks.numpy().astype(bool)
        for i, image_id in enumerate(batch_ids):
            row = {
                "dataset": "ISIC2018_Task1_manual_masks",
                "image_id": image_id,
                "deep_probability": float(probs[i]),
            }
            row.update(gradcam_metrics(cams[i], masks_np[i]))
            rows.append(row)
        seen += len(batch_ids)
        if seen % 256 == 0:
            print(f"  Grad-CAM processed {seen} images")
    cam_fn.remove()

    per_image = pd.DataFrame(rows)
    per_image.to_csv(RESULT_AUDIT / "gradcam_localization_per_image.csv", index=False)
    summary_rows = []
    for metric in [
        "top20_saliency_fraction_inside_mask",
        "mean_saliency_inside_mask",
        "mean_saliency_outside_mask",
        "inside_outside_saliency_ratio",
        "mask_area_fraction",
    ]:
        values = per_image[metric].dropna().to_numpy()
        summary_rows.append(
            {
                "dataset": "ISIC2018_Task1_manual_masks",
                "metric": metric,
                "N": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(RESULT_TABLES / "table_gradcam_localization_sanity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(per_image["top20_saliency_fraction_inside_mask"], bins=30, color="0.25", alpha=0.85)
    axes[0].axvline(per_image["top20_saliency_fraction_inside_mask"].median(), color="tab:red", linewidth=2)
    axes[0].set_xlabel("Fraction of top 20% saliency inside mask")
    axes[0].set_ylabel("Images")
    axes[1].scatter(
        per_image["mask_area_fraction"],
        per_image["top20_saliency_fraction_inside_mask"],
        s=8,
        alpha=0.35,
        linewidths=0,
    )
    axes[1].set_xlabel("Mask area fraction")
    axes[1].set_ylabel("Fraction of top 20% saliency inside mask")
    save_figure(fig, RESULT_FIGURES / "fig_gradcam_mask_overlap.png")


def run_subgroup_calibration_bootstrap(n_boot=1000, seed=42):
    print("Running subgroup calibration bootstrap...")
    rng = np.random.default_rng(seed)
    df = load_bcn_metadata_with_predictions()
    models = {"ABC-only": "ABC-only_prob", "Deep": "Deep_prob", "Hybrid": "Hybrid_prob"}
    subgroup_cols = ["age_group", "sex", "anatom_site_general", "diagnosis_1", "diagnosis_2", "diagnosis_3"]
    rows = []
    for model_name, prob_col in models.items():
        for col in subgroup_cols:
            for level, g in df.groupby(col):
                if len(g) < 30:
                    continue
                y = g["y_true"].astype(int).to_numpy()
                p = g[prob_col].astype(float).to_numpy()
                briers = np.empty(n_boot, dtype=float)
                eces = np.empty(n_boot, dtype=float)
                n = len(y)
                for b in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    briers[b] = brier_score_loss(y[idx], p[idx])
                    eces[b] = expected_calibration_error(y[idx], p[idx])
                rows.append(
                    {
                        "dataset": "BCN20000",
                        "model": model_name,
                        "subgroup_variable": col,
                        "subgroup_level": level,
                        "N": int(n),
                        "positives": int(y.sum()),
                        "prevalence": float(y.mean()),
                        "Brier": float(brier_score_loss(y, p)),
                        "ECE": expected_calibration_error(y, p),
                        "Brier_boot_mean": float(briers.mean()),
                        "Brier_ci_low": float(np.quantile(briers, 0.025)),
                        "Brier_ci_high": float(np.quantile(briers, 0.975)),
                        "ECE_boot_mean": float(eces.mean()),
                        "ECE_ci_low": float(np.quantile(eces, 0.025)),
                        "ECE_ci_high": float(np.quantile(eces, 0.975)),
                        "bootstrap_resamples": int(n_boot),
                    }
                )
    pd.DataFrame(rows).to_csv(RESULT_TABLES / "table_subgroup_calibration_bootstrap.csv", index=False)


class ConditionalLogitCalibrator(nn.Module):
    def __init__(self, n_features, condition_alpha=True, condition_beta=True, eps=1e-4):
        super().__init__()
        self.alpha_0 = nn.Parameter(torch.tensor(0.0))
        self.beta_0 = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.condition_alpha = condition_alpha
        self.condition_beta = condition_beta
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, s, c):
        alpha_linear = self.alpha_0
        beta_linear = self.beta_0
        if self.condition_alpha:
            alpha_linear = alpha_linear + c.matmul(self.alpha)
        if self.condition_beta:
            beta_linear = beta_linear + c.matmul(self.beta)
        a = self.softplus(alpha_linear) + self.eps
        return torch.sigmoid(a * s + beta_linear)


def fit_conditional_calibrator(s, c, y, condition_alpha=True, condition_beta=True, lambdas=None, seed=42):
    if lambdas is None:
        lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    s_t = torch.tensor(s, dtype=torch.float32)
    c_t = torch.tensor(c, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    best = None
    for lam in lambdas:
        torch.manual_seed(seed)
        model = ConditionalLogitCalibrator(c.shape[1], condition_alpha=condition_alpha, condition_beta=condition_beta)
        opt = torch.optim.LBFGS(model.parameters(), max_iter=150, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            p = model(s_t, c_t)
            bce = nn.BCELoss()(p, y_t)
            l2 = lam * (torch.norm(model.alpha) ** 2 + torch.norm(model.beta) ** 2)
            loss = bce + l2
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            val_loss = nn.BCELoss()(model(s_t, c_t), y_t).item()
        if best is None or val_loss < best[0]:
            best = (val_loss, lam, model)
    return best[2], best[1]


def prepare_abccal_data(pred_csv, abc_csv):
    pred = pd.read_csv(pred_csv)
    feats = pd.read_csv(abc_csv)
    df = pred.merge(feats, on="image_id", how="inner")
    df["deep_logit"] = safe_logit(df["y_prob"])
    return df


def run_abccal_ablation(n_boot=1000, seed=42):
    print("Running ABC-conditioned score-correction ablations...")
    df_val = prepare_abccal_data(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv", ROOT / "data/derived/features/ham_abc.csv")
    df_test = prepare_abccal_data(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv", ROOT / "data/derived/features/ham_abc.csv")
    df_bcn = prepare_abccal_data(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv", ROOT / "data/derived/features/bcn20000_abc.csv")
    descriptor_cols = [c for c in df_val.columns if c.startswith(("A_", "B_", "C_"))]
    variant_specs = {
        "alpha-only": {"cols": descriptor_cols, "alpha": True, "beta": False},
        "beta-only": {"cols": descriptor_cols, "alpha": False, "beta": True},
        "A-only descriptors": {"cols": [c for c in descriptor_cols if c.startswith("A_")], "alpha": True, "beta": True},
        "B-only descriptors": {"cols": [c for c in descriptor_cols if c.startswith("B_")], "alpha": True, "beta": True},
        "C-only descriptors": {"cols": [c for c in descriptor_cols if c.startswith("C_")], "alpha": True, "beta": True},
        "GLCM-only descriptors": {"cols": [c for c in descriptor_cols if "glcm" in c.lower()], "alpha": True, "beta": True},
        "all descriptors": {"cols": descriptor_cols, "alpha": True, "beta": True},
    }

    y_val = df_val["y_true"].astype(int).to_numpy()
    s_val = df_val["deep_logit"].to_numpy()
    y_test = df_test["y_true"].astype(int).to_numpy()
    y_bcn = df_bcn["y_true"].astype(int).to_numpy()
    platt = LogisticRegression(max_iter=1000)
    platt.fit(s_val.reshape(-1, 1), y_val)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df_val["y_prob"].to_numpy(), y_val)

    predictions = {
        "HAM test": {
            "Raw": df_test["y_prob"].to_numpy(),
            "Platt": platt.predict_proba(df_test["deep_logit"].to_numpy().reshape(-1, 1))[:, 1],
            "Isotonic": iso.predict(df_test["y_prob"].to_numpy()),
        },
        "BCN20000": {
            "Raw": df_bcn["y_prob"].to_numpy(),
            "Platt": platt.predict_proba(df_bcn["deep_logit"].to_numpy().reshape(-1, 1))[:, 1],
            "Isotonic": iso.predict(df_bcn["y_prob"].to_numpy()),
        },
    }
    fitted_info = {}
    for name, spec in variant_specs.items():
        cols = spec["cols"]
        scaler = StandardScaler()
        c_val = scaler.fit_transform(df_val[cols].to_numpy())
        model, best_lambda = fit_conditional_calibrator(
            s_val,
            c_val,
            y_val,
            condition_alpha=spec["alpha"],
            condition_beta=spec["beta"],
            seed=seed,
        )
        fitted_info[name] = {"best_lambda": best_lambda, "n_descriptors": len(cols), "descriptors": ";".join(cols)}
        with torch.no_grad():
            for dataset, df in [("HAM test", df_test), ("BCN20000", df_bcn)]:
                c = scaler.transform(df[cols].to_numpy())
                p = model(
                    torch.tensor(df["deep_logit"].to_numpy(), dtype=torch.float32),
                    torch.tensor(c, dtype=torch.float32),
                ).numpy()
                predictions[dataset][name] = p

    rows = []
    y_by_dataset = {"HAM test": y_test, "BCN20000": y_bcn}
    for dataset, preds in predictions.items():
        y = y_by_dataset[dataset]
        for name, p in preds.items():
            m = model_metrics(y, p, include_calibration=True)
            rows.append(
                {
                    "dataset": dataset,
                    "score_correction": name,
                    "best_lambda": fitted_info.get(name, {}).get("best_lambda", np.nan),
                    "n_descriptors": fitted_info.get(name, {}).get("n_descriptors", 0),
                    "descriptors": fitted_info.get(name, {}).get("descriptors", ""),
                    **{
                        k: m[k]
                        for k in [
                            "AUC",
                            "PR-AUC",
                            "Brier",
                            "ECE",
                            "calibration_slope",
                            "calibration_intercept",
                            "mean_net_benefit_0.02_0.50",
                        ]
                    },
                }
            )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(RESULT_TABLES / "table_abccal_ablation.csv", index=False)

    bootstrap_metrics = ["AUC", "PR-AUC", "Brier", "ECE", "mean_net_benefit_0.02_0.50"]
    bcn_preds = predictions["BCN20000"]
    point = {name: model_metrics(y_bcn, p, include_calibration=False) for name, p in bcn_preds.items()}
    rng = np.random.default_rng(seed)
    boot_values = {name: {metric: np.empty(n_boot, dtype=float) for metric in bootstrap_metrics} for name in bcn_preds}
    names = list(bcn_preds.keys())
    n = len(y_bcn)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_bcn[idx]
        for name in names:
            p = bcn_preds[name][idx]
            mm = model_metrics(yb, p, include_calibration=False)
            for metric in bootstrap_metrics:
                boot_values[name][metric][b] = mm[metric]
        if (b + 1) % 200 == 0:
            print(f"  ABC-correction bootstrap {b + 1}/{n_boot}")

    comparisons = ["Raw", "Platt", "Isotonic", "all descriptors"]
    ablation_names = [name for name in variant_specs.keys()]
    diff_rows = []
    for variant in ablation_names:
        for comparator in comparisons:
            if variant == comparator:
                continue
            for metric in bootstrap_metrics:
                diff = boot_values[variant][metric] - boot_values[comparator][metric]
                diff_rows.append(
                    {
                        "dataset": "BCN20000",
                        "score_correction": variant,
                        "comparator": comparator,
                        "metric": metric,
                        "point_difference": point[variant][metric] - point[comparator][metric],
                        "bootstrap_mean_difference": float(np.nanmean(diff)),
                        "ci_low": float(np.nanquantile(diff, 0.025)),
                        "ci_high": float(np.nanquantile(diff, 0.975)),
                        "probability_difference_gt_0": float(np.nanmean(diff > 0)),
                        "bootstrap_resamples": int(n_boot),
                    }
                )
    pd.DataFrame(diff_rows).to_csv(RESULT_TABLES / "table_abccal_bootstrap_bcn.csv", index=False)

    bcn_plot = ablation[ablation["dataset"] == "BCN20000"].copy()
    order = ["Raw", "Platt", "Isotonic"] + list(variant_specs.keys())
    bcn_plot["score_correction"] = pd.Categorical(bcn_plot["score_correction"], categories=order, ordered=True)
    bcn_plot = bcn_plot.sort_values("score_correction")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "AUC", "AUC"),
        (axes[1], "Brier", "Brier score"),
        (axes[2], "mean_net_benefit_0.02_0.50", "Mean net benefit"),
    ]:
        ax.bar(bcn_plot["score_correction"].astype(str), bcn_plot[metric], color="0.35")
        ax.set_ylabel(ylabel)
    axes[-1].set_xticklabels(bcn_plot["score_correction"].astype(str), rotation=35, ha="right")
    save_figure(fig, RESULT_FIGURES / "fig_abccal_ablation_bcn.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps or all: split,bcn_subgroups,domain_shift,segmentation,gradcam,subgroup_bootstrap,abccal_ablation",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradcam-batch-size", type=int, default=16)
    parser.add_argument("--gradcam-limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_dirs()
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config()
    selected = (
        {"split", "bcn_subgroups", "domain_shift", "segmentation", "gradcam", "subgroup_bootstrap", "abccal_ablation"}
        if args.steps == "all"
        else {s.strip() for s in args.steps.split(",") if s.strip()}
    )

    if "split" in selected:
        run_split_integrity(cfg)
    if "bcn_subgroups" in selected:
        run_bcn_subgroup_analysis()
    if "domain_shift" in selected:
        run_domain_shift()
    if "segmentation" in selected:
        run_segmentation_sensitivity()
    if "gradcam" in selected:
        run_gradcam_localization(cfg, batch_size=args.gradcam_batch_size, limit=args.gradcam_limit)
    if "subgroup_bootstrap" in selected:
        run_subgroup_calibration_bootstrap(n_boot=args.bootstrap, seed=args.seed)
    if "abccal_ablation" in selected:
        run_abccal_ablation(n_boot=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
