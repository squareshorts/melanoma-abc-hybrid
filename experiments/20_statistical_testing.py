import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
AUDIT_DIR = ROOT / "results" / "audit"
EPS = 1e-7


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(y)) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def calibration_slope_intercept(y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2 or len(np.unique(np.round(p, 12))) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(safe_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def delong_components(predictions, labels):
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(float)
    m = int(np.sum(labels == 1))
    n = int(np.sum(labels == 0))
    if m == 0 or n == 0:
        return float("nan"), None, None
    pos = predictions[labels == 1]
    neg = predictions[labels == 0]
    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(predictions)
    auc = (np.sum(tz[labels == 1]) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[labels == 1] - tx) / n
    v10 = 1.0 - (tz[labels == 0] - ty) / m
    return float(auc), v01, v10


def paired_delong_test(y_true, p_a, p_b):
    y = np.asarray(y_true).astype(int)
    p_a = np.asarray(p_a).astype(float)
    p_b = np.asarray(p_b).astype(float)
    m = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    auc_a, v01_a, v10_a = delong_components(p_a, y)
    auc_b, v01_b, v10_b = delong_components(p_b, y)
    if v01_a is None:
        return {
            "auc_a": auc_a,
            "auc_b": auc_b,
            "auc_difference_a_minus_b": float("nan"),
            "se_difference": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    cov01 = np.cov(v01_a, v01_b, ddof=1)[0, 1]
    cov10 = np.cov(v10_a, v10_b, ddof=1)[0, 1]
    var_a = np.var(v01_a, ddof=1) / m + np.var(v10_a, ddof=1) / n
    var_b = np.var(v01_b, ddof=1) / m + np.var(v10_b, ddof=1) / n
    cov = cov01 / m + cov10 / n
    var_diff = max(float(var_a + var_b - 2 * cov), 1e-14)
    diff = float(auc_a - auc_b)
    se = math.sqrt(var_diff)
    z = diff / se if se > 0 else float("nan")
    p_value = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_difference_a_minus_b": diff,
        "se_difference": se,
        "z": float(z),
        "p_value": p_value,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
    }


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def load_prediction_file(path, model_name):
    df = pd.read_csv(path)
    return df[["image_id", "y_true", "y_prob"]].rename(columns={"y_prob": model_name})


def load_main_model_dataset(dataset_key):
    paths = {
        "HAM test": {
            "ABC-only": ROOT / "results/runs/handcrafted/ham_test_predictions_handcrafted.csv",
            "Deep": ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv",
            "Hybrid": ROOT / "results/runs/hybrid/ham_test_predictions_hybrid.csv",
        },
        "BCN20000": {
            "ABC-only": ROOT / "results/runs/handcrafted/bcn20000_predictions_handcrafted.csv",
            "Deep": ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv",
            "Hybrid": ROOT / "results/runs/hybrid/bcn20000_predictions_hybrid.csv",
        },
        "ISIC Task 3 label audit": {
            "ABC-only": ROOT / "results/runs/handcrafted/isic_task3_predictions_handcrafted.csv",
            "Deep": ROOT / "results/runs/deep_baseline/isic_task3_predictions_deep.csv",
            "Hybrid": ROOT / "results/runs/hybrid/isic_task3_predictions_hybrid.csv",
        },
    }[dataset_key]
    merged = None
    for model, path in paths.items():
        df = load_prediction_file(path, model)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["image_id", "y_true", model]], on=["image_id", "y_true"], how="inner")
    return merged, list(paths.keys())


def run_delong_tests():
    rows = []
    for dataset in ["HAM test", "BCN20000", "ISIC Task 3 label audit"]:
        df, models = load_main_model_dataset(dataset)
        y = df["y_true"].astype(int).to_numpy()
        for model_a, model_b in itertools.combinations(models, 2):
            result = paired_delong_test(y, df[model_a], df[model_b])
            rows.append(
                {
                    "analysis": "main_model_auc",
                    "dataset": dataset,
                    "N": int(len(df)),
                    "positives": int(y.sum()),
                    "model_a": model_a,
                    "model_b": model_b,
                    **result,
                }
            )
    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_adjust(out["p_value"].to_numpy())
    out.to_csv(TABLE_DIR / "table_delong_auc_tests.csv", index=False)


class ConditionalLogitCalibrator(nn.Module):
    def __init__(self, n_features, eps=1e-4):
        super().__init__()
        self.alpha_0 = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(n_features))
        self.beta_0 = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self, s, c):
        a = self.softplus(self.alpha_0 + c.matmul(self.alpha)) + self.eps
        b = self.beta_0 + c.matmul(self.beta)
        return torch.sigmoid(a * s + b)


def fit_abccalibrator(s_train, c_train, y_train, lambdas=None, seed=42):
    if lambdas is None:
        lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    s_t = torch.tensor(s_train, dtype=torch.float32)
    c_t = torch.tensor(c_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    best = None
    for lam in lambdas:
        torch.manual_seed(seed)
        model = ConditionalLogitCalibrator(c_train.shape[1])
        opt = torch.optim.LBFGS(model.parameters(), max_iter=150, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = model(s_t, c_t)
            loss = nn.BCELoss()(pred, y_t) + lam * (torch.norm(model.alpha) ** 2 + torch.norm(model.beta) ** 2)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            val_loss = float(nn.BCELoss()(model(s_t, c_t), y_t).item())
        if best is None or val_loss < best[0]:
            best = (val_loss, lam, model)
    return best[2], best[1]


def prepare_abccal_predictions(seed=42):
    ham_abc = pd.read_csv(ROOT / "data/derived/features/ham_abc.csv")
    bcn_abc = pd.read_csv(ROOT / "data/derived/features/bcn20000_abc.csv")
    val = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_val_predictions_deep.csv").merge(ham_abc, on="image_id")
    test = pd.read_csv(ROOT / "results/runs/deep_baseline/ham_test_predictions_deep.csv").merge(ham_abc, on="image_id")
    bcn = pd.read_csv(ROOT / "results/runs/deep_baseline/bcn20000_predictions_deep.csv").merge(bcn_abc, on="image_id")
    feat_cols = [c for c in ham_abc.columns if c.startswith(("A_", "B_", "C_"))]

    for df in [val, test, bcn]:
        df["deep_logit"] = safe_logit(df["y_prob"])

    y_val = val["y_true"].astype(int).to_numpy()
    platt = LogisticRegression(max_iter=1000)
    platt.fit(val["deep_logit"].to_numpy().reshape(-1, 1), y_val)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val["y_prob"].to_numpy(), y_val)

    scaler = StandardScaler()
    c_val = scaler.fit_transform(val[feat_cols].to_numpy())
    abc_model, best_lambda = fit_abccalibrator(val["deep_logit"].to_numpy(), c_val, y_val, seed=seed)

    out = {}
    for dataset, df in [("HAM test", test), ("BCN20000", bcn)]:
        c = scaler.transform(df[feat_cols].to_numpy())
        with torch.no_grad():
            p_abc = abc_model(
                torch.tensor(df["deep_logit"].to_numpy(), dtype=torch.float32),
                torch.tensor(c, dtype=torch.float32),
            ).numpy()
        out[dataset] = pd.DataFrame(
            {
                "image_id": df["image_id"],
                "y_true": df["y_true"].astype(int),
                "Raw": df["y_prob"].to_numpy(),
                "Platt": platt.predict_proba(df["deep_logit"].to_numpy().reshape(-1, 1))[:, 1],
                "Isotonic": iso.predict(df["y_prob"].to_numpy()),
                "ABC-conditioned": p_abc,
            }
        )
    pd.DataFrame([{"best_lambda": best_lambda, "n_descriptors": len(feat_cols), "descriptors": ";".join(feat_cols)}]).to_csv(
        AUDIT_DIR / "abccal_statistical_testing_fit.csv", index=False
    )
    return out


def run_abccal_delong(seed=42):
    rows = []
    pred_by_dataset = prepare_abccal_predictions(seed=seed)
    for dataset, df in pred_by_dataset.items():
        models = ["Raw", "Platt", "Isotonic", "ABC-conditioned"]
        y = df["y_true"].to_numpy().astype(int)
        for model_a, model_b in itertools.combinations(models, 2):
            result = paired_delong_test(y, df[model_a], df[model_b])
            rows.append(
                {
                    "analysis": "deep_score_correction_auc",
                    "dataset": dataset,
                    "N": int(len(df)),
                    "positives": int(y.sum()),
                    "model_a": model_a,
                    "model_b": model_b,
                    **result,
                }
            )
    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_adjust(out["p_value"].to_numpy())
    out.to_csv(TABLE_DIR / "table_abccal_delong_auc_tests.csv", index=False)
    return pred_by_dataset


def calibration_curve_table(dataset, model, y_true, y_prob, n_bins=10, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    order = np.argsort(p)
    bin_indices = np.array_split(order, n_bins)
    rows = []
    for bin_id, idx in enumerate(bin_indices, 1):
        if len(idx) == 0:
            continue
        yy = y[idx]
        pp = p[idx]
        boot_pred = np.empty(n_boot, dtype=float)
        boot_obs = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sample = rng.integers(0, len(idx), size=len(idx))
            boot_pred[b] = pp[sample].mean()
            boot_obs[b] = yy[sample].mean()
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "bin": bin_id,
                "binning": "equal_frequency",
                "N": int(len(idx)),
                "positives": int(yy.sum()),
                "predicted_mean": float(pp.mean()),
                "predicted_mean_ci_low": float(np.quantile(boot_pred, 0.025)),
                "predicted_mean_ci_high": float(np.quantile(boot_pred, 0.975)),
                "observed_fraction": float(yy.mean()),
                "observed_fraction_ci_low": float(np.quantile(boot_obs, 0.025)),
                "observed_fraction_ci_high": float(np.quantile(boot_obs, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary_row(dataset, model, y_true, y_prob):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "dataset": dataset,
        "model": model,
        "N": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "PR-AUC": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": expected_calibration_error(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def plot_calibration_curves(curves, dataset, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, g in curves[curves["dataset"] == dataset].groupby("model", sort=False):
        x = g["predicted_mean"].to_numpy()
        y = g["observed_fraction"].to_numpy()
        yerr = np.vstack(
            [
                y - g["observed_fraction_ci_low"].to_numpy(),
                g["observed_fraction_ci_high"].to_numpy() - y,
            ]
        )
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=2.5, label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed melanoma fraction")
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, out_path)


def run_calibration_curves(n_boot=1000, seed=42, abccal_predictions=None):
    curve_rows = []
    summary_rows = []
    for dataset in ["HAM test", "BCN20000"]:
        df, models = load_main_model_dataset(dataset)
        y = df["y_true"].astype(int).to_numpy()
        for model in models:
            p = df[model].to_numpy()
            curve_rows.append(calibration_curve_table(dataset, model, y, p, n_boot=n_boot, seed=seed))
            summary_rows.append(calibration_summary_row(dataset, model, y, p))
    curves = pd.concat(curve_rows, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)
    curves.to_csv(TABLE_DIR / "table_calibration_curve_bins.csv", index=False)
    summaries.to_csv(TABLE_DIR / "table_calibration_summary_stats.csv", index=False)
    plot_calibration_curves(curves, "HAM test", FIGURE_DIR / "fig_calibration_curves_ham_ci.png")
    plot_calibration_curves(curves, "BCN20000", FIGURE_DIR / "fig_calibration_curves_bcn_ci.png")

    if abccal_predictions is not None:
        rows = []
        summary = []
        for dataset, df in abccal_predictions.items():
            y = df["y_true"].astype(int).to_numpy()
            for model in ["Raw", "Platt", "Isotonic", "ABC-conditioned"]:
                p = df[model].to_numpy()
                rows.append(calibration_curve_table(dataset, model, y, p, n_boot=n_boot, seed=seed))
                summary.append(calibration_summary_row(dataset, model, y, p))
        abccal_curves = pd.concat(rows, ignore_index=True)
        abccal_summary = pd.DataFrame(summary)
        abccal_curves.to_csv(TABLE_DIR / "table_abccal_calibration_curve_bins.csv", index=False)
        abccal_summary.to_csv(TABLE_DIR / "table_abccal_calibration_summary_stats.csv", index=False)
        plot_calibration_curves(abccal_curves, "HAM test", FIGURE_DIR / "fig_abccal_calibration_curves_ham_ci.png")
        plot_calibration_curves(abccal_curves, "BCN20000", FIGURE_DIR / "fig_abccal_calibration_curves_bcn_ci.png")


def paired_bootstrap_calibration_tests(n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in ["HAM test", "BCN20000"]:
        df, models = load_main_model_dataset(dataset)
        y = df["y_true"].astype(int).to_numpy()
        n = len(y)
        point = {
            model: {
                "Brier": float(brier_score_loss(y, df[model])),
                "ECE": expected_calibration_error(y, df[model]),
            }
            for model in models
        }
        boot = {model: {"Brier": np.empty(n_boot), "ECE": np.empty(n_boot)} for model in models}
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yy = y[idx]
            for model in models:
                pp = df[model].to_numpy()[idx]
                boot[model]["Brier"][b] = brier_score_loss(yy, pp)
                boot[model]["ECE"][b] = expected_calibration_error(yy, pp)
        for model_a, model_b in itertools.combinations(models, 2):
            for metric in ["Brier", "ECE"]:
                diff = boot[model_a][metric] - boot[model_b][metric]
                rows.append(
                    {
                        "dataset": dataset,
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "point_difference_a_minus_b": point[model_a][metric] - point[model_b][metric],
                        "bootstrap_mean_difference": float(np.mean(diff)),
                        "ci_low": float(np.quantile(diff, 0.025)),
                        "ci_high": float(np.quantile(diff, 0.975)),
                        "probability_difference_gt_0": float(np.mean(diff > 0)),
                        "bootstrap_resamples": int(n_boot),
                    }
                )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_calibration_stat_tests.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.chdir(ROOT)
    ensure_dirs()
    run_delong_tests()
    abccal_predictions = run_abccal_delong(seed=args.seed)
    run_calibration_curves(n_boot=args.bootstrap, seed=args.seed, abccal_predictions=abccal_predictions)
    paired_bootstrap_calibration_tests(n_boot=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
