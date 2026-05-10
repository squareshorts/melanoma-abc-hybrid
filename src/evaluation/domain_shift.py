"""Domain-shift hypothesis testing utilities.

Provides:
- Logistic-regression domain classifier
- Maximum Mean Discrepancy (MMD) with RBF kernel
- Energy distance
All with bootstrap confidence intervals.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------
# Domain classifier
# -----------------------------------------------------------------------
def train_domain_classifier(X_source, X_target, n_permutations=500, seed=42):
    """Train a logistic regression to distinguish source from target.

    Returns dict with accuracy, AUC, and permutation-test p-value.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([X_source, X_target]).astype(np.float32)
    y = np.array([0] * len(X_source) + [1] * len(X_target))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated predictions for unbiased estimate
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    proba = cross_val_predict(clf, X_scaled, y, cv=5, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float(accuracy_score(y, pred))
    auc = float(roc_auc_score(y, proba))

    # Permutation test for AUC
    perm_aucs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            perm_aucs[i] = roc_auc_score(y_perm, proba)
        except ValueError:
            perm_aucs[i] = 0.5
    p_value = float(np.mean(perm_aucs >= auc))

    return {
        "accuracy": acc,
        "AUC": auc,
        "permutation_p_value": p_value,
        "n_permutations": n_permutations,
    }


# -----------------------------------------------------------------------
# MMD (Maximum Mean Discrepancy)
# -----------------------------------------------------------------------
def _rbf_kernel(X, Y, gamma):
    """Compute RBF kernel matrix between X and Y."""
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    dist_sq = XX + YY - 2.0 * X @ Y.T
    return np.exp(-gamma * dist_sq)


def compute_mmd(X, Y, gamma=None, n_boot=1000, seed=42):
    """Compute MMD^2 with RBF kernel and bootstrap CI.

    Parameters
    ----------
    X, Y : arrays of shape (n, d) and (m, d)
    gamma : RBF bandwidth; if None, uses median heuristic
    n_boot : number of bootstrap resamples
    seed : random seed

    Returns
    -------
    dict with mmd2, ci_low, ci_high
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if gamma is None:
        # Median heuristic: gamma = 1 / (2 * median_distance^2)
        sub_x = X[rng.choice(len(X), min(500, len(X)), replace=False)]
        sub_y = Y[rng.choice(len(Y), min(500, len(Y)), replace=False)]
        dists = np.sqrt(np.sum((sub_x[:, None] - sub_y[None, :]) ** 2, axis=2))
        median_dist = float(np.median(dists))
        gamma = 1.0 / (2.0 * median_dist ** 2 + 1e-8)

    def _mmd2(x, y):
        Kxx = _rbf_kernel(x, x, gamma)
        Kyy = _rbf_kernel(y, y, gamma)
        Kxy = _rbf_kernel(x, y, gamma)
        n, m = len(x), len(y)
        return (
            float(np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
            + (float(np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1)))
            - 2.0 * float(np.mean(Kxy))
        )

    # Subsample for speed if large
    max_n = 2000
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    point = _mmd2(X, Y)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ix = rng.integers(0, len(X), size=len(X))
        iy = rng.integers(0, len(Y), size=len(Y))
        boots[b] = _mmd2(X[ix], Y[iy])

    return {
        "mmd2": float(point),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "gamma": float(gamma),
        "n_boot": n_boot,
    }


# -----------------------------------------------------------------------
# Energy distance
# -----------------------------------------------------------------------
def compute_energy_distance(X, Y, n_boot=1000, seed=42):
    """Compute energy distance with bootstrap CI.

    E(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Subsample for speed
    max_n = 2000
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def _energy(x, y):
        from scipy.spatial.distance import cdist
        # Pairwise distances (efficient)
        dxy = cdist(x, y, metric='euclidean')
        dxx = cdist(x, x, metric='euclidean')
        dyy = cdist(y, y, metric='euclidean')
        return 2.0 * np.mean(dxy) - np.mean(dxx) - np.mean(dyy)

    point = _energy(X, Y)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ix = rng.integers(0, len(X), size=len(X))
        iy = rng.integers(0, len(Y), size=len(Y))
        boots[b] = _energy(X[ix], Y[iy])

    return {
        "energy_distance": float(point),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n_boot": n_boot,
    }
