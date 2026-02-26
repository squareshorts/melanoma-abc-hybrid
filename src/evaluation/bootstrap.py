import numpy as np
from src.evaluation.metrics import compute_basic

def bootstrap_ci(y_true, y_score, metric_key: str, n=1000, seed=42, thr=0.5):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    m = len(y_true)
    vals = []
    for _ in range(int(n)):
        idx = rng.integers(0, m, size=m)
        stats = compute_basic(y_true[idx], y_score[idx], thr=thr)
        v = stats.get(metric_key, np.nan)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
