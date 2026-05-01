import numpy as np
from src.evaluation.metrics import compute_basic

def bootstrap_ci(y_true, y_score, metric_key: str, n=1000, seed=42, thr=0.5, groups=None):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    m = len(y_true)
    vals = []
    
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
        
    for _ in range(int(n)):
        if groups is not None:
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        else:
            idx = rng.integers(0, m, size=m)
        
        stats = compute_basic(y_true[idx], y_score[idx], thr=thr)
        v = stats.get(metric_key, np.nan)
        if np.isfinite(v):
            vals.append(v)
            
    if len(vals) < 10:
        return (np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
