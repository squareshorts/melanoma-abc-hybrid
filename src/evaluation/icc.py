"""Intraclass Correlation Coefficient — ICC(2,1) two-way random, absolute agreement."""

import numpy as np


def icc_2way_agreement(ratings):
    """Compute ICC(2,1) — two-way random, absolute agreement.

    Parameters
    ----------
    ratings : array-like of shape (n_subjects, k_raters)
        Each row is a subject, each column is a rater/method.
        NaN rows are dropped.

    Returns
    -------
    dict with icc, ci_low, ci_high (F-based approximate 95% CI),
    MSR, MSE, MSC, n, k.
    """
    R = np.asarray(ratings, dtype=float)
    # Drop rows with any NaN
    valid = ~np.isnan(R).any(axis=1)
    R = R[valid]
    n, k = R.shape
    if n < 3 or k < 2:
        return {"icc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(n), "k": int(k)}

    grand_mean = R.mean()
    row_means = R.mean(axis=1)
    col_means = R.mean(axis=0)

    SST = np.sum((R - grand_mean) ** 2)
    SSR = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
    SSC = n * np.sum((col_means - grand_mean) ** 2)  # between raters
    SSE = SST - SSR - SSC                            # residual

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    # ICC(2,1)
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    # F-based 95% CI (Shrout & Fleiss 1979, McGraw & Wong 1996)
    F_val = MSR / MSE if MSE > 0 else float("inf")
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    from scipy.stats import f as f_dist
    F_lo = F_val / f_dist.ppf(0.975, df1, df2)
    F_hi = F_val / f_dist.ppf(0.025, df1, df2)

    ci_low = (F_lo - 1) / (F_lo + k - 1)
    ci_high = (F_hi - 1) / (F_hi + k - 1)

    return {
        "icc": float(icc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "MSR": float(MSR),
        "MSE": float(MSE),
        "MSC": float(MSC),
        "n": int(n),
        "k": int(k),
    }
