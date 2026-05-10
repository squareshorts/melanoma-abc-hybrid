"""Beta calibration wrapper with inline fallback."""

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

EPS = 1e-7


def _safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class BetaCalibration(BaseEstimator):
    """Three-parameter beta calibration (Kull et al., 2017).

    Fits:  logit(q) = a * log(p) + b * log(1-p) + c

    Falls back to an inline implementation if the ``betacal`` package is
    not installed.
    """

    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.c_ = None
        self._use_package = False

    # ------------------------------------------------------------------
    def fit(self, y_prob, y_true):
        y = np.asarray(y_true, dtype=int)
        p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)

        try:
            from betacal import BetaCalibration as _BC

            self._bc = _BC(parameters="abm")
            self._bc.fit(p.reshape(-1, 1), y)
            self._use_package = True
            return self
        except ImportError:
            pass

        # Inline implementation ------------------------------------------
        log_p = np.log(p)
        log_1mp = np.log(1.0 - p)

        def neg_ll(params):
            a, b, c = params
            logit_q = a * log_p + b * log_1mp + c
            q = 1.0 / (1.0 + np.exp(-logit_q))
            q = np.clip(q, EPS, 1.0 - EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize(neg_ll, x0=[1.0, -1.0, 0.0], method="L-BFGS-B")
        self.a_, self.b_, self.c_ = res.x
        return self

    # ------------------------------------------------------------------
    def predict(self, y_prob):
        p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
        if self._use_package:
            return self._bc.predict(p.reshape(-1, 1))
        logit_q = self.a_ * np.log(p) + self.b_ * np.log(1.0 - p) + self.c_
        return 1.0 / (1.0 + np.exp(-logit_q))
