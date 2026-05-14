"""Split conformal prediction for binary image classification.

Provides guaranteed marginal coverage at level 1-alpha using a held-out
calibration set.  Works with any classifier that outputs class probabilities.

Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification" (2022).
"""
from __future__ import annotations

import numpy as np


class SplitConformalPredictor:
    """Inductive conformal predictor using the 1-softmax nonconformity score.

    Usage::

        cal_probs  # (N_cal, C) softmax probabilities on calibration set
        cal_labels # (N_cal,) integer labels

        cp = SplitConformalPredictor(alpha=0.10)
        cp.calibrate(cal_probs, cal_labels)

        test_sets = cp.predict_sets(test_probs)  # list of lists
        metrics   = cp.evaluate(test_probs, test_labels)
    """

    def __init__(self, alpha: float = 0.10):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.threshold: float | None = None
        self.n_cal: int = 0

    # ------------------------------------------------------------------
    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> "SplitConformalPredictor":
        """Set the conformal threshold from a calibration set.

        Uses the finite-sample corrected quantile so that empirical coverage
        on fresh test data is ≥ 1 - alpha with high probability.
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        if n < 10:
            raise ValueError("Need at least 10 calibration samples.")
        scores = 1.0 - probs[np.arange(n), labels]   # nonconformity score
        # Finite-sample correction: ceil((n+1)(1-α))/n
        q_level = float(np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        q_level = min(q_level, 1.0)
        self.threshold = float(np.quantile(scores, q_level, method="higher"))
        self.n_cal = n
        return self

    # ------------------------------------------------------------------
    def predict_sets(self, probs: np.ndarray) -> list[list[int]]:
        """Return a prediction set for each sample.

        A class is included when its nonconformity score ≤ threshold,
        i.e. when its softmax probability ≥ 1 - threshold.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() first.")
        probs = np.asarray(probs, dtype=np.float64)
        return [
            list(np.where(1.0 - row <= self.threshold)[0])
            for row in probs
        ]

    # ------------------------------------------------------------------
    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """Return coverage and efficiency metrics on a test set."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate() first.")
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        sets = self.predict_sets(probs)
        sizes = np.array([len(s) for s in sets])
        n_classes = probs.shape[1]

        covered = np.array([int(y in s) for y, s in zip(labels, sets)])
        coverage = float(covered.mean())
        avg_size = float(sizes.mean())
        singleton_rate = float((sizes == 1).mean())
        empty_rate = float((sizes == 0).mean())
        full_rate = float((sizes == n_classes).mean())

        # Conditional coverage: coverage split by correct / incorrect
        # base prediction (argmax)
        base_pred = probs.argmax(axis=1)
        correct_mask = base_pred == labels
        cov_correct = float(covered[correct_mask].mean()) if correct_mask.any() else None
        cov_incorrect = float(covered[~correct_mask].mean()) if (~correct_mask).any() else None

        return {
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": coverage,
            "coverage_gap": round(coverage - (1.0 - self.alpha), 6),
            "threshold": float(self.threshold),
            "n_cal": self.n_cal,
            "n_test": len(labels),
            "avg_prediction_set_size": avg_size,
            "singleton_rate": singleton_rate,
            "empty_set_rate": empty_rate,
            "full_set_rate": full_rate,
            "conditional_coverage_correct": cov_correct,
            "conditional_coverage_incorrect": cov_incorrect,
        }


def conformal_across_alphas(
    probs: np.ndarray,
    labels: np.ndarray,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alphas: list[float] | None = None,
) -> list[dict]:
    """Evaluate conformal predictor at multiple alpha levels.

    Returns a list of metric dicts, one per alpha, suitable for plotting
    coverage vs set-size curves.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
    rows = []
    for a in alphas:
        cp = SplitConformalPredictor(alpha=a)
        cp.calibrate(cal_probs, cal_labels)
        rows.append(cp.evaluate(probs, labels))
    return rows
