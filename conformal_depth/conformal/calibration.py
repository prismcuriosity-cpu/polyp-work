"""
Conformal calibration for dense depth prediction.

Implements:
1. Standard split-conformal calibration (Venn-Abers / RCPS variant).
2. The finite-sample coverage guarantee:
       P(y ∈ [ŷ_lo − q̂, ŷ_hi + q̂]) ≥ 1 − α
   where q̂ = quantile_{⌈(n+1)(1-α)⌉/n}(s₁, …, sₙ).

3. Risk-Controlling Prediction Sets (RCPS, Angelopoulos 2022) variant that
   controls a user-specified risk functional R(λ) ≤ α at level δ using a
   one-sided concentration inequality (Bentkus/Hoeffding).

Reference: Angelopoulos et al., "Learn then Test: Calibrating Predictive
Algorithms to Achieve Risk Control", ICML 2022.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.stats as stats


# ---------------------------------------------------------------------------
# Standard split-conformal threshold
# ---------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal quantile q̂ from calibration nonconformity scores.

    Coverage guarantee: P(new sample conforms) ≥ 1 − α.

    Args:
        scores: (n,) array of nonconformity scores on the calibration set.
        alpha:  Miscoverage level (e.g. 0.10 for 90% coverage).

    Returns:
        q̂ — the threshold to subtract/add from the predicted bounds.
    """
    n = len(scores)
    # Finite-sample correction: quantile at level ⌈(n+1)(1-α)⌉/n
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level))


# ---------------------------------------------------------------------------
# RCPS (Risk-Controlling Prediction Sets)
# ---------------------------------------------------------------------------

class RCPSCalibrator:
    """
    Risk-Controlling Prediction Sets (Angelopoulos et al., ICML 2022).

    Finds the smallest λ* from a finite grid Λ such that the empirical risk
    R̂(λ) satisfies R̂(λ) ≤ α with probability ≥ 1 − δ.

    The risk functional is the *mean per-frame miscoverage rate*:
        R(λ) = E[1{ŷ_lo(x) − λ ≥ y  or  y ≥ ŷ_hi(x) + λ}]

    The bound uses Bentkus' inequality (tighter than Hoeffding for bounded
    risks) to inflate the empirical risk.

    Usage::

        calibrator = RCPSCalibrator(alpha=0.10, delta=0.05)
        lambda_star = calibrator.fit(scores, lambda_grid)
        # At test time: padded_lo = pred_lo - lambda_star
        #               padded_hi = pred_hi + lambda_star
    """

    def __init__(self, alpha: float = 0.10, delta: float = 0.05):
        self.alpha = alpha
        self.delta = delta

    def _bentkus_bound(self, r_hat: float, n: int) -> float:
        """
        Upper confidence bound on R(λ) via Bentkus' inequality.

        UCB(r̂, n, δ) = r̂ + √(r̂ · 2log(1/δ) / n) + 2log(1/δ) / n
        (simplified Bentkus — see Maurer & Pontil 2009 for exact form)
        """
        log_inv_delta = math.log(1.0 / self.delta)
        term = math.sqrt(r_hat * 2 * log_inv_delta / n)
        return r_hat + term + 2 * log_inv_delta / n

    def fit(
        self,
        scores: np.ndarray,         # (n,) nonconformity scores
        lambda_grid: np.ndarray,    # sorted ascending candidate λ values
    ) -> float:
        """
        Returns the smallest λ* in lambda_grid such that RCPS bound holds.

        When no λ in the grid is sufficient, returns lambda_grid.max()
        and emits a warning (the grid may need to be extended).
        """
        n = len(scores)

        for lam in sorted(lambda_grid):
            # Miscoverage at this λ: fraction of calibration frames not covered
            r_hat = float(np.mean(scores > lam))
            ucb   = self._bentkus_bound(r_hat, n)
            if ucb <= self.alpha:
                return float(lam)

        import warnings
        warnings.warn(
            f"RCPS: no λ in grid achieved R̂(λ) ≤ {self.alpha}. "
            "Consider extending the grid. Returning grid maximum.",
            RuntimeWarning,
        )
        return float(lambda_grid.max())


# ---------------------------------------------------------------------------
# Adaptive grid search (for when we don't know the score range a priori)
# ---------------------------------------------------------------------------

def build_lambda_grid(
    scores: np.ndarray,
    n_points: int = 200,
    margin: float = 1.5,
) -> np.ndarray:
    """
    Build a λ grid spanning [0, margin * max(scores)] with n_points values.

    Args:
        scores: Calibration set nonconformity scores.
        n_points: Grid resolution.
        margin: Extend grid beyond max score by this factor.

    Returns:
        (n_points,) sorted numpy array.
    """
    lo, hi = 0.0, float(scores.max()) * margin
    return np.linspace(lo, hi, n_points)


# ---------------------------------------------------------------------------
# Multi-alpha calibration (calibrate for several coverage levels at once)
# ---------------------------------------------------------------------------

class MultiAlphaCalibrator:
    """
    Calibrate conformal thresholds for a range of miscoverage levels α.

    This is useful to plot coverage vs. interval-width trade-off curves.
    """

    def __init__(self, alphas: list[float] | None = None):
        self.alphas = alphas or [0.05, 0.10, 0.15, 0.20]

    def fit(self, scores: np.ndarray) -> dict[float, float]:
        """Returns {alpha: q_hat} for each α in self.alphas."""
        return {a: conformal_quantile(scores, a) for a in self.alphas}


# ---------------------------------------------------------------------------
# Calibration state — serializable result
# ---------------------------------------------------------------------------

class ConformalCalibrationResult:
    """
    Holds the calibration output and exposes threshold look-up.

    Attributes:
        q_hat:       Standard conformal threshold.
        lambda_star: RCPS threshold (if computed).
        scores:      Raw calibration scores (for diagnostics).
        alpha:       Target miscoverage level.
        n_calib:     Calibration set size.
    """

    def __init__(
        self,
        scores: np.ndarray,
        alpha: float,
        q_hat: float,
        lambda_star: float | None = None,
    ):
        self.scores      = scores
        self.alpha       = alpha
        self.q_hat       = q_hat
        self.lambda_star = lambda_star
        self.n_calib     = len(scores)

    @property
    def threshold(self) -> float:
        """Returns RCPS threshold if available, else standard q̂."""
        return self.lambda_star if self.lambda_star is not None else self.q_hat

    def empirical_coverage(self, test_scores: np.ndarray) -> float:
        """Fraction of test scores ≤ threshold (should be ≥ 1-alpha)."""
        return float(np.mean(test_scores <= self.threshold))

    def __repr__(self) -> str:
        return (
            f"ConformalCalibrationResult("
            f"alpha={self.alpha}, q_hat={self.q_hat:.4f}, "
            f"lambda_star={self.lambda_star}, n_calib={self.n_calib})"
        )

    def save(self, path: str) -> None:
        import json
        d = {
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "lambda_star": self.lambda_star,
            "n_calib": self.n_calib,
            "score_mean": float(self.scores.mean()),
            "score_std":  float(self.scores.std()),
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrationResult":
        import json
        with open(path) as f:
            d = json.load(f)
        dummy_scores = np.array([d["q_hat"]])   # reconstruct placeholder
        return cls(
            scores=dummy_scores,
            alpha=d["alpha"],
            q_hat=d["q_hat"],
            lambda_star=d.get("lambda_star"),
        )


# ---------------------------------------------------------------------------
# Convenience: run full calibration pipeline
# ---------------------------------------------------------------------------

def run_calibration(
    scores: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.05,
    use_rcps: bool = True,
) -> ConformalCalibrationResult:
    """
    Run both standard and RCPS calibration.

    Args:
        scores:    (n,) per-frame nonconformity scores from calibration set.
        alpha:     Target miscoverage level.
        delta:     RCPS confidence level (probability of exceeding α).
        use_rcps:  Whether to also compute RCPS λ*.

    Returns:
        ConformalCalibrationResult
    """
    q_hat = conformal_quantile(scores, alpha)

    lambda_star = None
    if use_rcps:
        grid = build_lambda_grid(scores)
        rcps = RCPSCalibrator(alpha=alpha, delta=delta)
        lambda_star = rcps.fit(scores, grid)

    return ConformalCalibrationResult(
        scores=scores,
        alpha=alpha,
        q_hat=q_hat,
        lambda_star=lambda_star,
    )
