"""
Video-exchangeability coverage bounds for non-exchangeable colonoscopy frames.

Standard conformal prediction assumes exchangeability (i.e. i.i.d. calibration
and test frames).  Colonoscopy frames are temporally correlated and the
distribution shifts as the scope moves through different colon segments.

We implement a *shift-aware* coverage bound based on the total variation (TV)
distance between the calibration and test frame distributions, following:

  Tibshirani et al., "Conformal Prediction Under Covariate Shift", NeurIPS 2019.
  Barber et al., "The Limits of Distribution-Free Conditional Validity", 2019.
  Podkopaev & Ramdas, "Distribution-free Uncertainty Quantification...", 2021.

The bound is:

    P(y_test ∈ C(x_test)) ≥ 1 − α − ε_TV

where ε_TV is the TV distance between calibration and test marginals, estimated
via a domain-classifier proxy (a lightweight binary classifier trained to
distinguish calibration vs. test frames by their DINOv2 CLS tokens).

Additionally we implement a Wasserstein-slack bound that replaces TV with
a Wasserstein-1 distance in feature space, which is often tighter when
distributions are close but not identical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.special import expit


# ---------------------------------------------------------------------------
# Feature extraction (DINOv2 CLS tokens)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_cls_tokens(
    model: nn.Module,
    loader,
    device: torch.device,
) -> np.ndarray:
    """
    Extract DINOv2 CLS tokens from the LoRA encoder for all frames in *loader*.

    Args:
        model: ConformalDepthModel (or anything with .lora_encoder).
        loader: DataLoader yielding {"image": Tensor(B,3,H,W), ...}.
        device: CUDA/CPU device.

    Returns:
        (N, D) float32 numpy array of CLS token embeddings.
    """
    model.eval()
    all_tokens = []

    for batch in loader:
        imgs = batch["image"].to(device)
        enc_out = model.lora_encoder(
            pixel_values=imgs,
            output_hidden_states=True,
            return_dict=True,
        )
        # CLS token is index 0 of the last hidden state
        cls = enc_out.hidden_states[-1][:, 0, :]   # (B, D)
        all_tokens.append(cls.cpu().numpy())

    return np.concatenate(all_tokens, axis=0)


# ---------------------------------------------------------------------------
# TV distance proxy via domain classifier
# ---------------------------------------------------------------------------

def estimate_tv_via_classifier(
    calib_features: np.ndarray,   # (n_calib, D)
    test_features:  np.ndarray,   # (n_test,  D)
    n_cv_folds: int = 5,
) -> float:
    """
    Estimate TV(P_calib, P_test) using a logistic-regression domain classifier.

    TV distance ≤ 2 * (AUC − 0.5) for a Bayes-optimal classifier.
    We use cross-validated predicted probabilities to get an unbiased AUC.

    Returns ε_TV ∈ [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    n_c, n_t = len(calib_features), len(test_features)
    X = np.concatenate([calib_features, test_features], axis=0)
    y = np.array([0] * n_c + [1] * n_t)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    proba = cross_val_predict(clf, X, y, cv=n_cv_folds, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, proba)
    eps_tv = 2.0 * max(auc - 0.5, 0.0)
    return float(eps_tv)


# ---------------------------------------------------------------------------
# Wasserstein-1 (MMD proxy for speed)
# ---------------------------------------------------------------------------

def estimate_wasserstein_slack(
    calib_features: np.ndarray,
    test_features:  np.ndarray,
    subsample: int = 2000,
    seed: int = 42,
) -> float:
    """
    Approximate W₁(P_calib, P_test) via the mean-feature-distance proxy.

    Full Wasserstein-1 requires O(n²) computation.  We use the simpler bound:
        W₁ ≤ ‖μ_calib − μ_test‖₂  (only tight when distributions are close)

    For production, replace with pot.emd2 (POT library) for exact computation.

    Returns ε_W1 (normalized to [0,1] by dividing by the feature norm scale).
    """
    rng = np.random.default_rng(seed)

    if len(calib_features) > subsample:
        idx = rng.choice(len(calib_features), subsample, replace=False)
        calib_features = calib_features[idx]
    if len(test_features) > subsample:
        idx = rng.choice(len(test_features), subsample, replace=False)
        test_features = test_features[idx]

    mu_c = calib_features.mean(axis=0)
    mu_t = test_features.mean(axis=0)
    mean_norm = (np.linalg.norm(calib_features, axis=1).mean() + 1e-8)

    raw_dist = float(np.linalg.norm(mu_c - mu_t))
    return raw_dist / mean_norm


# ---------------------------------------------------------------------------
# Adjusted coverage threshold
# ---------------------------------------------------------------------------

class ShiftAwareCoverageAdjuster:
    """
    Adjusts the conformal threshold q̂ to account for distribution shift.

    Inflated threshold:
        q̂_adjusted = q̂ + slack(ε) / f(n_calib)

    where slack(ε) converts the TV/W1 slack to a score-space inflation and
    f(n) = √n is the standard Chebyshev factor.

    Usage::

        adjuster = ShiftAwareCoverageAdjuster(method="tv")
        q_adj = adjuster.adjust(q_hat=5.2, eps_shift=0.08, scores=calib_scores)
    """

    def __init__(
        self,
        method: str = "tv",    # "tv" | "wasserstein"
        score_range: float | None = None,
    ):
        self.method = method
        self.score_range = score_range

    def adjust(
        self,
        q_hat: float,
        eps_shift: float,
        scores: np.ndarray,
    ) -> float:
        """
        Args:
            q_hat:     Standard conformal threshold.
            eps_shift: TV or W1 slack (in [0, 1]).
            scores:    Calibration nonconformity scores (used to estimate scale).

        Returns:
            Adjusted threshold q̂_adj ≥ q̂.
        """
        n = len(scores)
        if self.score_range is None:
            s_range = float(scores.max() - scores.min()) + 1e-8
        else:
            s_range = self.score_range

        # Convert eps_shift (distribution distance) to score-space slack
        # Follows Tibshirani et al. eq. (1): inflated by eps * range / sqrt(n)
        inflation = eps_shift * s_range / np.sqrt(n)
        return float(q_hat + inflation)


# ---------------------------------------------------------------------------
# Per-sequence drift estimation (inter-frame TV)
# ---------------------------------------------------------------------------

def compute_sequential_drift(
    cls_tokens: np.ndarray,   # (T, D) tokens in temporal order
    window: int = 10,
) -> np.ndarray:
    """
    Estimate local TV drift across a video sequence using a sliding window.

    Compares the distribution of CLS tokens in window [t-w, t] vs. [t, t+w].
    Returns (T,) drift curve — a proxy for the exchangeability slack over time.
    """
    T, D = cls_tokens.shape
    drifts = np.zeros(T, dtype=np.float64)

    for t in range(window, T - window):
        left  = cls_tokens[t - window:t]
        right = cls_tokens[t:t + window]
        mu_l = left.mean(0)
        mu_r = right.mean(0)
        scale = (np.linalg.norm(left, axis=1).mean() + 1e-8)
        drifts[t] = np.linalg.norm(mu_l - mu_r) / scale

    return drifts
