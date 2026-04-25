"""
redundancy_analysis.py
======================
Core analysis functions implementing the three redundancy metrics:

A. Nearest-Neighbour Overlap
   – For every D1 item, find its closest match in D2.
   – Report similarity distribution and % above threshold.

B. Coverage Metric
   – Sweep a similarity threshold τ ∈ [0, 1].
   – Fraction of D1 "covered" = fraction with NN similarity ≥ τ.

C. Reconstruction Proxy
   – Approximate each D1 vector as the weighted sum of its k nearest
     neighbours in D2.
   – Report mean squared reconstruction error (MSE).

All functions accept pre-computed numpy matrices so they are representation-
agnostic (k-mer vectors or embeddings).
"""

from typing import Tuple, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_nn_similarity(
    X1: np.ndarray,
    X2: np.ndarray,
) -> np.ndarray:
    """
    For each row in X1, compute cosine similarity to its nearest neighbour
    in X2.

    Returns
    -------
    np.ndarray of shape (len(X1),) — one score per D1 item
    """
    # Normalise rows
    def _l2norm(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(norms == 0, 1.0, norms)

    X1n = _l2norm(X1.astype(np.float32))
    X2n = _l2norm(X2.astype(np.float32))

    # Batch cosine similarity to keep memory manageable
    batch = 256
    best_scores = np.zeros(len(X1), dtype=np.float32)
    for start in range(0, len(X1), batch):
        end = min(start + batch, len(X1))
        sims = X1n[start:end] @ X2n.T            # (batch, n2)
        best_scores[start:end] = sims.max(axis=1)
    return best_scores


# ---------------------------------------------------------------------------
# A. Nearest-Neighbour Overlap
# ---------------------------------------------------------------------------

def nearest_neighbour_overlap(
    X1: np.ndarray,
    X2: np.ndarray,
    threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Compute nearest-neighbour similarity between D1 and D2.

    Parameters
    ----------
    X1        : feature matrix for D1  (n1, d)
    X2        : feature matrix for D2  (n2, d)
    threshold : similarity threshold for "close match"

    Returns
    -------
    dict with:
        'nn_scores'        : array (n1,) of best cosine similarities
        'mean_similarity'  : float
        'std_similarity'   : float
        'coverage_pct'     : % of D1 items with sim >= threshold
        'threshold'        : the threshold used
    """
    nn_scores = _cosine_nn_similarity(X1, X2)
    covered = (nn_scores >= threshold).mean() * 100.0

    return {
        "nn_scores": nn_scores,
        "mean_similarity": float(nn_scores.mean()),
        "std_similarity": float(nn_scores.std()),
        "coverage_pct": float(covered),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# B. Coverage Metric (threshold sweep)
# ---------------------------------------------------------------------------

def coverage_vs_threshold(
    X1: np.ndarray,
    X2: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep similarity thresholds and compute coverage fraction at each.

    Parameters
    ----------
    X1         : feature matrix for D1
    X2         : feature matrix for D2
    thresholds : array of τ values; defaults to np.linspace(0, 1, 101)

    Returns
    -------
    (thresholds, coverage_fractions) — both 1-D arrays
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    nn_scores = _cosine_nn_similarity(X1, X2)
    coverage = np.array(
        [(nn_scores >= t).mean() for t in thresholds], dtype=np.float32
    )
    return thresholds, coverage


# ---------------------------------------------------------------------------
# C. Reconstruction Proxy
# ---------------------------------------------------------------------------

def reconstruction_error(
    X1: np.ndarray,
    X2: np.ndarray,
    k: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Approximate each D1 vector from its k-nearest neighbours in D2 using
    distance-weighted averaging.  Report mean squared error (MSE).

    Baseline: reconstruct each D1 vector from k *randomly* chosen D2 vectors
    (uniform weights, no nearest-neighbour search).  This represents a naive
    adversary who picks arbitrary public sequences as a reference.

    norm_mse = mean_mse / baseline_mse
        ~0  -> D1 perfectly recoverable from D2 (high redundancy)
        ~1  -> NN adds no advantage over random lookup (low redundancy)
    """
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X2)), metric="cosine")
    nbrs.fit(X2)
    distances, indices = nbrs.kneighbors(X1)  # (n1, k)

    similarities = 1.0 - distances
    similarities = np.clip(similarities, 1e-9, None)
    weights = similarities / similarities.sum(axis=1, keepdims=True)
    X1_hat = np.einsum("ij,ijk->ik", weights, X2[indices])
    mse_per_item = ((X1 - X1_hat) ** 2).mean(axis=1)

    # Random baseline: k uniformly-chosen D2 vectors, equal weights
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(X2), size=(len(X1), k))
    rand_w = np.full((len(X1), k), 1.0 / k, dtype=np.float32)
    X1_rand = np.einsum("ij,ijk->ik", rand_w, X2[rand_idx])
    baseline_per_item = ((X1 - X1_rand) ** 2).mean(axis=1)

    return {
        "mse_per_item": mse_per_item,
        "mean_mse": float(mse_per_item.mean()),
        "std_mse": float(mse_per_item.std()),
        "baseline_mse": float(baseline_per_item.mean()),
    }


# ---------------------------------------------------------------------------
# Ablation experiment helper
# ---------------------------------------------------------------------------

def ablation_information_retained(
    d1_kmer: np.ndarray,
    d2_kmer: np.ndarray,
    threshold: float = 0.90,
    k_recon: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Summarise information retained in D2 when D1 is removed.

    Returns a structured dict of all key metrics.
    """
    nn_result = nearest_neighbour_overlap(d1_kmer, d2_kmer, threshold=threshold)
    thresholds, coverage = coverage_vs_threshold(d1_kmer, d2_kmer)
    recon_result = reconstruction_error(d1_kmer, d2_kmer, k=k_recon, seed=seed)

    # Estimated Redundancy Score
    # Definition:
    #   R = 0.5 * (coverage_pct / 100)
    #     + 0.5 * (1 - normalised_mean_mse)
    # where normalised_mean_mse = mean_mse / baseline_mse
    # R ∈ [0, 1]; higher → more redundant (D1 well represented by D2)
    norm_mse = recon_result["mean_mse"] / max(recon_result["baseline_mse"], 1e-9)
    norm_mse = min(norm_mse, 1.0)  # cap at 1 for degenerate cases
    redundancy_score = 0.5 * (nn_result["coverage_pct"] / 100.0) + 0.5 * (
        1.0 - norm_mse
    )

    return {
        "nn_result": nn_result,
        "thresholds": thresholds,
        "coverage_fractions": coverage,
        "recon_result": recon_result,
        "redundancy_score": float(redundancy_score),
        "norm_mse": float(norm_mse),
    }
