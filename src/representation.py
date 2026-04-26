"""
representation.py
=================
Converts biological sequences into numeric feature vectors.

Three methods, in increasing representational power:

1. K-mer frequency vectors       (fast, interpretable, no ML)
2. Random-projection embeddings  (lightweight geometric proxy)
3. ESM-2 embeddings              (protein language model, 320-dim)
   Model: facebook/esm2_t6_8M_UR50D — ~30 MB, CPU-friendly.
   Falls back gracefully when transformers/torch are absent.
"""

import itertools
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# K-mer vectorisation
# ---------------------------------------------------------------------------

def _build_kmer_index(alphabet: List[str], k: int) -> Dict[str, int]:
    kmers = ["".join(p) for p in itertools.product(alphabet, repeat=k)]
    return {km: i for i, km in enumerate(sorted(kmers))}


def kmer_vectorise(
    sequences: List[str],
    alphabet: List[str],
    k: int = 4,
    normalise: bool = True,
) -> np.ndarray:
    """
    Convert sequences to k-mer frequency vectors.

    Returns
    -------
    np.ndarray of shape (len(sequences), |alphabet|^k)
    """
    index = _build_kmer_index(alphabet, k)
    n_features = len(index)
    X = np.zeros((len(sequences), n_features), dtype=np.float32)

    for row, seq in enumerate(sequences):
        n_kmers = len(seq) - k + 1
        if n_kmers <= 0:
            continue
        for i in range(n_kmers):
            kmer = seq[i: i + k]
            if kmer in index:
                X[row, index[kmer]] += 1
        if normalise and n_kmers > 0:
            X[row] /= n_kmers

    return X


# ---------------------------------------------------------------------------
# Random-projection embedding (Johnson-Lindenstrauss proxy)
# ---------------------------------------------------------------------------

def random_projection_embed(
    X: np.ndarray,
    out_dim: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """
    Project k-mer vectors into a lower-dimensional space via a random
    Gaussian projection matrix.

    Returns
    -------
    np.ndarray of shape (n_sequences, out_dim)
    """
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((X.shape[1], out_dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-9
    return X @ P


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_representations(
    d1_seqs: List[str],
    d2_seqs: List[str],
    alphabet: List[str],
    k: int = 4,
    embed_dim: Optional[int] = 64,
    run_esm2: bool = True,
    esm2_subset: int = 150,
    precomputed_d1_kmer: Optional[np.ndarray] = None,
    precomputed_d2_kmer: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build k-mer, random-projection, and (optionally) ESM-2 representations
    for D1 and D2.

    Parameters
    ----------
    d1_seqs / d2_seqs       : sequence lists
    alphabet                 : valid characters (e.g. 20 amino acids)
    k                        : k-mer length
    embed_dim                : random-projection output dimension
    run_esm2                 : whether to attempt ESM-2 encoding
    esm2_subset              : max sequences per split sent to ESM-2
    precomputed_d1_kmer /
    precomputed_d2_kmer      : pass pre-sliced matrices from cluster split
                               to avoid re-computing k-mer vectors

    Returns
    -------
    dict with keys:
        'd1_kmer', 'd2_kmer'         — always present
        'd1_embed', 'd2_embed'       — present when embed_dim is not None
        'd1_esm', 'd2_esm'           — present when ESM-2 succeeds
        'd1_kmer_esm', 'd2_kmer_esm' — kmer subset matching ESM-2 rows
    """
    # ── K-mer ────────────────────────────────────────────────────────────────
    if precomputed_d1_kmer is not None and precomputed_d2_kmer is not None:
        d1_kmer = precomputed_d1_kmer
        d2_kmer = precomputed_d2_kmer
    else:
        all_seqs = d1_seqs + d2_seqs
        all_kmer = kmer_vectorise(all_seqs, alphabet, k=k, normalise=True)
        d1_kmer = all_kmer[: len(d1_seqs)]
        d2_kmer = all_kmer[len(d1_seqs):]

    result: Dict[str, np.ndarray] = {"d1_kmer": d1_kmer, "d2_kmer": d2_kmer}

    # ── Random-projection embedding ───────────────────────────────────────────
    if embed_dim is not None:
        all_kmer_cat = np.vstack([d1_kmer, d2_kmer])
        all_embed = random_projection_embed(all_kmer_cat, out_dim=embed_dim)
        result["d1_embed"] = all_embed[: len(d1_seqs)]
        result["d2_embed"] = all_embed[len(d1_seqs):]

    # ── ESM-2 ────────────────────────────────────────────────────────────────
    if run_esm2:
        from src.esm_encoder import esm2_available, esm2_embed
        if esm2_available():
            n1 = min(len(d1_seqs), esm2_subset)
            n2 = min(len(d2_seqs), esm2_subset)
            d1_esm = esm2_embed(d1_seqs[:n1])
            d2_esm = esm2_embed(d2_seqs[:n2])
            if d1_esm is not None and d2_esm is not None:
                result["d1_esm"] = d1_esm
                result["d2_esm"] = d2_esm
                result["d1_kmer_esm"] = d1_kmer[:n1]
                result["d2_kmer_esm"] = d2_kmer[:n2]

    return result
