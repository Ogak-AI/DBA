"""
representation.py
=================
Converts biological sequences into numeric feature vectors.

Two methods are implemented:

1. **K-mer frequency vectors**  (fast, no ML dependencies)
   - Slide a window of length k across the sequence
   - Count occurrences of each k-mer
   - Normalise by sequence length to produce a probability vector

2. **Pseudo-embedding via random projection** (bonus method)
   - First build a k-mer vector, then project into a lower-dimensional space
   - Simulates the behaviour of learned embeddings without requiring a GPU or
     large pretrained models
   - In production, replace with ESM-2 (facebook/esm2_t6_8M_UR50D) via
     HuggingFace Transformers

Both return numpy arrays of shape (n_sequences, n_features).
"""

import itertools
from typing import List, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# K-mer vectorisation
# ---------------------------------------------------------------------------

def _build_kmer_index(alphabet: List[str], k: int) -> Dict[str, int]:
    """Return a sorted mapping {kmer: column_index}."""
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

    Parameters
    ----------
    sequences : list of sequence strings
    alphabet  : list of valid characters (e.g. ['A','T','C','G'])
    k         : k-mer length
    normalise : if True, divide counts by (len(seq) - k + 1)

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
# Pseudo-embedding via random projection (bonus method)
# ---------------------------------------------------------------------------

def random_projection_embed(
    X: np.ndarray,
    out_dim: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """
    Project k-mer vectors into a lower-dimensional space via a random
    Gaussian projection matrix (Johnson-Lindenstrauss style).

    Parameters
    ----------
    X       : input matrix (n_sequences, n_kmer_features)
    out_dim : target embedding dimension
    seed    : RNG seed for the projection matrix

    Returns
    -------
    np.ndarray of shape (n_sequences, out_dim)
    """
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((X.shape[1], out_dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-9  # column-normalise
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
) -> Dict[str, np.ndarray]:
    """
    Build both k-mer and (optionally) projected embeddings for D1 and D2.

    Returns
    -------
    dict with keys:
        'd1_kmer'   : k-mer matrix for D1
        'd2_kmer'   : k-mer matrix for D2
        'd1_embed'  : projected embedding for D1  (if embed_dim is not None)
        'd2_embed'  : projected embedding for D2
    """
    all_seqs = d1_seqs + d2_seqs
    # Build a shared projection matrix fitted on all sequences
    all_kmer = kmer_vectorise(all_seqs, alphabet, k=k, normalise=True)
    d1_kmer = all_kmer[: len(d1_seqs)]
    d2_kmer = all_kmer[len(d1_seqs):]

    result = {"d1_kmer": d1_kmer, "d2_kmer": d2_kmer}

    if embed_dim is not None:
        # Fit projection matrix on ALL data for consistency
        all_embed = random_projection_embed(all_kmer, out_dim=embed_dim)
        result["d1_embed"] = all_embed[: len(d1_seqs)]
        result["d2_embed"] = all_embed[len(d1_seqs):]

    return result
