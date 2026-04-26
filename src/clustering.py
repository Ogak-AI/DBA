"""
clustering.py
=============
Cluster-aware dataset splitting for biosecurity evaluation.

Instead of a random sequence split (which can place near-identical sequences
in both D1 and D2), this module clusters sequences by k-mer composition and
assigns whole clusters exclusively to D1 or D2.

This mirrors how biosecurity screening categories work in practice:
a screening rule targets a functional/structural family, so all members of
that family are either screened out (D1) or left in the public corpus (D2).

Speed note
----------
k-mer matrices are high-dimensional (8,000 for k=3).  We project to 100
dimensions with TruncatedSVD before clustering — this preserves most of the
variance while reducing MiniBatchKMeans runtime from minutes to seconds.
"""
import logging
import random
from typing import List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def cluster_aware_split(
    sequences: List[str],
    kmer_vectors: np.ndarray,
    n_clusters: int = 150,
    d1_fraction: float = 0.33,
    seed: int = 42,
    svd_components: int = 100,
) -> Tuple[List[str], List[str], List[int], List[int], np.ndarray]:
    """
    Split sequences into D1 and D2 at cluster boundaries.

    Sequences within the same k-mer cluster (compositionally similar) are
    assigned exclusively to one split, never divided across D1 and D2.
    This prevents information leakage from near-duplicate sequences appearing
    in both sets — a methodological weakness of random splitting.

    Parameters
    ----------
    sequences      : full list of sequence strings (length N)
    kmer_vectors   : pre-computed k-mer frequency matrix (N, n_features)
    n_clusters     : target number of clusters (capped at N//5)
    d1_fraction    : target fraction assigned to D1
    seed           : reproducibility seed
    svd_components : reduce to this many dims before clustering (speed)

    Returns
    -------
    (d1_seqs, d2_seqs, d1_indices, d2_indices, cluster_labels)
    """
    n = len(sequences)
    k = min(n_clusters, max(2, n // 5))
    logger.info("Cluster-aware split: %d clusters on %d sequences ...", k, n)

    # Dimensionality reduction before clustering (avoids 8000-dim bottleneck)
    n_components = min(svd_components, kmer_vectors.shape[1] - 1, n - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X_reduced = svd.fit_transform(kmer_vectors.astype(np.float32))
    X_reduced = normalize(X_reduced, norm="l2")

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=1,
        max_iter=50,
        batch_size=min(1024, n),
    )
    labels = km.fit_predict(X_reduced)

    clusters: dict = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(i)

    rng = random.Random(seed)
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)

    n_d1_target = int(n * d1_fraction)
    d1_idx: List[int] = []
    d2_idx: List[int] = []

    for cid in cluster_ids:
        if len(d1_idx) < n_d1_target:
            d1_idx.extend(clusters[cid])
        else:
            d2_idx.extend(clusters[cid])

    logger.info(
        "Cluster split done: D1=%d seqs, D2=%d seqs",
        len(d1_idx), len(d2_idx),
    )
    d1_seqs = [sequences[i] for i in d1_idx]
    d2_seqs = [sequences[i] for i in d2_idx]
    return d1_seqs, d2_seqs, d1_idx, d2_idx, labels
