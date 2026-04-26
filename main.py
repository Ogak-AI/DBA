"""
main.py
=======
Orchestrator for the Dataset Bottleneck Analysis (DBA) pipeline.

Usage
-----
    python main.py [options]

Key flags
---------
  --n-total INT          Sequences to download (default 5000)
  --split-mode           random | cluster (default cluster)
  --no-esm2              Skip ESM-2 encoding
  --esm2-subset INT      Max seqs per split for ESM-2 (default 150)
  --n-bootstrap INT      Bootstrap resamples for CIs (default 200)
  --toxin                Also run the toxin-protein experiment
  --source               uniprot | ncbi | auto
  --d1-fraction FLOAT    Fraction to D1 (default 0.33)
  --k INT                K-mer length (default 3)
  --embed-dim INT        Random-projection dim (default 64)
  --threshold FLOAT      Coverage threshold tau (default 0.90)
  --k-recon INT          Neighbours for reconstruction proxy (default 5)
  --min-len INT          Minimum sequence length (default 50)
  --max-len INT          Maximum sequence length (default 2000)
  --seed INT             Random seed (default 42)
  --force-dl             Re-download even if cache exists
"""

import argparse
import logging
import random
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np

from src.data_loader import (
    fetch_sequences, fetch_toxin_sequences,
    split_datasets, detect_alphabet,
)
from src.representation import build_representations, kmer_vectorise
from src.redundancy_analysis import (
    ablation_information_retained,
    bootstrap_redundancy_ci,
    null_model_redundancy,
    wilcoxon_test,
)
from src.visualisation import (
    save_similarity_histogram,
    save_coverage_curve,
    save_ablation_plot,
    save_size_sensitivity_plot,
    save_validation_plot,
    save_representation_comparison,
    save_toxin_comparison,
    build_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TIMINGS: Dict[str, float] = {}


def _tick(label: str) -> float:
    t = time.time()
    TIMINGS[label] = t
    return t


def _tock(label: str) -> float:
    elapsed = time.time() - TIMINGS.pop(label)
    logger.info("  [timer] %s: %.1fs", label, elapsed)
    return elapsed


# =============================================================================
# Stage 1 — Download & split
# =============================================================================

def stage_load(args: argparse.Namespace) -> Dict[str, Any]:
    print("\n" + "=" * 62)
    print("STAGE 1 - Real Data Download & Split")
    print("=" * 62)

    _tick("download")
    records = fetch_sequences(
        n=args.n_total,
        source=args.source,
        min_len=args.min_len,
        max_len=args.max_len,
        force_download=args.force_dl,
    )
    t_dl = _tock("download")

    sequences = [seq for _, seq in records]
    seq_type, alphabet = detect_alphabet(sequences)

    d1_kmer_pre: Optional[np.ndarray] = None
    d2_kmer_pre: Optional[np.ndarray] = None

    if args.split_mode == "cluster":
        from src.clustering import cluster_aware_split
        _tick("cluster_split")
        all_kmer = kmer_vectorise(sequences, alphabet, k=args.k)
        d1_seqs, d2_seqs, d1_idx, d2_idx, _ = cluster_aware_split(
            sequences, all_kmer,
            n_clusters=150,
            d1_fraction=args.d1_fraction,
            seed=args.seed,
        )
        d1_kmer_pre = all_kmer[d1_idx]
        d2_kmer_pre = all_kmer[d2_idx]
        _tock("cluster_split")
        split_label = "cluster-aware"
    else:
        d1_seqs, d2_seqs = split_datasets(
            records, d1_fraction=args.d1_fraction, seed=args.seed
        )
        split_label = "random"

    seq_lengths = [len(s) for s in d1_seqs + d2_seqs]
    print(f"  Source          : {args.source}")
    print(f"  Split mode      : {split_label}")
    print(f"  Total sequences : {len(records):,}")
    print(f"  D1 (restricted) : {len(d1_seqs):,} sequences")
    print(f"  D2 (reference)  : {len(d2_seqs):,} sequences")
    print(f"  Detected type   : {seq_type.upper()}")
    print(f"  Length range    : {min(seq_lengths)}-{max(seq_lengths)} chars")
    print(f"  Mean length     : {sum(seq_lengths)/len(seq_lengths):.0f} chars")
    print(f"  Download time   : {t_dl:.1f}s")

    return {
        "d1": d1_seqs,
        "d2": d2_seqs,
        "seq_type": seq_type,
        "alphabet": alphabet,
        "d1_kmer_pre": d1_kmer_pre,
        "d2_kmer_pre": d2_kmer_pre,
        "t_download": t_dl,
    }


# =============================================================================
# Validation — metric sanity check
# =============================================================================

def stage_validate(data: Dict[str, Any], args: argparse.Namespace) -> Tuple[float, float]:
    print("\n" + "=" * 62)
    print("VALIDATION - Metric Sanity Check")
    print("=" * 62)

    d1, d2, alphabet = data["d1"], data["d2"], data["alphabet"]
    all_seqs = d1 + d2
    rng_val = random.Random(args.seed + 99)

    n_check = min(50, len(all_seqs) // 4)
    val_d1 = rng_val.sample(all_seqs, n_check)
    val_d2 = all_seqs[:]

    r_high = build_representations(
        val_d1, val_d2, alphabet, k=args.k, embed_dim=None,
        run_esm2=False,
    )
    res_high = ablation_information_retained(
        r_high["d1_kmer"], r_high["d2_kmer"],
        threshold=args.threshold, k_recon=args.k_recon, seed=args.seed,
    )

    rng_low = random.Random(args.seed + 100)
    shuffled = all_seqs[:]
    rng_low.shuffle(shuffled)
    low_d1 = shuffled[:n_check]
    low_d2 = shuffled[n_check:]
    r_low = build_representations(
        low_d1, low_d2, alphabet, k=args.k, embed_dim=None,
        run_esm2=False,
    )
    res_low = ablation_information_retained(
        r_low["d1_kmer"], r_low["d2_kmer"],
        threshold=args.threshold, k_recon=args.k_recon, seed=args.seed,
    )

    r_high_val = res_high["redundancy_score"]
    r_low_val  = res_low["redundancy_score"]
    passed = r_high_val > r_low_val
    print(f"  HIGH (D1 subset of D2) : R = {r_high_val:.4f}  (expect ~1.0)")
    print(f"  LOW  (D1 disjoint D2)  : R = {r_low_val:.4f}  (expect ~0.0)")
    print(f"  Sanity check {'PASSED' if passed else 'FAILED'}: HIGH > LOW = {passed}")

    p = save_validation_plot(r_high_val, r_low_val)
    print(f"  Saved: {p}")
    return r_high_val, r_low_val


# =============================================================================
# Stage 2 — Representations
# =============================================================================

def stage_represent(data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, np.ndarray]:
    print("\n" + "=" * 62)
    print("STAGE 2 - Building Representations")
    print("=" * 62)

    _tick("represent")
    reps = build_representations(
        data["d1"], data["d2"], data["alphabet"],
        k=args.k,
        embed_dim=args.embed_dim,
        run_esm2=not args.no_esm2,
        esm2_subset=args.esm2_subset,
        precomputed_d1_kmer=data["d1_kmer_pre"],
        precomputed_d2_kmer=data["d2_kmer_pre"],
    )
    t_rep = _tock("represent")

    print(f"  K-mer (k={args.k}) feature dim : {reps['d1_kmer'].shape[1]:,}")
    if "d1_embed" in reps:
        print(f"  Projection embedding dim : {reps['d1_embed'].shape[1]}")
    if "d1_esm" in reps:
        print(f"  ESM-2 embedding dim      : {reps['d1_esm'].shape[1]}  "
              f"({reps['d1_esm'].shape[0]} D1 + {reps['d2_esm'].shape[0]} D2 seqs)")
    print(f"  Representation time      : {t_rep:.1f}s")
    # Persist D2 ESM-2 embeddings so stage_toxin can reuse them without re-encoding
    if "d2_esm" in reps:
        np.save("results/d2_esm.npy", reps["d2_esm"])
    return reps


# =============================================================================
# Stage 3+4 — Analysis (metrics, CIs, significance, null model, held-out)
# =============================================================================

def stage_analyse(
    reps: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print("\n" + "=" * 62)
    print("STAGE 3+4 - Redundancy Analysis")
    print("=" * 62)

    results: Dict[str, Any] = {}
    all_nn_scores: Dict[str, np.ndarray] = {}

    method_pairs = [("K-mer", "d1_kmer", "d2_kmer")]
    if "d1_embed" in reps:
        method_pairs.append(("Embedding", "d1_embed", "d2_embed"))
    if "d1_esm" in reps:
        method_pairs.append(("ESM-2", "d1_esm", "d2_esm"))

    for method, k1, k2 in method_pairs:
        print(f"\n  [{method}]")
        _tick(f"analyse_{method}")

        res = ablation_information_retained(
            reps[k1], reps[k2],
            threshold=args.threshold,
            k_recon=args.k_recon,
            seed=args.seed,
        )

        print(f"    Coverage @ t={args.threshold:.2f}  : {res['nn_result']['coverage_pct']:.2f}%")
        print(f"    Mean NN similarity    : {res['nn_result']['mean_similarity']:.4f} "
              f"+/- {res['nn_result']['std_similarity']:.4f}")
        print(f"    Mean recon MSE        : {res['recon_result']['mean_mse']:.2e}  "
              f"(baseline {res['recon_result']['baseline_mse']:.2e})")
        print(f"    Norm. MSE             : {res['norm_mse']:.4f}")
        print(f"    Redundancy Score      : {res['redundancy_score']:.4f}")

        # Bootstrap CI
        print(f"    Bootstrap CI (n={args.n_bootstrap}) ...")
        ci = bootstrap_redundancy_ci(
            reps[k1], reps[k2],
            n_bootstrap=args.n_bootstrap,
            threshold=args.threshold,
            k_recon=args.k_recon,
            seed=args.seed,
        )
        print(f"    R = {ci['mean']:.4f}  [95% CI: {ci['ci_low']:.4f} – {ci['ci_high']:.4f}]")

        # Null model
        null = null_model_redundancy(
            reps[k1], reps[k2],
            threshold=args.threshold,
            k_recon=args.k_recon,
            seed=args.seed,
        )
        delta = ci["mean"] - null["redundancy_score"]
        print(f"    Null model R          : {null['redundancy_score']:.4f}  "
              f"(real - null = +{delta:.4f})")

        t_m = _tock(f"analyse_{method}")
        res["ci"] = ci
        res["null"] = null
        res["t_analysis"] = t_m
        results[method] = res
        all_nn_scores[method] = res["nn_result"]["nn_scores"]

    # Held-out validation (10 % of D1)
    print("\n  [Held-out stability check]")
    k1_main, k2_main = "d1_kmer", "d2_kmer"
    d1_X = reps[k1_main]
    rng_ho = np.random.default_rng(args.seed + 7)
    n_ho = max(5, int(len(d1_X) * 0.10))
    ho_idx = rng_ho.choice(len(d1_X), size=n_ho, replace=False)
    res_ho = ablation_information_retained(
        d1_X[ho_idx], reps[k2_main],
        threshold=args.threshold,
        k_recon=args.k_recon,
        seed=args.seed,
    )
    full_R = results["K-mer"]["redundancy_score"]
    ho_R   = res_ho["redundancy_score"]
    print(f"    Full D1 R             : {full_R:.4f}")
    print(f"    Held-out 10% D1 R     : {ho_R:.4f}  (delta = {abs(full_R - ho_R):.4f})")
    results["_held_out"] = {"full_R": full_R, "ho_R": ho_R}

    # Wilcoxon test between K-mer and best embedding method
    if "Embedding" in all_nn_scores or "ESM-2" in all_nn_scores:
        b_label = "ESM-2" if "ESM-2" in all_nn_scores else "Embedding"
        # Align lengths (ESM-2 may be on a subset)
        a = all_nn_scores["K-mer"]
        b = all_nn_scores[b_label]
        n_pairs = min(len(a), len(b))
        wtest = wilcoxon_test(a[:n_pairs], b[:n_pairs])
        if wtest["available"]:
            print(f"\n  [Wilcoxon test: K-mer vs {b_label}]")
            print(f"    n = {wtest['n_pairs']}, stat = {wtest['statistic']:.1f}, "
                  f"p = {wtest['pvalue']:.2e}")
            results["_wilcoxon"] = wtest
            results["_wilcoxon"]["methods"] = f"K-mer vs {b_label}"

    return results


# =============================================================================
# Stage — Toxin experiment
# =============================================================================

def stage_toxin(data: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    print("\n" + "=" * 62)
    print("TOXIN EXPERIMENT - Biosecurity-Relevant Sequences")
    print("=" * 62)

    try:
        toxin_records = fetch_toxin_sequences(
            n=500,
            min_len=args.min_len,
            max_len=args.max_len,
            force_download=args.force_dl,
        )
    except Exception as exc:
        print(f"  Toxin fetch failed: {exc}")
        return None

    toxin_seqs = [seq for _, seq in toxin_records]
    d2_seqs    = data["d2"]
    alphabet   = data["alphabet"]

    n_use = min(len(toxin_seqs), len(d2_seqs) // 2)
    toxin_seqs = toxin_seqs[:n_use]
    print(f"  Toxin D1 sequences  : {len(toxin_seqs):,}")
    print(f"  Reference D2 (Swiss-Prot) : {len(d2_seqs):,}")

    reps_t = build_representations(
        toxin_seqs, d2_seqs, alphabet,
        k=args.k, embed_dim=args.embed_dim,
        run_esm2=False,
    )
    res_t = ablation_information_retained(
        reps_t["d1_kmer"], reps_t["d2_kmer"],
        threshold=args.threshold,
        k_recon=args.k_recon,
        seed=args.seed,
    )
    ci_t = bootstrap_redundancy_ci(
        reps_t["d1_kmer"], reps_t["d2_kmer"],
        n_bootstrap=min(args.n_bootstrap, 100),
        threshold=args.threshold,
        seed=args.seed,
    )
    res_t["ci"] = ci_t
    print(f"  Toxin K-mer R = {ci_t['mean']:.4f}  [95% CI: {ci_t['ci_low']:.4f} – {ci_t['ci_high']:.4f}]")

    # ESM-2 toxin: encode only toxin D1; reuse precomputed D2 embeddings if available
    import os
    if not args.no_esm2:
        from src.esm_encoder import esm2_available, esm2_embed
        d2_esm_path = "results/d2_esm.npy"
        if esm2_available() and os.path.exists(d2_esm_path):
            print("  Encoding toxin sequences with ESM-2 (reusing precomputed D2 embeddings) ...")
            d2_esm = np.load(d2_esm_path)
            d1_esm_t = esm2_embed(toxin_seqs)
            if d1_esm_t is not None:
                # Align: D2 ESM-2 subset size = min(len(d2_esm), len(d1_esm_t)*10)
                n_d2 = min(len(d2_esm), len(d1_esm_t) * 10)
                d2_esm_sub = d2_esm[:n_d2]
                res_esm_t = ablation_information_retained(
                    d1_esm_t, d2_esm_sub,
                    threshold=args.threshold,
                    k_recon=args.k_recon,
                    seed=args.seed,
                )
                ci_esm_t = bootstrap_redundancy_ci(
                    d1_esm_t, d2_esm_sub,
                    n_bootstrap=min(args.n_bootstrap, 50),
                    threshold=args.threshold,
                    seed=args.seed,
                )
                print(f"  Toxin ESM-2 R = {ci_esm_t['mean']:.4f}  "
                      f"[95% CI: {ci_esm_t['ci_low']:.4f} – {ci_esm_t['ci_high']:.4f}]")
                print(f"  Toxin ESM-2 Coverage@{args.threshold:.2f}: "
                      f"{res_esm_t['nn_result']['coverage_pct']:.2f}%")
                res_t["esm_ci"] = ci_esm_t
                res_t["esm_res"] = res_esm_t

    rand_ci = args._random_ci_cache
    p = save_toxin_comparison(
        random_mean    = rand_ci["mean"],
        random_ci_low  = rand_ci["ci_low"],
        random_ci_high = rand_ci["ci_high"],
        toxin_mean     = ci_t["mean"],
        toxin_ci_low   = ci_t["ci_low"],
        toxin_ci_high  = ci_t["ci_high"],
        method_label   = "K-mer",
    )
    print(f"  Saved: {p}")
    return res_t


# =============================================================================
# Stage 5 — Outputs
# =============================================================================

def stage_output(
    data: Dict[str, Any],
    results: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    print("\n" + "=" * 62)
    print("STAGE 5 - Outputs")
    print("=" * 62)

    saved = []
    core_methods = {k: v for k, v in results.items() if not k.startswith("_")}

    # 1. Similarity histograms
    for method, label in [
        ("K-mer", f"K-mer (k={args.k})"),
        ("Embedding", "Projected Embedding"),
        ("ESM-2", "ESM-2"),
    ]:
        if method not in core_methods:
            continue
        p = save_similarity_histogram(
            core_methods[method]["nn_result"]["nn_scores"],
            method_label=label,
            threshold=args.threshold,
            filename=f"similarity_histogram_{method.lower().replace('-','_')}",
        )
        saved.append(p)
        print(f"  Saved: {p}")

    # 2. Coverage curve
    t_k = core_methods["K-mer"]["thresholds"]
    c_k = core_methods["K-mer"]["coverage_fractions"]
    t_e = c_e = None
    best_embed = "ESM-2" if "ESM-2" in core_methods else ("Embedding" if "Embedding" in core_methods else None)
    if best_embed:
        t_e = core_methods[best_embed]["thresholds"]
        c_e = core_methods[best_embed]["coverage_fractions"]
    p = save_coverage_curve(t_k, c_k, t_e, c_e)
    saved.append(p)
    print(f"  Saved: {p}")

    # 3. Ablation bar chart
    p = save_ablation_plot(core_methods)
    saved.append(p)
    print(f"  Saved: {p}")

    # 4. Representation comparison (with CIs)
    p = save_representation_comparison(core_methods)
    saved.append(p)
    print(f"  Saved: {p}")

    # 5. Size-sensitivity
    print("\n  [Size sensitivity] ...")
    all_seqs = data["d1"] + data["d2"]
    rng = random.Random(args.seed)
    alphabet = data["alphabet"]
    max_d1 = len(data["d1"])
    step = max(1, max_d1 // 9)
    d1_targets = sorted(set([10, 20] + list(range(step, max_d1 + 1, step)) + [max_d1]))
    d1_targets = [n for n in d1_targets if n <= max_d1]

    size_results = []
    for n_d1_target in d1_targets:
        n_d2_target = max(10, int(n_d1_target * (1 - args.d1_fraction) / args.d1_fraction))
        n_needed = n_d1_target + n_d2_target
        if n_needed > len(all_seqs):
            continue
        subset = rng.sample(all_seqs, n_needed)
        d1s, d2s = subset[:n_d1_target], subset[n_d1_target:]
        r = build_representations(d1s, d2s, alphabet, k=args.k, embed_dim=args.embed_dim,
                                  run_esm2=False)
        ak = ablation_information_retained(r["d1_kmer"], r["d2_kmer"],
                                           threshold=args.threshold, seed=args.seed)
        ae = ablation_information_retained(r["d1_embed"], r["d2_embed"],
                                           threshold=args.threshold, seed=args.seed) \
             if "d1_embed" in r else ak
        size_results.append((n_d1_target, ak["redundancy_score"], ae["redundancy_score"]))
        print(f"     D1={n_d1_target:4d}  kmer={ak['redundancy_score']:.3f}  "
              f"embed={ae['redundancy_score']:.3f}")

    if size_results:
        p = save_size_sensitivity_plot(size_results)
        saved.append(p)
        print(f"  Saved: {p}")

    # 6. Summary CSV
    dataset_sizes = {"n_d1": len(data["d1"]), "n_d2": len(data["d2"])}
    df = build_summary_table(core_methods, dataset_sizes)
    print("\n  -- Summary Table --")
    print(df.to_string(index=False))
    print(f"\n  Saved: results/summary_table.csv")

    # 7. Timing summary
    print("\n  -- Timing Benchmarks --")
    print(f"    Download          : {data['t_download']:.1f}s")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dataset Bottleneck Analysis — Biosecurity Screening Audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--source", choices=["uniprot", "ncbi", "auto"], default="auto")
    p.add_argument("--n-total", type=int, default=5000,
                   help="Sequences to download (before filtering)")
    p.add_argument("--d1-fraction", type=float, default=0.33)
    p.add_argument("--split-mode", choices=["random", "cluster"], default="cluster",
                   help="Split strategy: cluster-aware (default) or random")
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=2000)
    p.add_argument("--force-dl", action="store_true")
    # Representation
    p.add_argument("-k", type=int, default=3, help="K-mer length")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--no-esm2", action="store_true", help="Skip ESM-2 encoding")
    p.add_argument("--esm2-subset", type=int, default=150,
                   help="Max sequences per split sent to ESM-2")
    # Analysis
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--k-recon", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    # Experiments
    p.add_argument("--toxin", action="store_true",
                   help="Run the biosecurity-relevant toxin-protein experiment")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(
        "\n  " + "=" * 60 + "\n"
        "  Dataset Bottleneck Analysis  —  Biosecurity Prototype\n"
        "  Real sequences: UniProt Swiss-Prot / NCBI\n"
        "  " + "=" * 60
    )

    t_start = time.time()

    data    = stage_load(args)
    stage_validate(data, args)
    reps    = stage_represent(data, args)
    results = stage_analyse(reps, args)

    # Cache the random-protein CI for toxin comparison baseline
    args._random_ci_cache = results["K-mer"]["ci"]

    if args.toxin:
        stage_toxin(data, args)

    stage_output(data, results, args)

    total = time.time() - t_start
    print(f"\nDone. Total pipeline time: {total:.1f}s")
    print("  All outputs in ./results/\n")


if __name__ == "__main__":
    main()
