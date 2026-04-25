"""
main.py
=======
Orchestrator for the Dataset Bottleneck Analysis pipeline.

Run with:
    python main.py [options]

Full options list:
    --source      uniprot|ncbi|auto       Data source          [auto]
    --n-total     INT                     Total seqs to fetch  [1500]
    --d1-fraction FLOAT                   Fraction -> D1       [0.33]
    --k           INT                     K-mer length         [4]
    --embed-dim   INT                     Projection dim       [64]
    --threshold   FLOAT                   Coverage threshold   [0.90]
    --k-recon     INT                     Recon neighbours     [5]
    --min-len     INT                     Min sequence length  [50]
    --max-len     INT                     Max sequence length  [2000]
    --force-dl                            Re-download data
    --seed        INT                     Random seed          [42]
"""

import argparse
import logging
import random
import time
from pathlib import Path

# ── project imports ──────────────────────────────────────────────────────────
from src.data_loader import fetch_sequences, split_datasets, detect_alphabet
from src.representation import build_representations
from src.redundancy_analysis import ablation_information_retained
from src.visualisation import (
    save_similarity_histogram,
    save_coverage_curve,
    save_ablation_plot,
    save_size_sensitivity_plot,
    save_validation_plot,
    build_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_load(args: argparse.Namespace):
    """Stage 1 - Download real sequences and split into D1 / D2."""
    print("\n" + "=" * 62)
    print("STAGE 1 - Real Data Download & Split")
    print("=" * 62)

    records = fetch_sequences(
        n=args.n_total,
        source=args.source,
        min_len=args.min_len,
        max_len=args.max_len,
        force_download=args.force_dl,
    )

    d1, d2 = split_datasets(records, d1_fraction=args.d1_fraction, seed=args.seed)

    seq_type, alphabet = detect_alphabet(d1 + d2)

    print(f"  Source          : {args.source}")
    print(f"  Total sequences : {len(records):,}")
    print(f"  D1 (restricted ): {len(d1):,} sequences")
    print(f"  D2 (reference)  : {len(d2):,} sequences")
    print(f"  Detected type   : {seq_type.upper()}")
    seq_lengths = [len(s) for s in d1 + d2]
    print(f"  Length range    : {min(seq_lengths)}-{max(seq_lengths)} chars")
    print(f"  Mean length     : {sum(seq_lengths)/len(seq_lengths):.0f} chars")

    return d1, d2, seq_type, alphabet


def stage_represent(d1, d2, alphabet, args: argparse.Namespace):
    """Stage 2 - Build k-mer and projected-embedding representations."""
    print("\n" + "=" * 62)
    print("STAGE 2 - Building Representations")
    print("=" * 62)

    t0 = time.time()
    reps = build_representations(d1, d2, alphabet, k=args.k, embed_dim=args.embed_dim)
    elapsed = time.time() - t0

    print(f"  K-mer (k={args.k}) feature dim  : {reps['d1_kmer'].shape[1]}")
    if "d1_embed" in reps:
        print(f"  Projected embedding dim : {reps['d1_embed'].shape[1]}")
    print(f"  Time elapsed            : {elapsed:.2f}s")
    return reps


def stage_validate(d1, d2, alphabet, args: argparse.Namespace):
    """
    Metric sanity check.

    Two conditions:
      HIGH redundancy — D1 is a strict subset of D2 (D1 IN D2).
                        Expect redundancy score near 1.0.
      LOW  redundancy — standard random split (D1 NOT in D2).
                        Expect redundancy score near 0.0.

    If HIGH >> LOW the metric is working correctly.
    """
    print("\n" + "=" * 62)
    print("VALIDATION - Metric Sanity Check")
    print("=" * 62)

    all_seqs = d1 + d2
    rng_val = random.Random(args.seed + 99)

    # HIGH: draw a small D1 from the full pool; keep full pool as D2
    n_check = min(50, len(all_seqs) // 4)
    val_d1 = rng_val.sample(all_seqs, n_check)
    val_d2 = all_seqs[:]  # D2 contains D1

    r_high = build_representations(val_d1, val_d2, alphabet, k=args.k, embed_dim=args.embed_dim)
    res_high = ablation_information_retained(
        r_high["d1_kmer"], r_high["d2_kmer"],
        threshold=args.threshold, k_recon=args.k_recon, seed=args.seed,
    )

    # LOW: standard random split (D1 disjoint from D2)
    rng_low = random.Random(args.seed + 100)
    shuffled = all_seqs[:]
    rng_low.shuffle(shuffled)
    n_low = n_check
    low_d1 = shuffled[:n_low]
    low_d2 = shuffled[n_low:]

    r_low = build_representations(low_d1, low_d2, alphabet, k=args.k, embed_dim=args.embed_dim)
    res_low = ablation_information_retained(
        r_low["d1_kmer"], r_low["d2_kmer"],
        threshold=args.threshold, k_recon=args.k_recon, seed=args.seed,
    )

    print(f"  HIGH-redundancy (D1 subset of D2)  : R = {res_high['redundancy_score']:.4f}  (expect ~1.0)")
    print(f"  LOW-redundancy  (D1 disjoint D2)   : R = {res_low['redundancy_score']:.4f}  (expect ~0.0)")
    passed = res_high["redundancy_score"] > res_low["redundancy_score"]
    print(f"  Sanity check {'PASSED' if passed else 'FAILED'}: HIGH > LOW = {passed}")

    p = save_validation_plot(res_high["redundancy_score"], res_low["redundancy_score"])
    print(f"  Saved: {p}")

    return res_high["redundancy_score"], res_low["redundancy_score"]


def stage_analyse(reps, args: argparse.Namespace):
    """Stage 3 + 4 - Redundancy analysis and ablation experiment."""
    print("\n" + "=" * 62)
    print("STAGE 3+4 - Redundancy Analysis & Ablation")
    print("=" * 62)

    results = {}

    print("\n  [K-mer vectors]")
    res_kmer = ablation_information_retained(
        reps["d1_kmer"], reps["d2_kmer"],
        threshold=args.threshold, k_recon=args.k_recon,
    )
    results["K-mer"] = res_kmer
    _print_summary(res_kmer, args.threshold)

    if "d1_embed" in reps:
        print("\n  [Projected Embedding]")
        res_embed = ablation_information_retained(
            reps["d1_embed"], reps["d2_embed"],
            threshold=args.threshold, k_recon=args.k_recon,
        )
        results["Embedding"] = res_embed
        _print_summary(res_embed, args.threshold)

    return results


def _print_summary(res: dict, threshold: float) -> None:
    nn = res["nn_result"]
    rc = res["recon_result"]
    print(f"    Coverage @ t={threshold:.2f}   : {nn['coverage_pct']:.1f}%")
    print(f"    Mean NN similarity      : {nn['mean_similarity']:.4f}  +/-{nn['std_similarity']:.4f}")
    print(f"    Mean recon MSE          : {rc['mean_mse']:.2e}  (baseline {rc['baseline_mse']:.2e})")
    print(f"    Redundancy Score        : {res['redundancy_score']:.4f}")


def stage_output(d1, d2, alphabet, results, args: argparse.Namespace):
    """Stage 5 - Generate all plots, tables, and size-sensitivity curve."""
    print("\n" + "=" * 62)
    print("STAGE 5 - Outputs")
    print("=" * 62)

    saved = []

    # 1. Similarity histograms
    for method, label in [("K-mer", f"K-mer (k={args.k})"), ("Embedding", "Projected Embedding")]:
        if method not in results:
            continue
        p = save_similarity_histogram(
            results[method]["nn_result"]["nn_scores"],
            method_label=label,
            threshold=args.threshold,
            filename=f"similarity_histogram_{method.lower()}",
        )
        saved.append(p)
        print(f"  Saved: {p}")

    # 2. Coverage curve
    t_k = results["K-mer"]["thresholds"]
    c_k = results["K-mer"]["coverage_fractions"]
    t_e = c_e = None
    if "Embedding" in results:
        t_e = results["Embedding"]["thresholds"]
        c_e = results["Embedding"]["coverage_fractions"]
    p = save_coverage_curve(t_k, c_k, t_e, c_e)
    saved.append(p)
    print(f"  Saved: {p}")

    # 3. Ablation bar chart
    p = save_ablation_plot(results)
    saved.append(p)
    print(f"  Saved: {p}")

    # 4. Size-sensitivity experiment — vary D1 size, keep D1/D2 ratio fixed
    print("\n  [Size sensitivity] D1 size vs redundancy score ...")
    all_seqs = d1 + d2
    rng = random.Random(args.seed)

    # Target D1 counts: 10 evenly-spaced points up to the full D1 size
    max_d1 = len(d1)
    step = max(1, max_d1 // 9)
    d1_targets = sorted(set(
        [10, 20] + list(range(step, max_d1 + 1, step)) + [max_d1]
    ))
    d1_targets = [n for n in d1_targets if n <= max_d1]

    size_results = []
    for n_d1_target in d1_targets:
        n_d2_target = max(10, int(n_d1_target * (1 - args.d1_fraction) / args.d1_fraction))
        n_needed = n_d1_target + n_d2_target
        if n_needed > len(all_seqs):
            continue
        subset = rng.sample(all_seqs, n_needed)
        d1s = subset[:n_d1_target]
        d2s = subset[n_d1_target:]
        r = build_representations(d1s, d2s, alphabet, k=args.k, embed_dim=args.embed_dim)
        ak = ablation_information_retained(r["d1_kmer"], r["d2_kmer"],
                                           threshold=args.threshold, seed=args.seed)
        ae = ablation_information_retained(r["d1_embed"], r["d2_embed"],
                                           threshold=args.threshold, seed=args.seed) \
             if "d1_embed" in r else ak
        size_results.append((n_d1_target, ak["redundancy_score"], ae["redundancy_score"]))
        print(f"     D1={n_d1_target:4d}  kmer_R={ak['redundancy_score']:.3f}  embed_R={ae['redundancy_score']:.3f}")

    if size_results:
        p = save_size_sensitivity_plot(size_results)
        saved.append(p)
        print(f"  Saved: {p}")

    # 5. Summary CSV
    dataset_sizes = {"n_d1": len(d1), "n_d2": len(d2)}
    df = build_summary_table(results, dataset_sizes)
    print("\n  -- Summary Table --")
    print(df.to_string(index=False))
    csv_path = Path("results") / "summary_table.csv"
    print(f"\n  Saved: {csv_path}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dataset Bottleneck Analysis — Real Biological Sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--source", choices=["uniprot", "ncbi", "auto"], default="auto",
                   help="Sequence database to use")
    p.add_argument("--n-total", type=int, default=1500,
                   help="Total sequences to download (before filtering)")
    p.add_argument("--d1-fraction", type=float, default=0.33,
                   help="Fraction of total assigned to D1")
    p.add_argument("--min-len", type=int, default=50,
                   help="Minimum sequence length (chars)")
    p.add_argument("--max-len", type=int, default=2000,
                   help="Maximum sequence length (chars)")
    p.add_argument("--force-dl", action="store_true",
                   help="Ignore cache and re-download")
    # Representation
    p.add_argument("-k", type=int, default=3,
                   help="K-mer length")
    p.add_argument("--embed-dim", type=int, default=64,
                   help="Random projection output dimension")
    # Analysis
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Similarity threshold t for coverage metric")
    p.add_argument("--k-recon", type=int, default=5,
                   help="Nearest neighbours for reconstruction proxy")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    banner = (
        "\n  " + "=" * 60 + "\n"
        "  Dataset Bottleneck Analysis  -  Biosecurity Prototype\n"
        "  Real sequences from UniProt / NCBI\n"
        "  " + "=" * 60
    )
    print(banner)

    t_start = time.time()

    d1, d2, seq_type, alphabet = stage_load(args)
    stage_validate(d1, d2, alphabet, args)
    reps    = stage_represent(d1, d2, alphabet, args)
    results = stage_analyse(reps, args)
    stage_output(d1, d2, alphabet, results, args)

    print(f"\nDone. Pipeline complete in {time.time() - t_start:.1f}s")
    print("  All outputs in ./results/\n")


if __name__ == "__main__":
    main()
