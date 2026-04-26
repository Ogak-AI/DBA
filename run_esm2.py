"""
run_esm2.py — standalone ESM-2 experiment script.

Downloads ESM-2 (first run only), encodes 150 D1 + 150 D2 sequences,
and appends ESM-2 redundancy metrics to results/esm2_results.txt.

Usage:
    python run_esm2.py [--n-total 2000] [--esm2-subset 150] [--seed 42]
"""
import argparse, logging, time
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-total",     type=int, default=2000)
    p.add_argument("--esm2-subset", type=int, default=150)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=100)
    args = p.parse_args()

    from src.data_loader    import fetch_sequences, split_datasets, detect_alphabet
    from src.representation import kmer_vectorise
    from src.clustering     import cluster_aware_split
    from src.esm_encoder    import esm2_available, esm2_embed
    from src.redundancy_analysis import (
        ablation_information_retained,
        bootstrap_redundancy_ci,
        wilcoxon_test,
    )

    if not esm2_available():
        print("ESM-2 not available — run: pip install transformers torch")
        return

    print("\nLoading sequences ...")
    records = fetch_sequences(n=args.n_total, source="uniprot")
    seqs    = [s for _, s in records]
    _, alphabet = detect_alphabet(seqs)

    print("Computing k-mer vectors and splitting ...")
    all_kmer = kmer_vectorise(seqs, alphabet, k=3)
    d1, d2, d1_idx, d2_idx, _ = cluster_aware_split(
        seqs, all_kmer, d1_fraction=0.33, seed=args.seed
    )

    n1 = min(len(d1), args.esm2_subset)
    n2 = min(len(d2), args.esm2_subset)
    d1k = all_kmer[d1_idx][:n1]
    d2k = all_kmer[d2_idx][:n2]

    print(f"\nEncoding {n1} D1 + {n2} D2 sequences with ESM-2 ...")
    t0 = time.time()
    d1_esm = esm2_embed(d1[:n1])
    d2_esm = esm2_embed(d2[:n2])
    t_enc  = time.time() - t0
    print(f"ESM-2 encoding: {t_enc:.1f}s")

    print("\nComputing ESM-2 redundancy metrics ...")
    res_esm  = ablation_information_retained(d1_esm, d2_esm, threshold=0.90)
    ci_esm   = bootstrap_redundancy_ci(d1_esm, d2_esm, n_bootstrap=args.n_bootstrap, seed=args.seed)

    print("\nComputing k-mer metrics on same subset ...")
    res_kmer = ablation_information_retained(d1k, d2k, threshold=0.90)
    ci_kmer  = bootstrap_redundancy_ci(d1k, d2k, n_bootstrap=args.n_bootstrap, seed=args.seed)

    wtest = wilcoxon_test(
        res_kmer["nn_result"]["nn_scores"],
        res_esm["nn_result"]["nn_scores"],
    )

    print("\n" + "=" * 60)
    print(f"  K-mer  R = {ci_kmer['mean']:.4f}  [95% CI: {ci_kmer['ci_low']:.4f} – {ci_kmer['ci_high']:.4f}]")
    print(f"  ESM-2  R = {ci_esm['mean']:.4f}  [95% CI: {ci_esm['ci_low']:.4f} – {ci_esm['ci_high']:.4f}]")
    print(f"  Ratio ESM-2/k-mer: {ci_esm['mean']/max(ci_kmer['mean'],1e-9):.2f}x")
    if wtest["available"]:
        print(f"  Wilcoxon K-mer vs ESM-2: p = {wtest['pvalue']:.2e}")
    print("=" * 60)

    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    with open("results/esm2_results.txt", "w") as f:
        f.write(f"ESM-2 subset: {n1} D1 + {n2} D2 sequences\n")
        f.write(f"Encoding time: {t_enc:.1f}s\n")
        f.write(f"K-mer R: {ci_kmer['mean']:.4f}  [95% CI: {ci_kmer['ci_low']:.4f} – {ci_kmer['ci_high']:.4f}]\n")
        f.write(f"ESM-2  R: {ci_esm['mean']:.4f}  [95% CI: {ci_esm['ci_low']:.4f} – {ci_esm['ci_high']:.4f}]\n")
        f.write(f"Ratio ESM-2/k-mer: {ci_esm['mean']/max(ci_kmer['mean'],1e-9):.2f}x\n")
        f.write(f"ESM-2 Coverage@0.90: {res_esm['nn_result']['coverage_pct']:.2f}%\n")
        f.write(f"ESM-2 Mean NN sim: {res_esm['nn_result']['mean_similarity']:.4f}\n")
        if wtest["available"]:
            f.write(f"Wilcoxon p: {wtest['pvalue']:.2e}\n")
    print("\nSaved: results/esm2_results.txt")


if __name__ == "__main__":
    main()
