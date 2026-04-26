"""
run_toxin_esm2.py — ESM-2 redundancy score for toxin proteins vs Swiss-Prot D2.

Encodes 416 toxin D1 sequences + up to --d2-subset D2 sequences with ESM-2
and reports toxin ESM-2 R alongside the k-mer baseline.

Usage:
    python run_toxin_esm2.py [--n-total 5000] [--d2-subset 500] [--n-bootstrap 50] [--seed 42]
"""
import argparse, logging, time
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-total",     type=int, default=5000)
    p.add_argument("--d2-subset",   type=int, default=500,
                   help="Max D2 sequences to encode with ESM-2")
    p.add_argument("--n-bootstrap", type=int, default=50)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    from src.data_loader       import fetch_sequences, fetch_toxin_sequences, split_datasets, detect_alphabet
    from src.representation    import kmer_vectorise
    from src.clustering        import cluster_aware_split
    from src.esm_encoder       import esm2_available, esm2_embed
    from src.redundancy_analysis import ablation_information_retained, bootstrap_redundancy_ci

    if not esm2_available():
        print("ESM-2 not available — run: pip install transformers torch")
        return

    print("\nLoading corpus ...")
    records = fetch_sequences(n=args.n_total, source="uniprot")
    seqs    = [s for _, s in records]
    _, alphabet = detect_alphabet(seqs)

    print("Cluster-aware split to get D2 ...")
    all_kmer = kmer_vectorise(seqs, alphabet, k=3)
    _, d2_seqs, _, d2_idx, _ = cluster_aware_split(
        seqs, all_kmer, d1_fraction=0.33, seed=args.seed
    )

    print("Fetching toxin sequences ...")
    toxin_records = fetch_toxin_sequences(n=500)
    toxin_seqs = [s for _, s in toxin_records]
    print(f"  Toxin D1: {len(toxin_seqs)} sequences")
    n_d2_actual = min(args.d2_subset, len(d2_seqs))
    d2_label = "full D2" if n_d2_actual >= len(d2_seqs) else f"subset of {n_d2_actual}"
    print(f"  Swiss-Prot D2: {len(d2_seqs)} sequences (using {d2_label})")

    # K-mer baseline
    toxin_kmer = kmer_vectorise(toxin_seqs, alphabet, k=3)
    d2_kmer    = all_kmer[d2_idx][:args.d2_subset]
    res_km  = ablation_information_retained(toxin_kmer, d2_kmer)
    ci_km   = bootstrap_redundancy_ci(toxin_kmer, d2_kmer,
                                       n_bootstrap=args.n_bootstrap, seed=args.seed)

    # ESM-2
    import os
    n_d2_esm = min(args.d2_subset, len(d2_seqs))
    d2_for_esm = d2_seqs[:n_d2_esm]
    d2_cache = "results/d2_esm.npy"

    print(f"\nEncoding {len(toxin_seqs)} toxin sequences with ESM-2 ...")
    t0 = time.time()
    d1_esm = esm2_embed(toxin_seqs)

    if n_d2_esm >= len(d2_seqs) and os.path.exists(d2_cache):
        print(f"Loading precomputed D2 ESM-2 embeddings from {d2_cache} ...")
        d2_esm = np.load(d2_cache)
    else:
        print(f"Encoding {n_d2_esm} D2 sequences with ESM-2 ...")
        d2_esm = esm2_embed(d2_for_esm)
        if d2_esm is not None and n_d2_esm >= len(d2_seqs):
            np.save(d2_cache, d2_esm)
            print(f"Saved D2 embeddings to {d2_cache} for future reuse.")

    t_enc = time.time() - t0
    print(f"ESM-2 encoding: {t_enc:.1f}s")

    if d1_esm is None or d2_esm is None:
        print("ESM-2 encoding failed.")
        return

    res_esm = ablation_information_retained(d1_esm, d2_esm)
    ci_esm  = bootstrap_redundancy_ci(d1_esm, d2_esm,
                                       n_bootstrap=args.n_bootstrap, seed=args.seed)

    print("\n" + "=" * 60)
    print(f"  Toxin K-mer  R = {ci_km['mean']:.4f}  "
          f"[95% CI: {ci_km['ci_low']:.4f} – {ci_km['ci_high']:.4f}]")
    print(f"  Toxin ESM-2  R = {ci_esm['mean']:.4f}  "
          f"[95% CI: {ci_esm['ci_low']:.4f} – {ci_esm['ci_high']:.4f}]")
    print(f"  Toxin ESM-2 Coverage@0.90: {res_esm['nn_result']['coverage_pct']:.2f}%")
    print(f"  Toxin ESM-2 Mean NN sim: {res_esm['nn_result']['mean_similarity']:.4f}")
    print(f"  Ratio ESM-2/K-mer: {ci_esm['mean'] / max(ci_km['mean'], 1e-9):.2f}x")
    print("=" * 60)

    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    with open("results/toxin_esm2_results.txt", "w") as f:
        f.write(f"Toxin D1: {len(toxin_seqs)} sequences\n")
        f.write(f"D2 ESM-2: {n_d2_esm} sequences ({'full D2' if n_d2_esm >= len(d2_seqs) else 'subset'})\n")
        f.write(f"Encoding time: {t_enc:.1f}s\n")
        f.write(f"Toxin K-mer R: {ci_km['mean']:.4f}  "
                f"[95% CI: {ci_km['ci_low']:.4f} – {ci_km['ci_high']:.4f}]\n")
        f.write(f"Toxin ESM-2 R: {ci_esm['mean']:.4f}  "
                f"[95% CI: {ci_esm['ci_low']:.4f} – {ci_esm['ci_high']:.4f}]\n")
        f.write(f"Toxin ESM-2 Coverage@0.90: {res_esm['nn_result']['coverage_pct']:.2f}%\n")
        f.write(f"Toxin ESM-2 Mean NN sim: {res_esm['nn_result']['mean_similarity']:.4f}\n")
        f.write(f"Ratio ESM-2/K-mer: {ci_esm['mean'] / max(ci_km['mean'], 1e-9):.2f}x\n")
    print("\nSaved: results/toxin_esm2_results.txt")


if __name__ == "__main__":
    main()
