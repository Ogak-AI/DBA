# Dataset Bottleneck Analysis (DBA)

> **AIxBio Hackathon — April 24–26, 2026**  
> *Can an AI adversary reconstruct restricted biological sequences from the remaining public corpus?*

DBA is a fast, validated framework that quantifies the **reconstruction gap** between a restricted sequence set (D1) and a public reference corpus (D2). It answers the question every biosecurity screening programme needs to ask before deployment:

> "Does removing these sequences actually withhold meaningful information from an AI-equipped adversary — or can they recover it from what's left?"

---

## What this means for biosecurity

![Redundancy scores by representation method](results/representation_comparison.png)

The figure above shows redundancy scores (R ∈ [0, 1]; higher = more reconstructable) for three representation types on **4,844 real UniProt Swiss-Prot proteins** (cluster-aware split, D1=1,698 / D2=3,146):

| Adversary type | Representation | R (bootstrap) | 95% CI | vs. K-mer |
|---------------|---------------|--------------|--------|-----------|
| Sequence copier (BLAST) | K-mer (k=3) | **0.064** | [0.061, 0.067] | — |
| Lightweight ML | Rnd. Projection | **0.209** | [0.205, 0.213] | 3.3× |
| Language model | ESM-2 (8M params, n=150) | **0.459** | [0.415, 0.501] | **7.2×** |
| Toxin proteins (K-mer) | K-mer (k=3) | **0.027** | [0.023, 0.030] | −59% vs random |
| K-mer null model (permuted D2) | K-mer | **0.010** | — | — |

*Wilcoxon K-mer vs ESM-2: n=150, p = 2.3 × 10⁻²⁶. All bootstrap CIs from n=200 resamples.*

**Key findings:**

1. **Cluster-aware splitting confirms genuine barriers.** At sequence-identity level, cluster-aware restrictions create real but imperfect information barriers (K-mer R = 0.064 vs null model R = 0.010). Zero sequences are recoverable at τ = 0.90 similarity under k-mer screening.

2. **ESM-2 reveals a 7.2× gap.** A protein language model adversary achieves 67% coverage of restricted sequences at cosine similarity ≥ 0.90 — compared to 0% for k-mer. Screening thresholds calibrated on BLAST-style identity underestimate AI-adversary reconstruction potential by roughly an order of magnitude.

3. **Toxin proteins are more compositionally isolated.** Biosecurity-relevant toxin families score 59% lower than random proteins (R = 0.027 vs 0.064) — sequence-level screening of toxin categories creates stronger information barriers than a random-protein baseline predicts.

4. **Random projections are not a reliable proxy for learned embeddings.** The random-projection null model (R = 0.217) exceeds the real embedding R (0.209), indicating the projection captures marginal k-mer statistics rather than genuine cross-dataset structure. ESM-2 is the only representation showing substantial above-null signal.

**Recommendation:** Calibrate screening thresholds using protein language model embeddings (ESM-2 or equivalent), not BLAST identity. The 7.2× representation gap is the *AI threat multiplier* — the factor by which language-model-aided adversaries exceed the reconstruction potential assumed by sequence-identity-based policy.

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. No GPU required for the core pipeline. ESM-2 evaluation requires `transformers` and `torch` (CPU-only, ~30s per 100 sequences).

---

## Running the pipeline

### Quickstart (downloads and caches data automatically)
```bash
python main.py
```

### Full options
```bash
python main.py \
  --n-total 2000 \           # sequences to download
  --split-mode cluster \     # cluster-aware split (default); or 'random'
  --n-bootstrap 200 \        # bootstrap CI resamples
  --toxin \                  # run toxin-protein experiment
  --seed 42
```

### ESM-2 evaluation (separate script — ~30s per 100 seqs on CPU)
```bash
python run_esm2.py --n-total 2000 --esm2-subset 100 --n-bootstrap 50
# Writes results/esm2_results.txt
```

### All flags (`main.py`)
```
--source          {uniprot,ncbi,auto}   Data source              [auto]
--n-total         INT                   Total seqs to download   [2000]
--split-mode      {random,cluster}      Split method             [cluster]
--d1-fraction     FLOAT                 Fraction assigned D1     [0.33]
--n-bootstrap     INT                   Bootstrap resamples      [200]
--toxin                                 Run toxin experiment
--no-esm2                               Skip ESM-2 (core pipeline)
--esm2-subset     INT                   Max seqs for ESM-2       [150]
--min-len         INT                   Min sequence length      [50]
--max-len         INT                   Max sequence length      [2000]
--force-dl                              Ignore cache, re-download
-k                INT                   K-mer length             [3]
--embed-dim       INT                   Projection dimension     [64]
--threshold       FLOAT                 Coverage threshold τ     [0.90]
--seed            INT                   Random seed              [42]
```

---

## Outputs (`results/`)

| File | Description |
|------|-------------|
| `summary_table.csv` | All metrics per method with CIs |
| `representation_comparison.png` | Redundancy scores (mean ± 95% CI) by method |
| `similarity_histogram_k_mer.png` | NN similarity distribution (k-mer) |
| `similarity_histogram_embedding.png` | NN similarity distribution (embedding) |
| `coverage_vs_threshold.png` | Coverage curve swept over τ ∈ [0, 1] |
| `ablation_comparison.png` | Coverage / similarity / score bar chart |
| `toxin_vs_random.png` | Toxin proteins vs random proteins |
| `size_sensitivity.png` | Redundancy score vs D1 size |
| `validation_sanity_check.png` | HIGH/LOW sanity check |
| `esm2_results.txt` | ESM-2 metrics (from `run_esm2.py`) |

---

## How DBA works

```
UniProt Swiss-Prot (real, public)
         │
         ▼  fetch_sequences() — cached FASTA
    1,929 sequences (quality filtered)
         │
         ▼  cluster_aware_split()
         │   TruncatedSVD (100 dim) → MiniBatchKMeans (150 clusters)
         │   Whole clusters → D1 (33%) or D2 (67%)
    ┌────┴────┐
   D1 (637)  D2 (1,292)
    │          │
    ▼          ▼
  k-mer vectors (8,000 dim, L1-norm)
  random projection (64 dim)
  ESM-2 embeddings (320 dim, optional)
         │
         ▼  redundancy_analysis
  NN overlap · coverage curve · reconstruction error
  bootstrap CIs (n=200) · null model · Wilcoxon test
         │
         ▼
  Redundancy Score R ∈ [0,1]  →  plots + CSV
```

### Redundancy score formula

```
R = 0.5 × Coverage@τ  +  0.5 × (1 − norm_MSE)

norm_MSE = MSE(x, x̂_NN) / MSE(x, x̂_random)
```

- **R → 1**: D1 broadly redundant in D2; restriction creates minimal barrier  
- **R → 0**: D1 genuinely unique; restriction is a real information bottleneck

---

## Interpreting results

| R value | Interpretation |
|---------|---------------|
| > 0.65 | AI-adversary can reconstruct most of D1; tighter restriction or embedding-aware policy needed |
| 0.15–0.65 | Partial barrier; sequence-identity screening partially effective but embedding gap present |
| < 0.15 | Strong information barrier at sequence level; verify with ESM-2 to check embedding gap |

---

## Project structure

```
DBA/
├── main.py                      ← pipeline orchestrator (cluster split, bootstrap, toxin)
├── run_esm2.py                  ← standalone ESM-2 experiment
├── requirements.txt
├── README.md
├── report_draft.md              ← full technical report with figures
├── data/                        ← cached FASTA files
├── src/
│   ├── data_loader.py           ← UniProt/NCBI download, cache, toxin query
│   ├── clustering.py            ← cluster-aware split (SVD + MiniBatchKMeans)
│   ├── representation.py        ← k-mer, random projection, ESM-2
│   ├── redundancy_analysis.py   ← metrics, bootstrap CI, null model, Wilcoxon
│   ├── esm_encoder.py           ← ESM-2 lazy loader (CPU, graceful fallback)
│   └── visualisation.py         ← all plots + CSV export
└── results/                     ← generated output
```

---

## Safety & ethics

- All sequences from UniProt Swiss-Prot (reviewed, public) or NCBI RefSeq
- No Select Agent sequences; toxin experiment uses public Swiss-Prot annotations only
- Analysis is purely statistical — no sequences are reconstructed or synthesised
- Tool is designed for defensive audit use by screening programme designers

Full dual-use discussion in Appendix A of `report_draft.md`.
