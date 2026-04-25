# Dataset Bottleneck Analysis

> **Biosecurity Hackathon Prototype**  
> Testing the hypothesis: *"Removing a specific biological dataset does NOT significantly reduce accessible biological knowledge, because equivalent information can be reconstructed from other public sources."*

---

## Overview

This pipeline quantifies **knowledge redundancy** between two biological sequence datasets, split from a single real-world download:

| Symbol | Role | Description |
|--------|------|-------------|
| **D1** | Primary (restricted) | The dataset we simulate "removing" |
| **D2** | Reference | The remaining public sequences |

It answers: **"If we restrict access to D1, how much biological capability is actually lost?"**

---

## Data Sources (Real, Public)

| Priority | Source | URL | Content |
|----------|--------|-----|---------|
| 1st | **UniProt Swiss-Prot** | `rest.uniprot.org` | Reviewed protein records |
| 2nd | **NCBI Protein** | `eutils.ncbi.nlm.nih.gov` | RefSeq protein records |

- Downloaded via official REST APIs — no synthetic or mock data
- Sequences cached in `data/` after first download (skip re-download on reruns)
- Randomly split into D1 / D2 using a fixed seed for reproducibility

---

## Experiment Design

```
UniProt / NCBI
     │
     ▼  fetch_sequences()
 1500 real sequences
     │
     ▼  split_datasets()
  D1 (500)  ────────────────────────────────┐
  D2 (1000) ──► [k-mer vectors]             │
             ──► [random-projection embed]  │
                        │                  │
                        ▼                  │
              [NN Overlap]  ◄──────────────┘
              [Coverage Sweep]
              [Reconstruction Proxy]
                        │
                        ▼
              Redundancy Score  →  plots + CSV
```

### Step 1 — Data Download & Split
- `--n-total 1500` sequences fetched (UniProt first, NCBI fallback)
- Quality filtered: length 50–2 000 chars, valid amino-acid alphabet
- 33% → D1, 67% → D2 (random, fixed seed)

### Step 2 — Sequence Representation
| Method | Description |
|--------|-------------|
| **K-mer frequency** | Sliding window (k=3), all 20^3 = 8000 features, L1-normalised |
| **Random-projection embed** | K-mer vectors → 64-D via Johnson-Lindenstrauss projection |

### Step 3 — Redundancy Metrics
| Metric | Formula |
|--------|---------|
| **NN Overlap** | For each D1 item cosine-similarity to its best match in D2 |
| **Coverage @ τ** | % of D1 items with NN sim ≥ τ, for τ ∈ [0, 1] |
| **Reconstruction MSE** | Weighted sum of 5-NN from D2; mean squared error |

### Step 4 — Ablation
Simulates "D1 is removed": all metrics computed with D2 only → quantifies retained information.

### Step 5 — Redundancy Score
```
R = 0.5 × Coverage@0.90  +  0.5 × (1 − normalised_MSE)
```
- **R → 1**: D1 is fully covered by D2 → restricting D1 causes minimal capability loss  
- **R → 0**: D1 is unique → its restriction is a genuine bottleneck

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10 required. No additional authentication needed.

---

## Running the Pipeline

### Quickstart (downloads & caches data automatically)
```bash
python main.py
```

### Custom options
```bash
python main.py \
  --source uniprot \     # or ncbi / auto
  --n-total 2000 \       # sequences to download
  --d1-fraction 0.4 \    # 40% → D1
  --threshold 0.85 \     # coverage threshold
  --force-dl             # ignore cache, re-download
```

### All options
```
--source       {uniprot,ncbi,auto}  Data source             [auto]
--n-total      INT                  Total seqs to download  [1500]
--d1-fraction  FLOAT                Fraction assigned D1    [0.33]
--min-len      INT                  Min sequence length     [50]
--max-len      INT                  Max sequence length     [2000]
--force-dl                          Ignore cache
-k             INT                  K-mer length            [3]
--embed-dim    INT                  Projection dimension    [64]
--threshold    FLOAT                Coverage threshold τ    [0.90]
--k-recon      INT                  Recon neighbours        [5]
--seed         INT                  Random seed             [42]
```

---

## Outputs (`results/`)

| File | Description |
|------|-------------|
| `summary_table.csv` | All key metrics per representation method |
| `similarity_histogram_k-mer.png` | NN similarity distribution (k-mer vectors) |
| `similarity_histogram_embedding.png` | NN similarity distribution (projected embeddings) |
| `coverage_vs_threshold.png` | Coverage vs τ sweep for both methods |
| `ablation_comparison.png` | Bar chart: coverage / similarity / redundancy score |
| `size_sensitivity.png` | Redundancy score vs D1 size (real data subsets) |

---

## Interpreting Results

| Redundancy Score | Interpretation |
|-----------------|---------------|
| > 0.75 | D1 is broadly redundant; restricting it has little impact |
| 0.40–0.75 | D1 has partial unique content; some capability loss expected |
| < 0.40 | D1 is a genuine bottleneck; restriction causes significant knowledge loss |

---

## Safety & Ethics

- All sequences come from UniProt Swiss-Prot (reviewed proteins) or NCBI RefSeq
- No pathogen-specific queries are made; queries use `reviewed:true` (Swiss-Prot)
- Analysis is purely statistical — no sequences are reconstructed or synthesised
- No biological capability is demonstrated or transferred

---

## Project Structure

```
DBA/
├── main.py                     ← pipeline orchestrator
├── requirements.txt
├── README.md
├── data/                       ← cached FASTA files (created on first run)
│   └── uniprot_1500.fasta
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← UniProt/NCBI download, cache, split
│   ├── representation.py       ← k-mer vectors + random projection
│   ├── redundancy_analysis.py  ← NN overlap, coverage, reconstruction
│   └── visualisation.py        ← plots + CSV export
└── results/                    ← generated output (created on run)
    ├── summary_table.csv
    ├── similarity_histogram_*.png
    ├── coverage_vs_threshold.png
    ├── ablation_comparison.png
    └── size_sensitivity.png
```

---

## Extending to Larger Datasets

```python
# Use your own FASTA file
from src.data_loader import _parse_fasta, _filter_sequences, split_datasets

with open("my_sequences.fasta") as f:
    records = _parse_fasta(f.read())
records = _filter_sequences(records)
d1, d2 = split_datasets(records, d1_fraction=0.33)
```

For protein embeddings (ESM-2), replace `build_representations()`:
```bash
pip install transformers torch
# Use: facebook/esm2_t6_8M_UR50D via HuggingFace
```
