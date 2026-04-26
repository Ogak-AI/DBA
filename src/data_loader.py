"""
data_loader.py
==============
Downloads REAL biological sequences from public databases.

Priority order
--------------
1. Local cache  (data/<source>.fasta)  — reused on every subsequent run
2. UniProt REST API  (reviewed Swiss-Prot proteins, FASTA)
3. NCBI E-utilities  (protein records, FASTA)

Raises RuntimeError if ALL sources fail — NEVER falls back to synthetic data.

Split strategy
--------------
After loading N sequences, a reproducible random split produces:
    D1  – the "restricted" dataset  (d1_fraction of total, default 33 %)
    D2  – the "reference" dataset   (remainder)

Safety note
-----------
Sequences are fetched from public, benign reference databases
(Swiss-Prot reviewed proteins, generic RefSeq).  No pathogen-specific
queries are made.
"""

import hashlib
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI requires an e-mail in the tool parameter for polite use
NCBI_TOOL    = "dataset_bottleneck_analysis"
NCBI_EMAIL   = "awajiogakfinomo@gmail.com"

REQUEST_TIMEOUT   = 60   # seconds per HTTP call
NCBI_FETCH_CHUNK  = 200  # IDs per efetch call (NCBI recommends ≤500)
RETRY_DELAYS      = [5, 15, 30]  # seconds between retries


# ── FASTA parser ─────────────────────────────────────────────────────────────

def _parse_fasta(text: str) -> List[Tuple[str, str]]:
    """
    Parse raw FASTA text.

    Returns
    -------
    List of (header, sequence) tuples – sequence is uppercase, no whitespace.
    """
    records: List[Tuple[str, str]] = []
    header = ""
    buf: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if header and buf:
                records.append((header, "".join(buf).upper()))
            header = line[1:].split()[0]   # accession only
            buf = []
        elif line:
            buf.append(line)
    if header and buf:
        records.append((header, "".join(buf).upper()))
    return records


def _filter_sequences(
    records: List[Tuple[str, str]],
    min_len: int = 50,
    max_len: int = 2000,
) -> List[Tuple[str, str]]:
    """Drop sequences that are too short, too long, or contain ambiguous chars."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_nt = set("ATCGN")
    out = []
    for h, s in records:
        if not (min_len <= len(s) <= max_len):
            continue
        chars = set(s)
        if chars <= valid_aa or chars <= valid_nt:
            out.append((h, s))
    return out


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _http_get(url: str, params: dict | None = None) -> str:
    """GET *url* with optional query params; retry on transient errors."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            logger.info("  Retry %d/%d — waiting %ds …", attempt, len(RETRY_DELAYS), delay)
            time.sleep(delay)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DatasetBottleneckAnalysis/1.0"},
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < len(RETRY_DELAYS):
                logger.warning("  HTTP %d — will retry", e.code)
                continue
            raise
        except Exception as e:
            if attempt < len(RETRY_DELAYS):
                logger.warning("  Network error (%s) — will retry", e)
                continue
            raise
    raise RuntimeError(f"All retries exhausted for {url}")


# ── Source 1: UniProt ─────────────────────────────────────────────────────────

def _fetch_uniprot(n: int, query: str = "reviewed:true") -> List[Tuple[str, str]]:
    """
    Download up to *n* Swiss-Prot reviewed protein sequences via the
    UniProt REST API (v2025).

    Parameters
    ----------
    n     : number of sequences requested
    query : UniProt query string

    Returns
    -------
    List of (accession, sequence) tuples
    """
    logger.info("UniProt: requesting %d sequences (query='%s') …", n, query)
    params = {
        "query":   query,
        "format":  "fasta",
        "size":    str(min(n, 500)),   # API max per page is 500
    }
    records: List[Tuple[str, str]] = []
    next_url: str | None = UNIPROT_BASE + "?" + urllib.parse.urlencode(params)

    while next_url and len(records) < n:
        req = urllib.request.Request(
            next_url,
            headers={"User-Agent": "DatasetBottleneckAnalysis/1.0"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            # Check Link header for next page
            link_header = resp.headers.get("Link", "")
            next_url = None
            if 'rel="next"' in link_header:
                import re
                m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
                if m:
                    next_url = m.group(1)

        page_records = _parse_fasta(raw)
        records.extend(page_records)
        logger.info("  … fetched %d so far", len(records))
        time.sleep(0.5)   # be polite

    return records[:n]


# ── Source 2: NCBI E-utilities ────────────────────────────────────────────────

def _fetch_ncbi(n: int, db: str = "protein", term: str = "refseq[filter] NOT partial[title]") -> List[Tuple[str, str]]:
    """
    Download up to *n* protein records from NCBI via E-utilities.

    Parameters
    ----------
    n    : number of sequences requested
    db   : NCBI database ('protein' or 'nucleotide')
    term : Entrez query string
    """
    logger.info("NCBI: searching db=%s for %d sequences …", db, n)

    # Step 1 – e-search to get GI / accession list
    search_params = {
        "db":      db,
        "term":    term,
        "retmax":  str(n),
        "rettype": "json",
        "tool":    NCBI_TOOL,
        "email":   NCBI_EMAIL,
    }
    raw = _http_get(NCBI_ESEARCH, search_params)

    import json, re
    # Parse JSON response
    data = json.loads(raw)
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise RuntimeError("NCBI esearch returned no IDs")
    logger.info("  Got %d IDs from NCBI", len(ids))

    # Step 2 – e-fetch in chunks
    records: List[Tuple[str, str]] = []
    for i in range(0, len(ids), NCBI_FETCH_CHUNK):
        chunk = ids[i: i + NCBI_FETCH_CHUNK]
        fetch_params = {
            "db":      db,
            "id":      ",".join(chunk),
            "rettype": "fasta",
            "retmode": "text",
            "tool":    NCBI_TOOL,
            "email":   NCBI_EMAIL,
        }
        fasta_text = _http_get(NCBI_EFETCH, fetch_params)
        records.extend(_parse_fasta(fasta_text))
        logger.info("  … fetched %d so far", len(records))
        time.sleep(0.4)  # NCBI rate limit: ≤3 req/s without API key

    return records[:n]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(source: str, n: int) -> Path:
    return CACHE_DIR / f"{source}_{n}.fasta"


def _write_fasta_cache(path: Path, records: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for acc, seq in records:
            fh.write(f">{acc}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i: i + 60] + "\n")
    logger.info("Cached %d sequences → %s", len(records), path)


def _read_fasta_cache(path: Path) -> List[Tuple[str, str]]:
    with open(path) as fh:
        return _parse_fasta(fh.read())


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_sequences(
    n: int = 1500,
    source: str = "auto",
    min_len: int = 50,
    max_len: int = 2000,
    force_download: bool = False,
    uniprot_query: str = "reviewed:true",
) -> List[Tuple[str, str]]:
    """
    Fetch *n* real biological sequences from public databases.

    Parameters
    ----------
    n              : number of sequences to fetch (before quality filtering)
    source         : 'uniprot', 'ncbi', or 'auto' (tries UniProt first)
    min_len        : minimum accepted sequence length
    max_len        : maximum accepted sequence length
    force_download : ignore local cache and re-download
    uniprot_query  : UniProt query string (default: all reviewed Swiss-Prot)

    Returns
    -------
    List of (accession, sequence) tuples — NEVER empty.

    Raises
    ------
    RuntimeError if all sources fail.
    """
    import hashlib
    query_tag = hashlib.md5(uniprot_query.encode()).hexdigest()[:8]
    cache_src = source if source != "auto" else "uniprot"
    cache_file = CACHE_DIR / f"{cache_src}_{n}_{query_tag}.fasta"

    # ── Cache hit ────────────────────────────────────────────────────────────
    if not force_download and cache_file.exists():
        logger.info("Loading from cache: %s", cache_file)
        records = _read_fasta_cache(cache_file)
        records = _filter_sequences(records, min_len, max_len)
        if records:
            logger.info("Cache hit: %d sequences after filtering", len(records))
            return records
        logger.warning("Cache file was empty after filtering — re-downloading")

    # ── Live download ─────────────────────────────────────────────────────────
    records: List[Tuple[str, str]] = []
    errors: List[str] = []

    sources_to_try = ["uniprot", "ncbi"] if source == "auto" else [source]

    for src in sources_to_try:
        try:
            if src == "uniprot":
                records = _fetch_uniprot(n, query=uniprot_query)
            elif src == "ncbi":
                records = _fetch_ncbi(n)

            if records:
                _write_fasta_cache(cache_file, records)
                break
            else:
                errors.append(f"{src}: returned 0 records")

        except Exception as exc:
            errors.append(f"{src}: {exc}")
            logger.warning("Source '%s' failed: %s", src, exc)

    if not records:
        raise RuntimeError(
            "ALL data sources failed. Cannot proceed without real data.\n"
            + "\n".join(errors)
        )

    # ── Quality filtering ────────────────────────────────────────────────────
    before = len(records)
    records = _filter_sequences(records, min_len, max_len)
    logger.info(
        "Quality filter: %d -> %d sequences (len %d-%d)",
        before, len(records), min_len, max_len,
    )

    if not records:
        raise RuntimeError(
            "All downloaded sequences were filtered out (length / character check). "
            "Try adjusting --min-len / --max-len."
        )

    return records


def fetch_toxin_sequences(
    n: int = 500,
    min_len: int = 50,
    max_len: int = 2000,
    force_download: bool = False,
) -> List[Tuple[str, str]]:
    """
    Fetch reviewed UniProt proteins annotated with keyword 'toxin'.

    These are publicly curated, non-pathogen-specific entries used to
    represent a biosecurity-relevant functional category (D1 in the
    toxin experiment).
    """
    logger.info("Fetching toxin sequences from UniProt ...")
    return fetch_sequences(
        n=n,
        source="uniprot",
        min_len=min_len,
        max_len=max_len,
        force_download=force_download,
        uniprot_query="reviewed:true AND keyword:toxin",
    )


def split_datasets(
    records: List[Tuple[str, str]],
    d1_fraction: float = 0.33,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Randomly split sequence records into D1 (primary/restricted) and
    D2 (reference) by sequence string only.

    Parameters
    ----------
    records     : list of (accession, sequence) tuples
    d1_fraction : fraction assigned to D1
    seed        : reproducibility seed

    Returns
    -------
    (d1_sequences, d2_sequences) — lists of raw sequence strings
    """
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_d1 = max(1, int(len(shuffled) * d1_fraction))
    d1 = [seq for _, seq in shuffled[:n_d1]]
    d2 = [seq for _, seq in shuffled[n_d1:]]

    logger.info(
        "Split: D1=%d sequences, D2=%d sequences (%.0f%%/%.0f%%)",
        len(d1), len(d2),
        d1_fraction * 100, (1 - d1_fraction) * 100,
    )
    return d1, d2


# ── Alphabet detection ────────────────────────────────────────────────────────

def detect_alphabet(sequences: List[str]) -> Tuple[str, List[str]]:
    """
    Infer whether sequences are DNA or protein.

    Returns
    -------
    (seq_type, alphabet_list)  — seq_type is 'dna' or 'protein'
    """
    dna_chars = set("ATCGN")
    protein_chars = set("ACDEFGHIKLMNPQRSTVWY")

    sample = sequences[:100]
    dna_votes = sum(1 for s in sample if set(s) <= dna_chars)
    return (
        ("dna",     list("ATCG"))
        if dna_votes > len(sample) * 0.8
        else ("protein", list("ACDEFGHIKLMNPQRSTVWY"))
    )
