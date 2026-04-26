"""
esm_encoder.py
==============
ESM-2 protein language model embeddings — CPU, no GPU required.
Model: facebook/esm2_t6_8M_UR50D  (~30 MB, 6-layer, 320-dim hidden states)

Gracefully disabled if transformers / torch are not installed;
callers check esm2_available() before use.
"""
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
_model = None
_tokenizer = None
_load_attempted = False


def _load() -> bool:
    global _model, _tokenizer, _load_attempted
    if _model is not None:
        return True
    if _load_attempted:
        return False
    _load_attempted = True
    try:
        from transformers import EsmModel, EsmTokenizer
        import torch  # noqa: F401
        logger.info("Loading ESM-2 (%s) on CPU ...", _MODEL_NAME)
        _tokenizer = EsmTokenizer.from_pretrained(_MODEL_NAME)
        _model = EsmModel.from_pretrained(_MODEL_NAME)
        _model.eval()
        logger.info("ESM-2 loaded  —  hidden dim 320, 6 layers.")
        return True
    except ImportError:
        logger.warning(
            "transformers or torch not installed — ESM-2 disabled. "
            "Install with: pip install transformers torch"
        )
    except Exception as exc:
        logger.warning("ESM-2 load failed (%s) — skipping.", exc)
    return False


def esm2_available() -> bool:
    """Return True if the ESM-2 model can be loaded on this machine."""
    return _load()


def esm2_embed(
    sequences: List[str],
    batch_size: int = 8,
    max_len: int = 512,
) -> Optional[np.ndarray]:
    """
    Return mean-pooled ESM-2 embeddings, shape (n, 320).

    Parameters
    ----------
    sequences  : amino-acid strings (standard 20-letter alphabet)
    batch_size : sequences per forward pass (reduce if OOM)
    max_len    : residues; longer sequences are truncated

    Returns
    -------
    float32 array of shape (n, 320), or None if ESM-2 is unavailable.
    """
    if not _load():
        return None

    import torch

    out: List[np.ndarray] = []
    total = len(sequences)

    for start in range(0, total, batch_size):
        batch = [s[:max_len] for s in sequences[start: start + batch_size]]
        enc = _tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        with torch.no_grad():
            hidden = _model(**enc).last_hidden_state      # (B, L, 320)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)   # (B, 320)
        out.append(pooled.cpu().numpy())
        done = min(start + batch_size, total)
        if done % 50 == 0 or done == total:
            logger.info("  ESM-2: %d / %d sequences encoded", done, total)

    return np.vstack(out).astype(np.float32)
