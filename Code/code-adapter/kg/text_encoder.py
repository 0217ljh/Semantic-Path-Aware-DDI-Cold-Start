"""Stage 2 / Part B / atom B3a - swappable FROZEN text encoder.

Default = PubMedBERT (biomedical, already cached locally under /mnt/g/hf_cache),
mean-pooled over tokens to a fixed sentence vector. The adapter's z_m/z_r geometry
depends only on the `.encode()` contract, so the encoder can be swapped later
(different biomedical model, instruction-tuned embedder, etc.) without touching
node_embed. Frozen: no gradients, eval mode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
_HF_CACHE_CANDIDATES = [Path("/mnt/g/hf_cache"), Path.home() / ".cache" / "huggingface"]
_MAX_TOKENS = 256

#: name -> (hf model id, pooling). Pooling follows each model's ORIGINAL recipe:
#: SapBERT / MedCPT embed the [CLS] token; PubMedBERT / BioBERT / BioLORD mean-pool.
ENCODERS = {
    "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "mean"),
    "biobert":    ("dmis-lab/biobert-base-cased-v1.2", "mean"),
    "sapbert":    ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "cls"),
    "medcpt":     ("ncbi/MedCPT-Article-Encoder", "cls"),
    "biolord":    ("FremyCompany/BioLORD-STAMB2-v1", "mean"),
}


def _find_cache_dir() -> str | None:
    for c in _HF_CACHE_CANDIDATES:
        if (c / "hub").is_dir():
            return str(c / "hub")
    return None


class HFEncoder:
    """Frozen HuggingFace sentence encoder. pooling='mean' (masked mean over the
    last layer) or 'cls' (first-token embedding, the recipe for SapBERT/MedCPT).
    The adapter only depends on `.encode()`, so any model can be swapped in."""

    def __init__(self, model_id: str = DEFAULT_MODEL, pooling: str = "mean",
                 device: str | None = None, cache_dir: str | None = None, log=print):
        import torch  # local import: heavy dep, only when an encoder is built
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self._torch = torch
        self.name = model_id
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = cache_dir or _find_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        log(f"[B3a] encoder {model_id} pooling={pooling} dim={self.dim} device={self.device}")

    def encode(self, texts: list[str], batch_size: int = 64, log=None) -> np.ndarray:
        """Return float32 [len(texts), dim] embeddings (unnormalized)."""
        torch = self._torch
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            chunk = [t if (t and t.strip()) else " " for t in texts[s:s + batch_size]]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=_MAX_TOKENS, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state           # [b, L, d]
                if self.pooling == "cls":
                    emb = h[:, 0]                                  # [CLS]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    emb = (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean
            out[s:s + batch_size] = emb.float().cpu().numpy()
            if log and (s // batch_size) % 50 == 0:
                log(f"[B3a] encoded {min(s + batch_size, len(texts))}/{len(texts)}")
        return out


# backward-compatible alias (older code used PubMedBertEncoder)
PubMedBertEncoder = HFEncoder


def make_encoder(name_or_id: str = "pubmedbert", pooling: str | None = None,
                 log=print) -> HFEncoder:
    """Build an encoder by registry key (pubmedbert/biobert/sapbert/medcpt/biolord)
    or by a raw HF model id (then `pooling` defaults to 'mean' unless given)."""
    if name_or_id in ENCODERS:
        mid, pool = ENCODERS[name_or_id]
        return HFEncoder(mid, pooling=pooling or pool, log=log)
    return HFEncoder(name_or_id, pooling=pooling or "mean", log=log)


def default_encoder(log=print) -> HFEncoder:
    return make_encoder("pubmedbert", log=log)


__all__ = ["HFEncoder", "PubMedBertEncoder", "make_encoder", "default_encoder",
           "ENCODERS", "DEFAULT_MODEL"]
