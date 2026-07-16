"""PubMedBERT [CLS] encoder over node texts with content-hashed disk cache.

Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (768d).

Cache invalidation
------------------
Per Codex review #1 CRITICAL #2: cache key includes a SHA256 of
(model, tag, max_length, node_ids, texts). Stale tags trigger rebuild.

Truncation tracking
-------------------
Per Codex review #1 WARN #11: encoder reports the fraction of texts
that were truncated to `max_length` tokens, so we can decide whether
to raise it for drug profiles.

Cache layout:
  Code/data/KG/_merged_kg/_cache/screen1_tag_init/<tag>__pubmedbert.pt
    payload = dict(
        node_ids: list[str],
        embeddings: Tensor[N, 768] fp16,
        content_hash: str (sha256 hex),
        model: str,
        text_source_tag: str,
        max_length: int,
        n_empty: int,
        n_truncated: int,
        elapsed_sec: float,
    )
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import time

import torch
from torch.utils.data import Dataset, DataLoader

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
CACHE_DIR = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "_cache" / "screen1_tag_init"

PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def _content_hash(node_ids: list[str], texts: list[str], tag: str, max_length: int) -> str:
    """Stable SHA256 over (model, tag, max_length, node_ids, texts)."""
    h = hashlib.sha256()
    h.update(PUBMEDBERT_MODEL.encode())
    h.update(b"||")
    h.update(tag.encode())
    h.update(b"||")
    h.update(str(max_length).encode())
    h.update(b"||")
    for nid in node_ids:
        h.update(nid.encode())
        h.update(b"\x1e")
    h.update(b"||")
    for t in texts:
        h.update(t.encode())
        h.update(b"\x1e")
    return h.hexdigest()


class _TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def _cache_path(tag: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = tag.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}__pubmedbert.pt"


def encode_pubmedbert(
    node_ids: list[str],
    texts: list[str],
    *,
    tag: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
    force_rebuild: bool = False,
) -> torch.Tensor:
    """Encode a list of texts via PubMedBERT [CLS], return float16 tensor.

    Cache key = content_hash(model, tag, max_length, node_ids, texts).
    """
    assert len(node_ids) == len(texts), "node_ids and texts must be same length"
    cache = _cache_path(tag)
    expected_hash = _content_hash(node_ids, texts, tag, max_length)

    if cache.exists() and not force_rebuild:
        try:
            payload = torch.load(cache, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"[encoder] cache UNREADABLE ({e}); rebuilding {cache.name}")
            payload = None
        if payload is not None:
            cached_hash = payload.get("content_hash")
            if cached_hash == expected_hash:
                print(f"[encoder] cache HIT (content_hash match): {cache.name}, N={len(node_ids):,}")
                return payload["embeddings"]
            else:
                print(f"[encoder] cache STALE (content_hash mismatch); rebuilding {cache.name}")
                print(f"           expected={expected_hash[:12]}..., found={(cached_hash or 'NONE')[:12]}...")

    from transformers import AutoTokenizer, AutoModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[encoder] loading {PUBMEDBERT_MODEL} on {device}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
    model = AutoModel.from_pretrained(PUBMEDBERT_MODEL).to(device).eval()

    out = torch.zeros((len(texts), 768), dtype=torch.float16)
    n_empty = 0
    n_truncated = 0
    with torch.inference_mode():
        loader = DataLoader(_TextDataset(texts), batch_size=batch_size, shuffle=False)
        progress_step = max(1, len(loader) // 20)
        for bi, batch_texts in enumerate(loader):
            mask_nonempty = [bool(t and t.strip()) for t in batch_texts]
            if not any(mask_nonempty):
                n_empty += len(batch_texts)
                continue
            nonempty_idx = [i for i, m in enumerate(mask_nonempty) if m]
            sub_texts = [batch_texts[i] for i in nonempty_idx]
            enc = tok(
                sub_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length,
                return_overflowing_tokens=False,
            )
            # Count truncations: tokenize WITHOUT max_length to compare
            raw_lens = [len(tok.encode(t, add_special_tokens=True)) for t in sub_texts]
            n_truncated += sum(1 for L in raw_lens if L > max_length)

            enc = enc.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(**enc)
                cls = outputs.last_hidden_state[:, 0, :]
            cls = cls.to(torch.float16).cpu()
            global_offset = bi * batch_size
            for local_i, sub_i in enumerate(nonempty_idx):
                out[global_offset + sub_i] = cls[local_i]
            n_empty += (len(batch_texts) - len(nonempty_idx))
            if bi % progress_step == 0:
                print(f"[encoder] batch {bi}/{len(loader)} ({bi*batch_size:,} done), "
                      f"elapsed {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    payload = {
        "node_ids": node_ids,
        "embeddings": out,
        "content_hash": expected_hash,
        "model": PUBMEDBERT_MODEL,
        "text_source_tag": tag,
        "max_length": max_length,
        "n_empty": n_empty,
        "n_truncated": n_truncated,
        "elapsed_sec": elapsed,
    }
    torch.save(payload, cache)
    trunc_pct = 100.0 * n_truncated / max(1, len(texts) - n_empty)
    print(f"[encoder] DONE tag={tag} N={len(texts):,} n_empty={n_empty:,} "
          f"n_truncated={n_truncated:,} ({trunc_pct:.1f}% of non-empty) "
          f"elapsed={elapsed:.0f}s -> {cache.name}")
    return out


if __name__ == "__main__":
    toy_texts = [
        "Aspirin is a salicylate used to treat pain, fever, and inflammation.",
        "Ibuprofen is a nonsteroidal anti-inflammatory drug for pain and fever.",
        "Hemoglobin is the iron-containing oxygen-transport metalloprotein in red blood cells.",
        "",
    ]
    toy_ids = ["t1", "t2", "t3", "t4"]
    emb = encode_pubmedbert(toy_ids, toy_texts, tag="_smoke", batch_size=4)
    emb_f = emb.float()
    cos = torch.nn.functional.cosine_similarity
    print()
    print(f"sim(Aspirin, Ibuprofen)   = {cos(emb_f[0:1], emb_f[1:2]).item():.4f}  (should be HIGH)")
    print(f"sim(Aspirin, Hemoglobin)  = {cos(emb_f[0:1], emb_f[2:3]).item():.4f}  (should be LOWER)")
    print(f"sim(Aspirin, empty)       = {cos(emb_f[0:1], emb_f[3:4]).item():.4f}  (should be 0)")
    print(f"||empty|| = {emb_f[3].norm().item():.4f}  (should be 0)")
