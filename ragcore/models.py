"""埋め込みモデル (bge-m3) とリランカー (bge-reranker-v2-m3) の読み込み。

GPU があれば fp16 で載せる。無ければ CPU (遅いが動く)。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .config import model_ref

log = logging.getLogger(__name__)


def _device(want: str) -> str:
    import torch
    want = (want or "cuda").strip()
    if want.startswith("cuda") and torch.cuda.is_available():
        return want
    return "cpu"


def load_embedder(cfg: Dict[str, Any], base_dir: Path):
    import torch
    from sentence_transformers import SentenceTransformer

    mcfg = cfg.get("model") or {}
    ref = model_ref(base_dir, mcfg.get("name_or_path") or "BAAI/bge-m3")
    device = _device(mcfg.get("device", "cuda"))
    kwargs = {}
    if mcfg.get("fp16", True) and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    log.info("Loading embedder %s (device=%s, fp16=%s)", ref, device, bool(kwargs))
    model = SentenceTransformer(ref, device=device, model_kwargs=kwargs or None)
    model.max_seq_length = int(mcfg.get("max_seq_length", 1024))
    return model


def load_reranker(cfg: Dict[str, Any], base_dir: Path) -> Optional[Any]:
    """retrieval.reranker.enabled が false、または読めなければ None (密検索だけで動く)。"""
    rcfg = ((cfg.get("retrieval") or {}).get("reranker") or {})
    if not rcfg.get("enabled", False):
        log.info("Reranker disabled in config")
        return None
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        log.warning("Reranker needs sentence-transformers + torch — skipping: %s", e)
        return None
    ref = model_ref(base_dir, rcfg.get("name_or_path") or "BAAI/bge-reranker-v2-m3")
    device = _device((cfg.get("model") or {}).get("device", "cuda"))
    kwargs = {}
    if rcfg.get("fp16", True) and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    max_len = int(rcfg.get("max_length", 1024))
    log.info("Loading reranker %s (device=%s, fp16=%s, max_len=%d)", ref, device, bool(kwargs), max_len)
    try:
        try:
            return CrossEncoder(ref, max_length=max_len, device=device, model_kwargs=kwargs or None)
        except TypeError:                     # 古い sentence-transformers
            return CrossEncoder(ref, max_length=max_len, device=device, automodel_args=kwargs or None)
    except Exception as e:
        log.warning("Failed to load reranker (%s) — falling back to dense-only", e)
        return None
