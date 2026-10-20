"""FAISS 索引 (内積 = 正規化済みベクトルのコサイン類似度)。

    flat       fp32・厳密        4 KB/ベクトル (bge-m3 1024 次元)
    flat_fp16  fp16・ほぼ厳密    2 KB/ベクトル
    flat_sq8   8 bit スカラー量子化 1 KB/ベクトル (学習が要る。既定)

どれも IndexIDMap2 で包み、ID = state.sqlite の chunks.chunk_id にする。
IDMap2 なので reconstruct (MMR で使う) と remove_ids (差分更新) ができる。
flat_sq8 は最初の train_size 件 (小さいコーパスなら全件) で量子化範囲を学習してから追加する。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

INDEX_FILE = "faiss.index"
META_FILE = "index_meta.json"
KINDS = ("flat", "flat_fp16", "flat_sq8")
_TRAINABLE = ("flat_sq8",)


def _write_index(index, path: Path) -> None:
    """faiss.write_index は C の fopen で開くため、Windows で日本語を含むパスを開けない。
    メモリ上で直列化して Python で書く (どんなパスでも書ける)。"""
    import faiss
    buf = faiss.serialize_index(index)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def _read_index(path: Path):
    import faiss
    return faiss.deserialize_index(np.fromfile(str(path), dtype=np.uint8))


def _make_inner(kind: str, dim: int):
    import faiss
    metric = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        return faiss.IndexFlatIP(dim)
    if kind == "flat_fp16":
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, metric)
    if kind == "flat_sq8":
        return faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, metric)
    raise ValueError(f"unknown index_type: {kind} (use one of {KINDS})")


class VectorIndex:
    """索引 1 つ + 学習待ちのバッファ。ビルドと検索の両方で使う。"""

    def __init__(self, folder: Path, *, kind: str = "flat_sq8", dim: int = 1024,
                 train_size: int = 262144):
        import faiss
        self.folder = Path(folder)
        self.kind = kind
        self.dim = int(dim)
        self.train_size = int(train_size)
        self.index = faiss.IndexIDMap2(_make_inner(kind, self.dim))
        self.trained = bool(self.index.is_trained)
        self._buf_x: list[np.ndarray] = []
        self._buf_ids: list[np.ndarray] = []
        self._buf_n = 0

    # ── 読み書き ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, folder: Path) -> "VectorIndex":
        import faiss
        folder = Path(folder)
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.folder = folder
        obj.kind = meta.get("type", "flat")
        obj.dim = int(meta.get("dim", 1024))
        obj.train_size = int(meta.get("train_size", 262144))
        obj.index = _read_index(folder / INDEX_FILE)
        obj.trained = bool(obj.index.is_trained)
        obj._buf_x, obj._buf_ids, obj._buf_n = [], [], 0
        return obj

    @classmethod
    def exists(cls, folder: Path) -> bool:
        folder = Path(folder)
        return (folder / META_FILE).is_file() and (folder / INDEX_FILE).is_file()

    def save(self) -> None:
        """一時ファイルに書いてから置き換える (途中で落ちても前の索引が残る)。"""
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / (INDEX_FILE + ".tmp")
        _write_index(self.index, tmp)
        os.replace(tmp, self.folder / INDEX_FILE)
        meta = {"type": self.kind, "dim": self.dim, "train_size": self.train_size,
                "ntotal": int(self.index.ntotal)}
        tmpm = self.folder / (META_FILE + ".tmp")
        tmpm.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmpm, self.folder / META_FILE)

    # ── ビルド側 ─────────────────────────────────────────────────────────
    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) + self._buf_n

    def add(self, x: np.ndarray, ids: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if self.trained:
            if len(ids):
                self.index.add_with_ids(x, ids)
            return
        self._buf_x.append(x)
        self._buf_ids.append(ids)
        self._buf_n += len(ids)
        if self._buf_n >= self.train_size:
            self._train_and_flush()

    def remove(self, ids) -> int:
        import faiss
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
        if not len(ids):
            return 0
        removed = 0
        if self.trained:
            removed = int(self.index.remove_ids(faiss.IDSelectorBatch(ids)))
        if self._buf_n:                       # 学習前のバッファからも消す
            drop = set(ids.tolist())
            for i, (bx, bid) in enumerate(zip(self._buf_x, self._buf_ids)):
                keep = np.array([j not in drop for j in bid.tolist()], dtype=bool)
                removed += int((~keep).sum())
                self._buf_x[i], self._buf_ids[i] = bx[keep], bid[keep]
            self._buf_n = sum(len(b) for b in self._buf_ids)
        return removed

    def _train_and_flush(self) -> None:
        if not self._buf_n:
            return
        x = np.concatenate(self._buf_x, axis=0)
        ids = np.concatenate(self._buf_ids, axis=0)
        log.info("Training %s on %d vectors...", self.kind, len(ids))
        self.index.train(x)
        self.trained = True
        self.index.add_with_ids(x, ids)
        self._buf_x, self._buf_ids, self._buf_n = [], [], 0

    def finalize(self) -> None:
        """ビルドの最後に、学習待ちで残ったベクトルを学習して追加する。"""
        self._train_and_flush()

    # ── 検索側 ───────────────────────────────────────────────────────────
    def search(self, q: np.ndarray, k: int, selector=None):
        import faiss
        k = max(1, min(int(k), int(self.index.ntotal) or 1))
        if selector is not None:
            params = faiss.SearchParameters()
            params.sel = selector
            return self.index.search(q, k, params=params)
        return self.index.search(q, k)

    def reconstruct(self, chunk_id: int) -> np.ndarray:
        return self.index.reconstruct(int(chunk_id))


def make_id_selector(ids, keep_alive: list):
    """IDSelectorBatch。faiss がポインタだけ持つので numpy 配列を keep_alive に入れて生かしておく。"""
    try:
        import faiss
    except ImportError:
        return None
    arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    keep_alive.append(arr)
    try:
        return faiss.IDSelectorBatch(arr)
    except (TypeError, ValueError):
        try:
            return faiss.IDSelectorBatch(arr.size, faiss.swig_ptr(arr))
        except Exception as e:
            log.warning("IDSelectorBatch failed (%s) — falling back to post-filter", e)
            return None


def parse_kind(cfg_vdb: Optional[dict]) -> str:
    kind = str((cfg_vdb or {}).get("index_type") or "flat_sq8")
    if kind not in KINDS:
        raise ValueError(f"vector_db.index_type must be one of {KINDS}, got {kind!r}")
    return kind
