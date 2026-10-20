"""2 段検索: 密検索 (bge-m3 + FAISS) → リランク (bge-reranker-v2-m3) → 任意で MMR → top_k。

重いもの (モデル・索引) はコンストラクタで 1 回だけ読み、search() を何度も呼ぶ。
各段は無くても動く: リランカーが無ければ密検索の順、MMR が off ならリランクの順、
min_score が null なら足切りしない。

絞り込み (拡張子・パスの部分一致) は FAISS に渡す前に SQLite で ID を解決し、IDSelector で
密検索そのものを絞る。上位 N 件を取ってから捨てる方式だと、少数派の文書を絞り込んだときに
候補がほとんど残らないため。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import data_path, load_config
from .vector_index import VectorIndex, make_id_selector

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    chunk_id: int
    score: float            # 最終スコア (score_type で意味が変わる)
    file_id: str
    path: str
    ext: str
    chunk_index: int
    num_chunks: int
    text: str
    score_type: str = "cosine"      # "cosine" (密検索のみ) / "rerank" (sigmoid(logit), 0..1)
    dense_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(self, config_path: str | Path = "config.yaml", *, model=None, reranker=None,
                 load_reranker: bool = True):
        self.cfg, self.base_dir = load_config(config_path)
        rcfg = (self.cfg.get("retrieval") or {}).get("reranker") or {}
        self.reranker_batch = int(rcfg.get("batch_size", 32))
        self._lock = threading.Lock()
        if model is None:
            from .models import load_embedder
            model = load_embedder(self.cfg, self.base_dir)
        self.model = model
        self.normalize = bool((self.cfg.get("model") or {}).get("normalize_embeddings", True))
        if reranker is None and load_reranker:
            from .models import load_reranker as _load_rr
            reranker = _load_rr(self.cfg, self.base_dir)
        self.reranker = reranker
        self._load_index()

    # ── 索引 ─────────────────────────────────────────────────────────────
    def _load_index(self) -> None:
        vcfg = self.cfg.get("vector_db") or {}
        pcfg = self.cfg.get("processing") or {}
        self.index_dir = data_path(self.cfg, self.base_dir, vcfg.get("path", "vector_db"))
        self.state_db = data_path(self.cfg, self.base_dir, pcfg.get("state_db", "vector_db/state.sqlite"))
        if not VectorIndex.exists(self.index_dir) or not self.state_db.is_file():
            log.warning("Index not found at %s — run scripts/build_index.py first", self.index_dir)
            self.index, self.total = None, 0
            return
        self.index = VectorIndex.load(self.index_dir)
        self.total = int(self.index.ntotal)
        with self._conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if n != self.total:
            log.warning("Index ntotal (%d) != SQLite chunks (%d) — consider --rebuild", self.total, n)
        log.info("Index loaded: %d vectors (%s)", self.total, self.index.kind)

    def reload(self) -> Dict[str, int]:
        with self._lock:
            old = self.total
            self._load_index()
            return {"old_total": old, "new_total": self.total}

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.state_db))
        conn.row_factory = sqlite3.Row
        return conn

    # ── 検索 ─────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5, ext_filter: Optional[List[str]] = None,
               path_contains: Optional[str] = None) -> List[SearchHit]:
        if self.total == 0 or self.index is None:
            return []
        query = (query or "").strip()
        if not query:
            return []
        rcfg = self.cfg.get("retrieval") or {}
        candidates = int(rcfg.get("candidates", 200))
        min_score = rcfg.get("min_score")
        mmr_cfg = rcfg.get("mmr") or {}
        mmr_on = bool(mmr_cfg.get("enabled", False))
        mmr_lambda = float(mmr_cfg.get("lambda", 0.65))
        index = self.index                         # reload() で差し替わっても、この検索は同じ索引で終える

        ext_set = None
        if ext_filter:
            ext_set = {(e if e.startswith(".") else f".{e}").lower() for e in ext_filter}
        needle = path_contains.lower() if path_contains else None
        has_filter = bool(ext_set or needle)
        selector, allowed, n_allowed, keep_alive = None, None, 0, []
        if has_filter:
            ids = self._resolve_filter_ids(ext_set, needle)
            if not ids:
                return []
            allowed, n_allowed = set(ids), len(ids)
            selector = make_id_selector(ids, keep_alive)

        fetch_k = self._fetch_k(top_k, candidates, self.total, n_allowed, has_filter=has_filter,
                                will_postprocess=(self.reranker is not None or mmr_on))
        q = self.model.encode([query], normalize_embeddings=self.normalize,
                              show_progress_bar=False, convert_to_numpy=True)
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32))
        distances, ids = index.search(q, fetch_k, selector=selector)
        valid = [(int(i), float(d)) for i, d in zip(ids[0], distances[0]) if i >= 0]
        if not valid:
            return []

        with self._conn() as conn:
            marks = ",".join("?" * len(valid))
            rows = conn.execute(
                f"SELECT chunk_id, file_id, chunk_index, num_chunks, path, ext, text "
                f"FROM chunks WHERE chunk_id IN ({marks})", [v[0] for v in valid]).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        hits: List[SearchHit] = []
        for cid, dist in valid:
            r = by_id.get(cid)
            if r is None or (allowed is not None and cid not in allowed):
                continue
            hits.append(SearchHit(chunk_id=cid, score=dist, file_id=r["file_id"], path=r["path"],
                                  ext=r["ext"], chunk_index=r["chunk_index"], num_chunks=r["num_chunks"],
                                  text=r["text"], score_type="cosine", dense_score=dist))
        if not hits:
            return []
        if self.reranker is not None:
            self._rerank_inplace(query, hits)
        if mmr_on and len(hits) > top_k:
            hits = self._apply_mmr(index, hits, top_k, mmr_lambda)
        hits = self._apply_min_score(hits, min_score)
        return hits[:top_k]

    @staticmethod
    def _fetch_k(top_k: int, candidates: int, total: int, n_allowed: int,
                 has_filter: bool, will_postprocess: bool) -> int:
        """FAISS から取る候補数。後段 (リランク・MMR・絞り込み) で並べ替えるときは広く取る。"""
        if has_filter:
            k = min(n_allowed, max(candidates, top_k * 4))
        elif will_postprocess:
            k = min(total, max(candidates, top_k * 4))
        else:
            k = min(total, top_k)
        return max(int(k), 1)

    @staticmethod
    def _apply_min_score(hits: List[SearchHit], min_score) -> List[SearchHit]:
        if min_score is None:
            return hits
        try:
            ms = float(min_score)
        except (TypeError, ValueError):
            return hits
        return [h for h in hits if h.score >= ms]

    def _resolve_filter_ids(self, ext_set, needle) -> List[int]:
        clauses, params = [], []
        if ext_set:
            clauses.append("ext IN (" + ",".join("?" * len(ext_set)) + ")")
            params += sorted(ext_set)
        if needle:
            clauses.append("LOWER(path) LIKE ?")
            params.append(f"%{needle}%")
        with self._conn() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks WHERE " + " AND ".join(clauses), params)
            return [int(r[0]) for r in rows]

    def _rerank_inplace(self, query: str, hits: List[SearchHit]) -> None:
        """クロスエンコーダで採点し直して並べ替える。logit を sigmoid で 0..1 にする (閾値を解釈しやすく)。"""
        import torch
        pairs = [(query, h.text) for h in hits]
        common = dict(batch_size=self.reranker_batch, show_progress_bar=False, convert_to_numpy=True)
        # sentence-transformers は既定で sigmoid を掛けて返す。生の logit を受け取ってから
        # ここで 1 回だけ sigmoid にする (版によって引数名が違う)
        logits = None
        for kw in ("activation_fn", "activation_fct"):
            try:
                logits = self.reranker.predict(pairs, **common, **{kw: torch.nn.Identity()})
                break
            except TypeError:
                continue
            except Exception as e:
                log.warning("Reranker.predict failed (%s) — keeping dense order", e)
                return
        if logits is None:
            try:
                logits = self.reranker.predict(pairs, **common)
            except Exception as e:
                log.warning("Reranker.predict failed (%s) — keeping dense order", e)
                return
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-logits))
        for h, s in zip(hits, scores):
            h.score, h.score_type = float(s), "rerank"
        hits.sort(key=lambda h: h.score, reverse=True)

    @staticmethod
    def _apply_mmr(index: VectorIndex, hits: List[SearchHit], top_k: int, lambda_: float) -> List[SearchHit]:
        """貪欲 MMR。関連度は直前の段のスコア、多様性は保存済みベクトル同士のコサイン類似度。"""
        try:
            embs = np.stack([index.reconstruct(h.chunk_id) for h in hits])
        except Exception as e:
            log.warning("MMR: reconstruct failed (%s) — skipping MMR", e)
            return hits
        sim = embs @ embs.T
        rel = np.array([h.score for h in hits], dtype=np.float32)
        selected: List[int] = []
        remaining = list(range(len(hits)))
        while remaining and len(selected) < min(top_k, len(hits)):
            if not selected:
                best = remaining[int(np.argmax(rel[remaining]))]
            else:
                best, best_score = remaining[0], -1e18
                for i in remaining:
                    s = lambda_ * float(rel[i]) - (1.0 - lambda_) * float(np.max(sim[i, selected]))
                    if s > best_score:
                        best_score, best = s, i
            selected.append(best)
            remaining.remove(best)
        return [hits[i] for i in selected]

    def stats(self) -> Dict[str, Any]:
        return {"total_chunks": self.total, "index_type": self.index.kind if self.index else None,
                "reranker": self.reranker is not None}
