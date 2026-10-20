"""索引のビルド (差分更新)。

    ファイルを列挙 → 変更の判定 (mtime/size → 内容の SHA-1) → 抽出 → チャンク分割
    → まとめて埋め込み → state.sqlite に本文、FAISS にベクトル (ID = chunk_id)

変わったファイルは古いチャンクを消して入れ直す。消えたファイルのチャンクも消す。
state.sqlite のスキーマ (files / chunks) は検索・評価 (rag_eval) が読むので変えないこと。
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .chunking import chunk_by_tokens
from .config import data_path, resolve_path
from .extract import extract
from .vector_index import VectorIndex, parse_kind

log = logging.getLogger(__name__)

DEFAULT_EXTS = [".txt", ".md", ".csv", ".json", ".html", ".htm", ".log", ".py",
                ".pdf", ".docx", ".xlsx", ".pptx", ".ipynb"]
DEFAULT_EXCLUDE = [".git", "__pycache__", ".venv", "venv", "node_modules", ".ipynb_checkpoints"]


def open_state_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            num_chunks INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            num_chunks INTEGER NOT NULL,
            path TEXT NOT NULL,
            ext TEXT NOT NULL,
            text TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")
    conn.commit()
    return conn


def file_id_for(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()


def hash_file_content(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


@dataclass
class SourceFile:
    abs_path: Path
    rel_path: str          # 索引上の名前 ("<フォルダ名>/<相対パス>"、区切りは /)
    ext: str
    mtime: float
    size: int


def iter_files(folders: List[Path], exts: List[str], exclude_dirs: List[str],
               max_mb: float) -> Iterator[SourceFile]:
    want = {e.lower() if e.startswith(".") else "." + e.lower() for e in exts}
    skip = {d.lower() for d in exclude_dirs}
    for root in folders:
        if not root.is_dir():
            log.warning("target folder not found: %s", root)
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(part.lower() in skip for part in p.relative_to(root).parts[:-1]):
                continue
            ext = p.suffix.lower()
            if ext not in want:
                continue
            st = p.stat()
            if st.st_size > max_mb * 1024 * 1024:
                log.info("skip (too large): %s", p)
                continue
            rel = root.name + "/" + p.relative_to(root).as_posix()
            yield SourceFile(p, rel, ext, st.st_mtime, st.st_size)


class IndexBuilder:
    def __init__(self, cfg: Dict[str, Any], base_dir: Path, *, model=None):
        self.cfg = cfg
        self.base_dir = base_dir
        vcfg = cfg.get("vector_db") or {}
        pcfg = cfg.get("processing") or {}
        self.index_dir = data_path(cfg, base_dir, vcfg.get("path", "vector_db"))
        self.state_db = data_path(cfg, base_dir, pcfg.get("state_db", "vector_db/state.sqlite"))
        self.kind = parse_kind(vcfg)
        self.train_size = int(vcfg.get("train_size", 262144))
        self.embed_batch_chunks = int(pcfg.get("embed_batch_chunks", 2048))
        ccfg = cfg.get("chunking") or {}
        self.chunk_size = int(ccfg.get("chunk_size", 512))
        self.chunk_overlap = int(ccfg.get("chunk_overlap", 64))
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from .models import load_embedder
            self._model = load_embedder(self.cfg, self.base_dir)
        return self._model

    def _folders(self) -> List[Path]:
        t = self.cfg.get("target") or {}
        return [resolve_path(self.base_dir, f) for f in (t.get("folders") or [])]

    def rebuild_reset(self) -> None:
        """--rebuild: 索引と状態 DB を消して最初から作る。"""
        if self.index_dir.is_dir():
            shutil.rmtree(self.index_dir)
        if self.state_db.exists():
            self.state_db.unlink()

    def build(self, *, limit: int = 0, progress: bool = True) -> Dict[str, Any]:
        t_start = time.time()
        tcfg = self.cfg.get("target") or {}
        files = list(iter_files(self._folders(), tcfg.get("include_extensions") or DEFAULT_EXTS,
                                tcfg.get("exclude_dirs") or DEFAULT_EXCLUDE,
                                float(tcfg.get("max_file_size_mb", 200))))
        if limit > 0:
            files = files[:limit]
        conn = open_state_db(self.state_db)
        dim = int(self.model.get_sentence_embedding_dimension())
        if VectorIndex.exists(self.index_dir):
            index = VectorIndex.load(self.index_dir)
            if index.kind != self.kind or index.dim != dim:
                raise RuntimeError(
                    f"existing index is {index.kind}/{index.dim}d but config wants {self.kind}/{dim}d "
                    "— run with --rebuild")
        else:
            index = VectorIndex(self.index_dir, kind=self.kind, dim=dim, train_size=self.train_size)

        known = {r[0]: r for r in conn.execute(
            "SELECT path, file_id, content_hash, mtime, size FROM files")}
        seen: set = set()
        counters = {"files": len(files), "processed": 0, "unchanged": 0, "empty": 0,
                    "failed": 0, "removed_files": 0, "chunks": 0}
        pending_text: List[str] = []
        pending_meta: List[Dict[str, Any]] = []
        pending_files: List[tuple] = []
        pending_remove_fids: set = set()

        def old_chunk_ids(fid: str) -> List[int]:
            return [r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE file_id = ?", (fid,))]

        def flush(force: bool = False) -> None:
            if not force and len(pending_text) < self.embed_batch_chunks:
                return
            if not (pending_text or pending_files or pending_remove_fids):
                return
            embs = None
            if pending_text:
                t0 = time.time()
                embs = self.model.encode(
                    pending_text, batch_size=int((self.cfg.get("model") or {}).get("batch_size", 32)),
                    normalize_embeddings=bool((self.cfg.get("model") or {}).get("normalize_embeddings", True)),
                    show_progress_bar=False, convert_to_numpy=True)
                embs = np.ascontiguousarray(np.asarray(embs, dtype=np.float32))
                dt = time.time() - t0
                log.info("Embedded %d chunks in %.1fs (%.0f chunks/s)",
                         len(pending_text), dt, len(pending_text) / max(dt, 1e-9))
            cur = conn.cursor()
            try:
                remove_ids: List[int] = []
                for fid in pending_remove_fids:
                    remove_ids += old_chunk_ids(fid)
                    cur.execute("DELETE FROM chunks WHERE file_id = ?", (fid,))
                ids: List[int] = []
                for meta, text in zip(pending_meta, pending_text):
                    cur.execute("INSERT INTO chunks (file_id, chunk_index, num_chunks, path, ext, text) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (meta["file_id"], meta["chunk_index"], meta["num_chunks"],
                                 meta["path"], meta["ext"], text))
                    ids.append(int(cur.lastrowid))
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for row in pending_files:
                    cur.execute("INSERT OR REPLACE INTO files (path, file_id, content_hash, mtime, size, "
                                "num_chunks, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (*row, now))
                if remove_ids:
                    index.remove(remove_ids)
                if embs is not None:
                    index.add(embs, np.asarray(ids, dtype=np.int64))
                index.save()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            counters["chunks"] += len(pending_text)
            pending_text.clear()
            pending_meta.clear()
            pending_files.clear()
            pending_remove_fids.clear()

        tokenizer = self.model.tokenizer
        it = files
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(files, desc="index", unit="file")
            except ImportError:
                pass
        for sf in it:
            seen.add(sf.rel_path)
            fid = file_id_for(sf.rel_path)
            prev = known.get(sf.rel_path)
            if prev and abs(prev[3] - sf.mtime) < 1e-6 and prev[4] == sf.size:
                counters["unchanged"] += 1
                continue
            content_hash = hash_file_content(sf.abs_path)
            if prev and prev[2] == content_hash:            # 中身は同じ: 状態だけ更新
                conn.execute("UPDATE files SET mtime = ?, size = ? WHERE path = ?",
                             (sf.mtime, sf.size, sf.rel_path))
                conn.commit()
                counters["unchanged"] += 1
                continue
            text = extract(sf.abs_path)
            if text is None:
                counters["failed"] += 1
                continue
            chunks = chunk_by_tokens(text, tokenizer, self.chunk_size, self.chunk_overlap)
            if prev:
                pending_remove_fids.add(fid)
            if not chunks:
                counters["empty"] += 1
            for i, c in enumerate(chunks):
                pending_text.append(c)
                pending_meta.append({"file_id": fid, "chunk_index": i, "num_chunks": len(chunks),
                                     "path": sf.rel_path, "ext": sf.ext})
            pending_files.append((sf.rel_path, fid, content_hash, sf.mtime, sf.size, len(chunks)))
            counters["processed"] += 1
            flush()

        # 消えたファイル (同じ対象フォルダの範囲だけ。limit 指定時は消さない)
        if limit <= 0:
            gone = [p for p in known if p not in seen]
            for p in gone:
                pending_remove_fids.add(known[p][1])
                conn.execute("DELETE FROM files WHERE path = ?", (p,))
            counters["removed_files"] = len(gone)
        flush(force=True)
        index.finalize()
        index.save()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        if total != index.ntotal:
            log.warning("index ntotal (%d) != SQLite chunks (%d) — consider --rebuild", index.ntotal, total)
        counters.update({"total_chunks": int(total), "index_type": self.kind,
                         "seconds": round(time.time() - t_start, 1)})
        return counters
