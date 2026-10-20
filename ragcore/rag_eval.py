"""検索の精度を測る: 正解つきの質問集 (golden) を自動で作り、採点して記録する。

なぜ質問集を「合成」するのか
---------------------------
人手で正解ラベルを書いた質問集は、一度作られたきり保守されない。代わりにコーパスからチャンクを
1 つ抜き出し、LLM に「その抜粋だけで答えられる質問」を書かせる。正解のチャンクは作り方から
決まっているので、人手のラベル付けが要らない (運用者は一度ざっと見て、変な質問を消すだけ)。

なぜラベルが chunk_id ではなく (path, chunk_index) なのか
--------------------------------------------------------
chunk_id は AUTOINCREMENT なので、作り直し (--rebuild) で番号が振り直される。番号をラベルにすると
質問集は正しそうに見えたまま点数だけが崩れる。(path, chunk_index) は文書そのものから決まるので、
作り直し・再埋め込み・索引の種類の変更をまたいで有効。

なぜ 2 種類の再現率を取るのか
------------------------------
* CHUNK (chunk-exact) — 正解のチャンクそのものが上位 k 件にあるか。厳しい: 隣のチャンクが同じくらい
  よく答えていても外れになる。
* FILE (file-level)  — 正解の文書のどれかのチャンクが上位 k 件にあるか。「利用者が必要なものに
  たどり着けたか」に近く、設定を比べるときに見る主指標。
リランカーを強くすると隣のチャンクを上げることがあり、CHUNK だけ見るとそれを「悪化」と読み違える。
"""
from __future__ import annotations

import csv
import json
import random as _random
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_eval_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at         TEXT NOT NULL,
    run_date       TEXT NOT NULL,
    source         TEXT NOT NULL,      -- 'manual' | 'scheduled'
    label          TEXT,               -- 何を試した run か
    golden         TEXT,               -- 質問集のファイル名
    n_questions    INTEGER NOT NULL,
    top_k          INTEGER NOT NULL,
    recall_1       REAL,               -- CHUNK: 1 位
    recall_5       REAL,               -- CHUNK: 上位 5
    recall_k       REAL,               -- CHUNK: 上位 top_k
    mrr            REAL,               -- CHUNK: 平均逆順位
    file_recall_1  REAL,               -- FILE: 1 位
    file_recall_k  REAL,               -- FILE: 上位 top_k (主指標)
    miss           INTEGER,            -- 正解の文書が 1 件も出なかった質問の数
    latency_ms     REAL,
    chunks_total   INTEGER,
    duration_sec   REAL,
    note           TEXT
);
CREATE INDEX IF NOT EXISTS idx_rag_eval_date ON rag_eval_runs(run_date DESC);
"""

CSV_COLUMNS = ["run_at", "run_date", "source", "label", "golden", "n_questions", "top_k",
               "recall_1", "recall_5", "recall_k", "mrr", "file_recall_1", "file_recall_k",
               "miss", "latency_ms", "chunks_total", "duration_sec", "note"]

#: 1 チャンク → 1 質問 にするプロンプト。質問集の質はこの文面で決まる。
QUESTION_PROMPT = """以下は文書の一部です。この抜粋だけを根拠に答えられる、具体的な日本語の質問を1つだけ作ってください。

条件:
- 抜粋に出てくる固有名詞・型番・数値・工程名を必ず含め、検索で該当文書に辿り着ける質問にする。
- 「この文書」「上記」「本資料」などの指示語は使わない（質問だけを読んで意味が通ること）。
- 答えではなく質問を書く。1文にする。
- 質問文だけを出力する。前置き・番号・引用符・記号は付けない。

--- 抜粋 ---
{text}
--- 抜粋ここまで ---"""


class GoldenItem(dict):
    """採点される質問 1 つ。q / path / chunk_index (ラベル) / ext / chunk_id (診断用)。"""

    @property
    def key(self) -> tuple:
        return (str(self.get("path") or ""), int(self.get("chunk_index") or 0))


# ── 質問集の読み書き (JSONL: テキストエディタで 1 行消せるように) ─────────────
def save_golden(path: Union[str, Path], items: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(dict(it), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_golden(path: Union[str, Path]) -> list:
    """空行・# のコメント・壊れた行は飛ばす (手で編集する前提)。"""
    out: list = []
    p = Path(path)
    if not p.is_file():
        return out
    with p.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("q") and obj.get("path"):
                out.append(GoldenItem(obj))
    return out


# ── コーパスからの抽出 ──────────────────────────────────────────────────────
def _ro_conn(path: Union[str, Path]) -> sqlite3.Connection:
    """state.sqlite を読み取り専用で開く (評価が索引を壊したりロックしたりしないように)。"""
    p = Path(path).resolve()
    if not p.is_file():
        raise sqlite3.OperationalError(f"state DB not found: {p}")
    conn = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def sample_chunks(state_db: Union[str, Path], n: int, *, min_chars: int = 200, max_chars: int = 2000,
                  one_per_file: bool = True, rng=None) -> list:
    """ランダムに n チャンク。ORDER BY RANDOM() は数千万行を全走査するので、ランダムな chunk_id を引き、
    その番号以上の最初の行を取る (主キーのシーク)。one_per_file で 1 文書 1 問にする。"""
    rnd = rng or _random.Random()
    conn = _ro_conn(state_db)
    try:
        row = conn.execute("SELECT MAX(chunk_id) AS m FROM chunks").fetchone()
        top = int((row["m"] if row else 0) or 0)
        if top <= 0:
            return []
        out, seen_files, seen_ids = [], set(), set()
        for _ in range(max(200, n * 40)):
            if len(out) >= n:
                break
            r = conn.execute("SELECT chunk_id, path, ext, chunk_index, text FROM chunks "
                             "WHERE chunk_id >= ? ORDER BY chunk_id LIMIT 1",
                             (rnd.randint(1, top),)).fetchone()
            if r is None:
                continue
            cid = int(r["chunk_id"])
            if cid in seen_ids:
                continue
            text = r["text"] or ""
            if not (min_chars <= len(text) <= max_chars):
                continue
            path = r["path"] or ""
            if one_per_file and path in seen_files:
                continue
            seen_ids.add(cid)
            seen_files.add(path)
            out.append({"chunk_id": cid, "path": path, "ext": r["ext"] or "",
                        "chunk_index": int(r["chunk_index"] or 0), "text": text})
        return out
    finally:
        conn.close()


def corpus_size(state_db: Union[str, Path]) -> Optional[int]:
    try:
        conn = _ro_conn(state_db)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT SUM(num_chunks) AS n FROM files").fetchone()
        return int((row["n"] if row else 0) or 0)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# ── 生成結果から質問を取り出す ──────────────────────────────────────────────
_FENCE_RE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_LEAD_RE = re.compile(r"^\s*(?:[-*・>]|\d+[.)、]|Q[.:：]?|質問[:：]?)\s*", re.IGNORECASE)


def extract_question(text: str) -> str:
    """フェンスを外し、？/? で終わる最後の行 (無ければ最後の行) を取り、箇条書きの記号や引用符を外す。"""
    s = str(text or "")
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    lines = [ln.strip() for ln in s.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    q = next((ln for ln in reversed(lines) if ln.endswith("？") or ln.endswith("?")), lines[-1])
    q = _LEAD_RE.sub("", q).strip().strip('"').strip("'").strip("「」")
    return q.strip()


# ── 採点 ───────────────────────────────────────────────────────────────────
def _hit_field(hit, name: str):
    return hit.get(name) if isinstance(hit, dict) else getattr(hit, name, None)


def rank_of(hits: Sequence, item: dict) -> Optional[int]:
    """正解チャンクそのものの順位 (1 始まり)。無ければ None。"""
    want = (str(item.get("path") or ""), int(item.get("chunk_index") or 0))
    for i, h in enumerate(hits, 1):
        if (str(_hit_field(h, "path") or ""), int(_hit_field(h, "chunk_index") or 0)) == want:
            return i
    return None


def file_rank_of(hits: Sequence, item: dict) -> Optional[int]:
    """正解チャンクと同じ文書の、最初のヒットの順位。大文字小文字は区別しない。"""
    want = str(item.get("path") or "").lower()
    for i, h in enumerate(hits, 1):
        if str(_hit_field(h, "path") or "").lower() == want:
            return i
    return None


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(ranks: Sequence[Optional[int]], file_ranks: Sequence[Optional[int]], *, top_k: int,
              latencies_ms: Sequence[float] = ()) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n_questions": 0, "top_k": int(top_k), "recall_1": None, "recall_5": None,
                "recall_k": None, "mrr": None, "file_recall_1": None, "file_recall_k": None,
                "miss": 0, "latency_ms": _mean(latencies_ms)}

    def frac(rs, k):
        return round(sum(1 for r in rs if r is not None and r <= k) / n, 4)

    return {
        "n_questions": n, "top_k": int(top_k),
        "recall_1": frac(ranks, 1), "recall_5": frac(ranks, 5), "recall_k": frac(ranks, int(top_k)),
        "mrr": round(sum((1.0 / r) for r in ranks if r) / n, 4),
        "file_recall_1": frac(file_ranks, 1), "file_recall_k": frac(file_ranks, int(top_k)),
        "miss": sum(1 for r in file_ranks if r is None),
        "latency_ms": _mean(latencies_ms),
    }


def score(golden: Sequence[dict], search: Callable[[str, int], Sequence], *, top_k: int = 10,
          on_item: Optional[Callable] = None) -> dict:
    """質問を 1 つずつ search(q, top_k) に通して採点する。1 問の例外は「外れ」として記録し、
    残りの測定を捨てない。search は差し込み式 (本物の Retriever でもテスト用の偽物でもよい)。"""
    ranks, file_ranks, lat, details = [], [], [], []
    for item in golden:
        t0 = time.time()
        try:
            hits = search(item["q"], top_k) or []
            err = None
        except Exception as exc:  # noqa: BLE001
            hits, err = [], f"{type(exc).__name__}: {exc}"
        lat.append((time.time() - t0) * 1000.0)
        r, fr = rank_of(hits, item), file_rank_of(hits, item)
        ranks.append(r)
        file_ranks.append(fr)
        rec = {"q": item.get("q"), "path": item.get("path"), "chunk_index": item.get("chunk_index"),
               "rank": r, "file_rank": fr, "error": err}
        details.append(rec)
        if on_item:
            on_item(rec)
    out = summarize(ranks, file_ranks, top_k=top_k, latencies_ms=lat)
    out["details"] = details
    return out


# ── 記録 (SQLite + Excel 用 CSV) ───────────────────────────────────────────
def _conn(db: Union[str, Path]) -> sqlite3.Connection:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def record_run(db: Union[str, Path], row: dict, *, csv_path: Optional[Union[str, Path]] = None) -> int:
    conn = _conn(db)
    try:
        with conn:
            conn.executescript(SCHEMA)
            cols = CSV_COLUMNS
            cur = conn.execute(f"INSERT INTO rag_eval_runs({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                               tuple(row.get(k) for k in cols))
            rid = int(cur.lastrowid)
    finally:
        conn.close()
    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with p.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return rid


def list_runs(db: Union[str, Path], *, limit: int = 30) -> list:
    if not Path(db).is_file():
        return []
    conn = _conn(db)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM rag_eval_runs ORDER BY id DESC LIMIT ?",
                                              (int(limit),)).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def now_row(**kw) -> dict:
    now = datetime.now()
    row = {"run_at": now.isoformat(timespec="seconds"), "run_date": now.strftime("%Y-%m-%d")}
    row.update(kw)
    return row
