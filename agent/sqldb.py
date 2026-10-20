"""読み取り専用の SQLite と、テーブルのカタログ (エージェントの DB 道具)。

安全策は 3 重:
  1. 接続: URI の mode=ro + PRAGMA query_only (書き込みはエンジン側で失敗する)
  2. 文の検査 (clean_select): SELECT / WITH の 1 文だけ。書き込み語・危険な関数・機密列を断る
  3. 認可コールバック (set_authorizer): 読み出しと関数以外の操作、機密列の読み出しを断る
加えて、実行時間の上限 (progress handler で打ち切り) と、返す行数の上限がある。

カタログはテーブル・列・型・主キー/外部キー・インデックス・件数・頻出値・範囲を DB から読み、
手書きの説明 (descriptions.json) を重ねる。エージェントは find_tables → describe_table で
列を確かめてから SQL を書く。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CELL_MAX_CHARS = 200
REDACTED = "***"

# 値を絶対に外へ出さない列。語単位で見る (pass_no のような業務列は対象外)
_SENSITIVE_COL = re.compile(
    r"(^|_)(pwd|passwd|password|passphrase|secret|token|api_?key|apikey|credentials?|private_?key)($|_)", re.I)


def is_sensitive(column: str) -> bool:
    return bool(_SENSITIVE_COL.search(column or ""))


def redact_rows(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[List[Any]]:
    mask = [is_sensitive(c) for c in columns]
    return [[REDACTED if m and v not in (None, "") else v for m, v in zip(mask, r)] for r in rows]


class DBError(RuntimeError):
    """エージェントが読んで直せる文言の失敗。"""


# ── SQL の検査 ─────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"""
      (?P<line>--[^\n]*)
    | (?P<block>/\*.*?\*/)
    | (?P<lit>'(?:[^']|'')*')
    | (?P<ident>"(?:[^"]|"")*"|\[[^\]]*\]|`[^`]*`)
""", re.S | re.X)

_FORBIDDEN_WORDS = re.compile(
    r"\b(insert|update|delete|replace\s+into|drop|alter|create|attach|detach|pragma|vacuum|reindex|"
    r"analyze|into|begin|commit|rollback|savepoint|release)\b", re.I)
_FORBIDDEN_FUNCS = re.compile(r"\b(load_extension|readfile|writefile|edit|fts3_tokenizer|zipfile|sqlar_\w+)\s*\(", re.I)


def clean_select(sql: str) -> str:
    """SELECT / WITH の単文だけを通し、コメントと末尾の ; を落とした SQL を返す。"""
    if not (sql or "").strip():
        raise DBError("SQL が空です。")

    def drop_comments(m: "re.Match[str]") -> str:
        return " " if (m.group("line") or m.group("block")) else m.group(0)

    cleaned = _TOKEN_RE.sub(drop_comments, sql).strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    masked = _TOKEN_RE.sub(" _ ", cleaned)              # 文字列・識別子の中身は検査しない
    if "/*" in masked or "'" in masked or '"' in masked:
        raise DBError("引用符またはコメントの対応が取れていません。")
    if ";" in masked:
        raise DBError("複数の文は実行できません (SELECT 1 文だけ)。")
    if not re.match(r"\s*\(*\s*(select|with)\b", masked, re.I):
        raise DBError("SELECT または WITH で始まる読み出し専用のクエリだけ実行できます。")
    m = _FORBIDDEN_WORDS.search(masked)
    if m:
        raise DBError("読み出し専用のため使えない語が含まれています: %s" % m.group(1).upper())
    m = _FORBIDDEN_FUNCS.search(masked)
    if m:
        raise DBError("使えない関数が含まれています: %s" % m.group(1))
    no_lits = re.sub(r"'(?:[^']|'')*'", " ", cleaned)
    for ident in re.findall(r'"((?:[^"]|"")+)"|([A-Za-z_][\w$]*)', no_lits):
        name = ident[0] or ident[1]
        if is_sensitive(name):
            raise DBError("機密列 (パスワード等) は参照できません: %s" % name)
    return cleaned


def jsonable(v: Any, max_chars: int = CELL_MAX_CHARS) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        if isinstance(v, float):
            return round(v, 6)
        return v
    if isinstance(v, bytes):
        return "<%d bytes>" % len(v)
    s = str(v)
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# ── 読み取り専用の接続 ──────────────────────────────────────────────────────
class SQLiteReader:
    _OK_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   getattr(sqlite3, "SQLITE_RECURSIVE", 33)}

    def __init__(self, path: str | Path, *, timeout_sec: float = 20.0, max_rows: int = 50):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise DBError(f"DB がありません: {self.path} (scripts/make_sample_db.py で作る)")
        self.timeout_sec = float(timeout_sec)
        self.max_rows = int(max_rows)
        self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")
        self._deadline = 0.0
        self._authorize = False

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _authorizer(self, action, arg1, arg2, dbname, source):
        if not self._authorize:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and is_sensitive(arg2 or ""):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and _FORBIDDEN_FUNCS.search((arg2 or "") + "("):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in self._OK_ACTIONS else sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0

    def _run(self, sql: str, params: Sequence[Any] = (), *, guard: bool) -> Tuple[List[str], List[tuple], bool]:
        """guard=True ならエージェントの SQL として認可を効かせる。(列, 行, 打ち切ったか)"""
        self._deadline = time.monotonic() + self.timeout_sec
        self.conn.set_progress_handler(self._progress, 10000)
        self._authorize = guard
        self.conn.set_authorizer(self._authorizer)
        try:
            cur = self.conn.execute(sql, params)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1) if guard else cur.fetchall()
            truncated = guard and len(rows) > self.max_rows
            return cols, rows[:self.max_rows] if guard else rows, truncated
        except sqlite3.DatabaseError as e:          # OperationalError もここ
            msg = str(e)
            if "interrupted" in msg:
                raise DBError("実行時間の上限 (%.0f 秒) を超えたため打ち切りました。条件で絞るか集計してください。"
                              % self.timeout_sec) from e
            if "not authorized" in msg or "prohibited" in msg:
                raise DBError("許可されていない操作です (読み出し専用。機密列は参照できません): %s" % msg) from e
            raise DBError(msg) from e
        finally:
            self.conn.set_progress_handler(None, 0)
            self._authorize = False

    # 道具 -------------------------------------------------------------------
    def query(self, sql: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        cleaned = clean_select(sql)
        t0 = time.time()
        keep = self.max_rows
        if max_rows:
            self.max_rows = max(1, min(int(max_rows), 5000))
        try:
            cols, rows, truncated = self._run(cleaned, guard=True)
        finally:
            self.max_rows = keep
        rows = redact_rows(cols, rows)
        return {"columns": cols, "rows": [[jsonable(v) for v in r] for r in rows], "row_count": len(rows),
                "truncated": truncated, "elapsed_ms": int((time.time() - t0) * 1000)}

    def explain(self, sql: str) -> Dict[str, Any]:
        cleaned = clean_select(sql)
        _, rows, _ = self._run("EXPLAIN QUERY PLAN " + cleaned, guard=False)
        plan = "\n".join(str(r[-1]) for r in rows)
        scans = [ln for ln in plan.splitlines() if ln.strip().startswith("SCAN")]
        return {"plan": plan, "full_scans": scans,
                "note": "SCAN は全件走査 (大きい表では遅い)。SEARCH … USING INDEX はインデックスで絞れている。"}

    def sample_rows(self, table: str, n: int = 3) -> Dict[str, Any]:
        cols, _, _ = self._run(f"SELECT * FROM {_qi(table)} LIMIT 0", guard=False)
        sel = ", ".join(("NULL AS " + _qi(c)) if is_sensitive(c) else _qi(c) for c in cols)
        cols, rows, _ = self._run(f"SELECT {sel} FROM {_qi(table)} LIMIT ?", (max(1, min(10, n)),), guard=False)
        rows = [[REDACTED if is_sensitive(c) else jsonable(v) for c, v in zip(cols, r)] for r in rows]
        return {"columns": cols, "rows": rows}

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Tuple[List[str], List[tuple]]:
        """アプリ自身の決まった SQL 用 (エージェントの SQL には使わない)。"""
        cols, rows, _ = self._run(sql, params, guard=False)
        return cols, rows


# ── カタログ ───────────────────────────────────────────────────────────────
def dump_schema(reader: SQLiteReader, mcv: int = 5) -> List[Dict[str, Any]]:
    """テーブルごとの列・制約・インデックス・件数・頻出値/範囲。"""
    _, trows = reader.fetch_all("SELECT name, sql FROM sqlite_master WHERE type='table' "
                                "AND name NOT LIKE 'sqlite_%' ORDER BY name")
    out = []
    for name, ddl in trows:
        _, cols = reader.fetch_all(f"PRAGMA table_info({_qi(name)})")
        _, fks = reader.fetch_all(f"PRAGMA foreign_key_list({_qi(name)})")
        _, idx = reader.fetch_all("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? "
                                  "AND sql IS NOT NULL", (name,))
        _, cnt = reader.fetch_all(f"SELECT COUNT(*) FROM {_qi(name)}")
        n_rows = int(cnt[0][0])
        columns = []
        for _cid, cname, ctype, _notnull, _dflt, pk in cols:
            col: Dict[str, Any] = {"name": cname, "type": ctype or "", "pk": bool(pk)}
            if not is_sensitive(cname) and n_rows:
                q = _qi(cname)
                _, st = reader.fetch_all(f"SELECT COUNT(DISTINCT {q}), MIN({q}), MAX({q}) FROM {_qi(name)}")
                ndist, mn, mx = st[0]
                stats: Dict[str, Any] = {"distinct": ndist}
                if ndist and ndist <= 30:
                    _, mv = reader.fetch_all(f"SELECT {q}, COUNT(*) c FROM {_qi(name)} GROUP BY {q} "
                                             f"ORDER BY c DESC LIMIT ?", (mcv,))
                    stats["common_vals"] = [jsonable(v[0], 60) for v in mv]
                if mn is not None and not isinstance(mn, bytes):
                    stats["min"], stats["max"] = jsonable(mn, 40), jsonable(mx, 40)
                col["stats"] = stats
            columns.append(col)
        pk_cols = [c["name"] for c in columns if c["pk"]]
        out.append({"name": name, "est_rows": n_rows, "columns": columns,
                    "primary_key": pk_cols,
                    "foreign_keys": ["%s → %s(%s)" % (fk[3], fk[2], fk[4]) for fk in fks],
                    "indexes": [s for _, s in idx]})
    return out


class Catalog:
    def __init__(self, tables: List[Dict[str, Any]], descriptions: Optional[Dict[str, Any]] = None):
        self.tables = {t["name"]: t for t in tables}
        self.desc = (descriptions or {}).get("tables", {})

    @classmethod
    def build(cls, reader: SQLiteReader, descriptions_path: Optional[Path] = None) -> "Catalog":
        desc = {}
        if descriptions_path and Path(descriptions_path).is_file():
            desc = json.loads(Path(descriptions_path).read_text(encoding="utf-8-sig"))
        return cls(dump_schema(reader), desc)

    def __len__(self) -> int:
        return len(self.tables)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        n = str(name or "").strip().strip('"').split(".")[-1].strip('"')
        if n in self.tables:
            return self.tables[n]
        low = {k.lower(): v for k, v in self.tables.items()}
        return low.get(n.lower())

    def described(self, name: str) -> Tuple[str, Dict[str, str], str]:
        """(説明, 列の説明, 出どころ manual / "")"""
        d = self.desc.get(name) or {}
        text = (d.get("description") or "").strip()
        return text, dict(d.get("columns") or {}), "manual" if text else ""

    def find(self, keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """表名・列名・説明を横断した部分一致。空白区切りの語は加点 (多く当たるほど上位)。"""
        words = [w.lower() for w in (keyword or "").split() if w.strip()]
        if not words:
            return []
        scored = []
        for name, t in self.tables.items():
            desc, col_desc, _ = self.described(name)
            col_text = " ".join("%s %s" % (c["name"], col_desc.get(c["name"], "")) for c in t["columns"]).lower()
            score = 0
            for w in words:
                score += 5 * (w in name.lower()) + 3 * (w in desc.lower()) + 1 * (w in col_text)
            if score:
                scored.append((score, name, t, desc))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{"table": name, "est_rows": t["est_rows"], "description": desc,
                 "columns": [c["name"] for c in t["columns"]][:30]} for _, name, t, desc in scored[:limit]]

    def brief(self, t: Dict[str, Any], columns_like: str = "", max_columns: int = 40) -> Dict[str, Any]:
        """LLM に渡す 1 テーブル分の要約。列が多い表は columns_like で絞る。"""
        desc, col_desc, src = self.described(t["name"])
        cols = t["columns"]
        words = [w.lower() for w in (columns_like or "").split()]
        if words:
            cols = [c for c in cols if any(w in ("%s %s" % (c["name"], col_desc.get(c["name"], ""))).lower()
                                           for w in words)]
        omitted = max(0, len(cols) - max_columns)
        items = []
        for c in cols[:max_columns]:
            item: Dict[str, Any] = {"name": c["name"], "type": c["type"]}
            if col_desc.get(c["name"]):
                item["description"] = col_desc[c["name"]]
            if is_sensitive(c["name"]):
                item["note"] = "機密列 (参照不可)"
            st = c.get("stats") or {}
            if st.get("common_vals"):
                item["common_vals"] = st["common_vals"]
            if st.get("min") is not None:
                item["range"] = [st.get("min"), st.get("max")]
            items.append(item)
        out = {"table": t["name"], "est_rows": t["est_rows"], "description": desc,
               "description_source": src, "column_count": len(t["columns"]), "columns": items,
               "primary_key": t["primary_key"], "foreign_keys": t["foreign_keys"],
               "indexes": [re.sub(r"^CREATE (UNIQUE )?INDEX \S+ ON ", "", s) for s in t["indexes"]]}
        if omitted:
            out["columns_omitted"] = omitted
            out["hint"] = "列が多いため %d 列を省略。columns_like で絞り込める。" % omitted
        return out

    def overview(self, max_chars: int = 3200) -> str:
        lines = []
        for name, t in sorted(self.tables.items()):
            desc, _, _ = self.described(name)
            lines.append("- %s (%s 行)%s" % (name, "{:,}".format(t["est_rows"]), ": " + desc[:90] if desc else ""))
        return "\n".join(lines)[:max_chars]


TOOL_SPECS: List[Dict[str, Any]] = [
    {"name": "find_tables",
     "description": "キーワードでテーブルを探す (テーブル名・列名・説明の部分一致)。DB には触らない。",
     "args": {"keyword": "str 検索語 (空白区切りで複数可)", "limit": "int 既定 15"}},
    {"name": "describe_table",
     "description": "テーブルの列・型・説明・頻出値・値の範囲・主キー/外部キー・インデックスを返す。",
     "args": {"table": "str テーブル名", "columns_like": "str 列名・説明の部分一致 (空白区切りで OR)。省略可"}},
    {"name": "sample_rows",
     "description": "テーブルの先頭数行を返す (値の形式を確かめる用)。",
     "args": {"table": "str", "n": "int 1〜10 (既定 3)"}},
    {"name": "explain",
     "description": "SELECT の実行計画 (SQLite の EXPLAIN QUERY PLAN) を返す。実行はしない。重そうなら先に使う。",
     "args": {"sql": "str SELECT / WITH の 1 文"}},
    {"name": "query",
     "description": "読み出し専用の SELECT / WITH を 1 文実行し、先頭の行だけ返す。集計は SQL 側で行うこと。",
     "args": {"sql": "str SELECT / WITH の 1 文", "max_rows": "int 既定 50"}},
]
TOOL_NAMES = tuple(s["name"] for s in TOOL_SPECS)


def run_tool(reader: SQLiteReader, catalog: Catalog, name: str, args: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """道具を 1 回実行して dict を返す。失敗も {"error": …} で返す (エージェントが読んで直せるように)。"""
    args = args or {}
    try:
        if name == "find_tables":
            return {"tables": catalog.find(str(args.get("keyword", "")), int(args.get("limit") or 15))}
        if name in ("describe_table", "sample_rows"):
            t = catalog.get(str(args.get("table", "")))
            if t is None:
                return {"error": "テーブル '%s' はありません。find_tables で探してください (候補: %s)。"
                                 % (args.get("table"), ", ".join(sorted(catalog.tables)))}
            if name == "describe_table":
                return catalog.brief(t, str(args.get("columns_like") or ""))
            return dict({"table": t["name"]}, **reader.sample_rows(t["name"], int(args.get("n") or 3)))
        if name == "explain":
            return reader.explain(str(args.get("sql", "")))
        if name == "query":
            return reader.query(str(args.get("sql", "")), int(args.get("max_rows") or 0) or None)
        return {"error": "未知の道具です: %s (使えるのは %s)" % (name, ", ".join(TOOL_NAMES))}
    except DBError as e:
        return {"error": str(e)}
    except (TypeError, ValueError) as e:
        return {"error": "引数が不正です: %s" % e}
