"""文書検索 (RAG) と読み取り専用 SQL を道具として使う ReAct エージェント。

LLM に function calling を要求しない (小さいローカルモデルでも動くように)。道具の一覧と返答の
形式をプロンプトに書き、LLM が返す JSON を読んで道具を実行する (1 手 = LLM 1 回 + 道具 1 回)。

    質問 → [LLM: {"action": "find_tables", "args": …}] → 道具を実行 → 観測を LLM に返す
         → … (最大 max_steps 手) …
         → [LLM: {"action": "answer"}] → 調査の記録だけを根拠に最終回答をストリームで書く

「原因」「なぜ」などを含む質問は原因調査として、調べ方 (群の比較で差を確かめる) と手数
(cause_max_steps) を足し、最終回答で原因の候補を確度順の表にする。確度は 高/中/低/仮説 の語で、
確率の数値は書かせない (計算した値ではないため)。

道具: search_docs (文書検索) / find_tables / describe_table / sample_rows / explain / query (SQL) /
      make_chart (グラフ)

LLM の失敗に備えた仕掛け:
  - JSON の形式エラーは 2 回まで言い直させ、3 回目で打ち切ってここまでの記録で答える
  - 同じ呼び出しの繰り返しを断り、2 回繰り返したら打ち切る
  - 観測は切り詰め、文脈が溢れそうなら古い観測から縮める
  - 成功した手順が 1 つも無ければ、LLM に書かせず定型文で答える (数値を作らせない)
  - 回答の数値を調査の記録と突き合わせ、記録に無い数値を警告する (unsupported_numbers)

イベント (Web アプリが NDJSON にして画面へ流す):
    {"type": "agent", tools, catalog_tables, max_steps, cause}
    {"type": "step", n, thought, tool, args}
    {"type": "observation", n, tool, ok, summary, detail, elapsed_ms}
    {"type": "chart", n, title, src, csv}
    {"type": "warn", message}
    {"type": "sources", hits}
    {"type": "content" | "thinking", text}
    {"type": "error", message}
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import date
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ragcore.llm import LLMClient, LLMError, extract_json, strip_think

from . import charts
from .sqldb import TOOL_NAMES as DB_TOOL_NAMES
from .sqldb import TOOL_SPECS as DB_TOOL_SPECS
from .sqldb import Catalog, DBError, SQLiteReader, run_tool

SEARCH_SPEC = {
    "name": "search_docs",
    "description": "文書 (作業標準・トラブル事例・検査基準・技術資料など) を検索する。結果の [番号] は回答で引用できる。",
    "args": {"query": "str 検索文 (日本語の自然文でよい)", "top_k": "int 既定 5",
             "path_contains": "str ファイルパスの部分一致で絞る (省略可)"},
}

CHART_SPEC = {
    "name": "make_chart",
    "description": "グラフを作って画面に表示する。ユーザーがグラフ・図・推移を求めたとき、または比較や傾向を"
                   "見せると分かりやすいときに使う。データは sql (SELECT 1 文) か、前の query の手順番号 from_step。"
                   "集計は SQL で済ませ、グラフに使う列だけを返すこと。y は数値の列。棒は項目 40 個・系列 8 個まで。"
                   "推移 (時系列) は type=line で x に日付の列。点が多いときは date(日時列) などで日ごとに集計してから描く。"
                   "装置別などは series で分ける。",
    "args": {"type": "str bar / barh (横棒) / line (推移) / scatter / hist",
             "title": "str グラフの題", "x": "str 横軸の列名 (hist は不要)",
             "y": "str 縦軸の列名 (複数なら list)", "series": "str 系列を分ける列 (省略可)",
             "sql": "str SELECT 1 文 (from_step と どちらか)", "from_step": "int 前の query の手順番号"},
}

ANSWER_ACTIONS = ("answer", "final", "final_answer", "finish")
REMIND = "(返答は指示の形式の JSON オブジェクト 1 つだけ。説明文や思考は書かない)"
PERSONA = "あなたは工場の品質データ (データベース) と技術文書を調べるアシスタントです。"

EXAMPLE = "\n".join([
    "## 手本 (形式の例。内容は質問に合わせる)",
    "質問: 9 月に POL-03 で研磨したロットの LPD の平均は？",
    '→ {"thought": "検査結果と工程履歴の表を探す", "action": "find_tables", "args": {"keyword": "LPD 装置 工程"}}',
    '→ {"thought": "列を確かめる", "action": "describe_table", "args": {"table": "process_history"}}',
    '→ {"thought": "9 月分を SQL で平均する", "action": "query", "args": {"sql": '
    '"SELECT avg(i.lpd_45nm) AS avg_lpd FROM inspection_results i JOIN process_history h ON h.lot_id = i.lot_id '
    'WHERE h.equipment_id = \'POL-03\' AND h.start_time >= \'2026-09-01\' AND h.start_time < \'2026-10-01\'"}}',
    '→ {"thought": "平均が分かった", "action": "answer"}',
])


def _tool_lines(specs: List[Dict[str, Any]]) -> str:
    out = []
    for s in specs:
        args = ", ".join('"%s": %s' % (k, v) for k, v in s["args"].items())
        out.append("- %s: %s\n  args: {%s}" % (s["name"], s["description"], args))
    return "\n".join(out)


def step_system_prompt(specs: List[Dict[str, Any]], *, db_on: bool, n_tables: int, overview: str,
                       max_steps: int, cause: bool = False, today: Optional[str] = None) -> str:
    parts = [
        PERSONA + "質問に答えるため、道具を 1 回ずつ使って調べ、十分に分かったら回答に移ります。"
        "使える手数は最大 %d 手です。今日は %s です。" % (max_steps, today or date.today().isoformat()),
        "",
        "## 道具",
        _tool_lines(specs),
    ]
    if db_on:
        parts += [
            "",
            "## DB (SQLite / 読み出し専用 / テーブル %d 個) の使い方" % n_tables,
            "- 「どのロット」「何件」「平均」「最近」など今の実データを問う質問は、文書だけで済ませず DB でも確かめる。"
            "文書 (search_docs) は過去の事例・対策・規程の根拠として使う。両方を問う質問は両方を調べる。",
            "- まず find_tables で候補を探し、describe_table で列・説明・値の範囲を確かめてから SQL を書く。"
            "列名・表名を推測で書かない。",
            "- 日時は 'YYYY-MM-DD HH:MM' の文字列。期間は文字列の比較 (>= '2026-09-01' AND < '2026-10-01') で絞る。"
            "月ごとは strftime('%Y-%m', 列)、日ごとは date(列) で集計する。SQLite に date_trunc・ILIKE は無い。",
            "- 集計 (count / avg / group by) は SQL の中で行い、生の行を大量に取らない。LIMIT を付ける。",
            "- 道具がエラーを返したら内容を読んで直す。同じ呼び出しを繰り返さない。",
            "",
            "## テーブルの一覧",
            overview or "(カタログ未作成)",
            "",
            EXAMPLE,
        ]
    if cause:
        parts += ["", *cause_step_rules(db_on)]
    parts += [
        "",
        "## 返答の形式 (厳守)",
        "毎回、次のどちらかの JSON オブジェクト 1 つだけを返す。前後に文章やコードフェンスを付けない。",
        '道具を使う: {"thought": "次に何をなぜ調べるか (1 文)", "action": "道具名", "args": {…}}',
        '回答に移る: {"thought": "分かったこと (1 文)", "action": "answer"}',
        "挨拶や一般知識だけで答えられる質問は、道具を使わずにすぐ answer でよい。",
    ]
    return "\n".join(parts)


FINAL_SYSTEM = (
    PERSONA + "下の「調査の記録」のうち「成功」した手順の結果だけを根拠に、日本語で簡潔に答えてください。\n"
    "- 記録に無い数値・ロット名・表名を絶対に作らない。数値は記録の値をそのまま使う。"
    "計算した値は、元の数値と式を併記する (例: 22,655 ÷ 80,363 ≈ 28.2 %)。\n"
    "- 文書の内容を使った箇所には、記録の cite の値 ([1] [2] のように角括弧に数字だけ) を付ける。\n"
    "- DB の値を使った場合は、使ったテーブル名を本文か末尾に書く。\n"
    "- make_chart のグラフは画面に表示済み。図の要点を文章で述べ、同じ数値の長い表は繰り返さない。\n"
    "- 複数の項目の比較 (ロット別・装置別の件数など) は Markdown の表 (| 列 | 列 |) で示す。\n"
    "- 調査の途中の失敗・再試行・道具の使い方には触れない (過程は画面に別に表示される)。\n"
    "- 記録から分からないことは、分からないと書き、次に何を調べればよいかを一言添える。\n"
)

CAUSE_RE = re.compile(r"原因|要因|真因|起因|なぜ|何故|どうして|理由|寄与")


def is_cause_question(question: str) -> bool:
    return bool(CAUSE_RE.search(question or ""))


def cause_step_rules(db_on: bool) -> List[str]:
    out = ["## 原因・要因を問う質問の調べ方 (この質問はこれに従う)",
           "- 最後に原因・要因の候補を確度順に 5〜10 件まとめる。1 つの候補だけで決めつけず、"
           "複数の候補について根拠になる値を集める。調べずにすぐ answer に移らない。"]
    if db_on:
        out += ["- DB では「問題がある群」と「比べる群」(不良ロットと良品ロット・発生前と後・装置別など) を "
                "SQL の集計で並べ、件数・平均・率の差を確かめる。差を確かめた列が候補になる。",
                "- 差が見つかった装置は、保全の記録や装置のパラメータを日付で突き合わせる。",
                "- 候補にする列は describe_table で実在と意味を確かめる。"]
    out.append("- 文書 (search_docs) で過去の類似事例・既知の原因・対策を探す。")
    return out


CAUSE_FINAL = (
    "\n## 原因・要因を問われたときの回答の形式 (この質問はこれに従う)\n"
    "1. 最初に結論を 1〜3 文で書く (最も確からしい候補と根拠の要点)。\n"
    "2. 次の見出しと表で、原因・要因の候補を確度の高い順に 5〜10 件挙げる。\n"
    "### 原因・要因の候補 (確度順)\n"
    "| 順位 | 候補 | 種別 | 確度 | 根拠 | 次に確かめること |\n"
    "|---|---|---|---|---|---|\n"
    "- 種別: DB 列 / 文書 / 工程・装置 / 仮説 のどれか。\n"
    "- 確度: 高 / 中 / 低 / 仮説 の語で書く。確度を確率や % の数値で書かない。"
    "高 = DB の比較で明確な差があり、文書の事例とも合う。中 = どちらか一方で支持される。"
    "低 = 差が小さい・間接的な根拠だけ。仮説 = 記録に直接の根拠が無い (一般的な知見による)。\n"
    "- 根拠: 記録の値 (例: 対象群の平均 12.4、比較群 3.1 — 手順 3) か文書の引用 [n]。仮説は「記録に根拠なし」。\n"
    "- 記録に根拠のある候補が 5 件に満たなければ、一般的な知見による候補を「仮説」として足す。"
    "数合わせのために記録の値や列名を作らない。\n"
    "- データの差は相関であり因果とは限らない。断定せず「〜が原因と考えられる」と書く。\n"
    "3. 表の後に「次の調査」として、上位の候補を確かめる手順を 2〜3 個の箇条書きで書く。\n"
)
TRUNCATED_MSG = ("回答が出力の上限 (num_predict) で途中で切れました。推論モデルは思考も上限を使うため、上限を増やすか思考を減らしてください。")
_RANK_CELL_RE = re.compile(r"(?m)^(\s*\|\s*)\d{1,2}(\s*\|)")


class Agent:
    def __init__(self, llm: LLMClient, *, search: Optional[Callable[..., List[Any]]] = None,
                 catalog: Optional[Catalog] = None,
                 reader_factory: Optional[Callable[[], SQLiteReader]] = None,
                 max_steps: int = 8, cause_max_steps: int = 12, obs_chars: int = 3500,
                 step_predict: int = 2048, num_ctx: int = 16384, num_predict: int = 4096,
                 est_chars_per_token: float = 1.5, stream: bool = True, chart_max_rows: int = 5000):
        """search(query, top_k, path_contains=None) → SearchHit (または dict) のリスト。None = 文書検索なし。"""
        self.llm = llm
        self.search_fn = search
        self.catalog = catalog
        self.reader_factory = reader_factory
        self.max_steps = max(1, int(max_steps))
        self.cause_max_steps = max(self.max_steps, int(cause_max_steps or 0))
        self.obs_chars = int(obs_chars)
        self.step_predict = int(step_predict)
        self.num_ctx = int(num_ctx)
        self.num_predict = int(num_predict)
        self.cpt = float(est_chars_per_token)
        self.stream = stream
        self.chart_max_rows = int(chart_max_rows)

    # 予算 --------------------------------------------------------------------
    def _budget_chars(self, predict: int) -> int:
        return int(max(512, self.num_ctx - predict - 256) * self.cpt)

    @staticmethod
    def _fit(messages: List[Dict[str, str]], budget: int, keep_last: int = 4) -> List[Dict[str, str]]:
        """文脈長を超えそうなら、古い観測から順に中身を縮める (system と直近は残す)。"""
        msgs = [dict(m) for m in messages]
        total = sum(len(m["content"]) for m in msgs)
        i = 1
        while total > budget and i < len(msgs) - keep_last:
            m = msgs[i]
            if m["role"] == "user" and m["content"].startswith("観測") and len(m["content"]) > 400:
                cut = m["content"][:300] + "\n…(古い観測のため省略)"
                total -= len(m["content"]) - len(cut)
                m["content"] = cut
            i += 1
        return msgs

    # 道具 --------------------------------------------------------------------
    def specs(self, db_on: bool) -> List[Dict[str, Any]]:
        out = [SEARCH_SPEC] if self.search_fn is not None else []
        if db_on:
            out += DB_TOOL_SPECS + [CHART_SPEC]
        return out

    def _search(self, args: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        q = str(args.get("query") or "").strip()
        if not q:
            return {"error": "query が空です。"}
        try:
            top_k = max(1, min(10, int(args.get("top_k") or 5)))
        except (TypeError, ValueError):
            top_k = 5
        try:
            res = self.search_fn(q, top_k, path_contains=(str(args.get("path_contains") or "") or None))
        except Exception as e:                      # noqa: BLE001 — 検索の失敗も観測として返す
            return {"error": "検索に失敗しました: %s" % e}
        out = []
        for h in res or []:
            h = h if isinstance(h, dict) else h.to_dict()
            known = next((x for x in hits if x.get("chunk_id") == h.get("chunk_id")), None)
            if known is None:
                known = dict(h, n=len(hits) + 1)
                hits.append(known)
            out.append({"cite": "[%d]" % known["n"], "path": h.get("path", ""),
                        "text": " ".join((h.get("text") or "").split())[:700]})
        return {"hits": out} if out else {"hits": [], "note": "該当する文書はありませんでした。"}

    def _chart(self, args: Dict[str, Any], reader: Optional[SQLiteReader],
               results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        note = ""
        if args.get("from_step") not in (None, ""):
            try:
                step = int(args["from_step"])
            except (TypeError, ValueError):
                return {"error": "from_step は手順番号 (整数) で指定する。"}
            data = results.get(step)
            if not data:
                return {"error": "手順 %d には表の結果がありません (使える手順: %s)。sql で指定してもよい。"
                                 % (step, ", ".join(map(str, sorted(results))) or "なし")}
            if data.get("truncated"):
                note = "手順 %d の結果は先頭 %d 行だけ (全体を描くには sql で指定する)" % (step, len(data["rows"]))
        elif args.get("sql"):
            if reader is None:
                return {"error": "DB は使えません。"}
            try:
                data = reader.query(str(args["sql"]), max_rows=self.chart_max_rows)
            except DBError as e:
                return {"error": str(e)}
            if data.get("truncated"):
                note = "先頭 %d 行だけを描いた (集計して行数を減らすとよい)" % self.chart_max_rows
        else:
            return {"error": "データを sql (SELECT 1 文) か from_step (前の query の手順番号) で指定する。"}
        if not data.get("rows"):
            return {"error": "データが 0 行のためグラフを描けません。"}
        spec_in = {k: args.get(k) for k in ("type", "title", "x", "y", "series", "xlabel", "ylabel")}
        try:
            spec = charts.normalize(spec_in, data["columns"])
            charts.check_data(spec, data["columns"], data["rows"])
            png = charts.render_png(spec, data["columns"], data["rows"])
        except charts.ChartError as e:
            return {"error": str(e)}
        except Exception as e:                        # matplotlib 内部の失敗など
            return {"error": "グラフを描けませんでした: %s" % e}
        summary = charts.describe(spec, data["columns"], data["rows"]) + ("。" + note if note else "")
        return {"summary": summary, "rows_used": len(data["rows"]),
                "_chart": {"title": spec["title"] or "グラフ", "src": charts.to_data_url(png),
                           "csv": {"columns": data["columns"], "rows": data["rows"]}}}

    def _execute(self, name: str, args: Dict[str, Any], reader: Optional[SQLiteReader],
                 cat: Optional[Catalog], hits: List[Dict[str, Any]],
                 results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        if name == "search_docs":
            if self.search_fn is None:
                return {"error": "文書検索は使えません。"}
            return self._search(args, hits)
        if name == "make_chart":
            result = self._chart(args, reader, results)
        elif reader is None or cat is None:
            return {"error": "DB は使えません。"}
        else:
            result = run_tool(reader, cat, name, args)
        if "error" in result and name in ("query", "explain", "make_chart") and cat is not None:
            hint = column_hint(str(args.get("sql", "")), cat, str(result["error"]))
            if hint:
                result["hint"] = hint
        return result

    # 実行 --------------------------------------------------------------------
    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None, *,
            temperature: Optional[float] = None, cause: Optional[bool] = None) -> Iterator[Dict[str, Any]]:
        cat = self.catalog
        db_on = bool(cat is not None and len(cat) and self.reader_factory is not None)
        specs = self.specs(db_on)
        names = {s["name"] for s in specs}
        cause = is_cause_question(question) if cause is None else bool(cause)
        max_steps = self.cause_max_steps if cause else self.max_steps
        yield {"type": "agent", "tools": sorted(names), "max_steps": max_steps,
               "catalog_tables": len(cat) if cat is not None else 0, "cause": cause}
        if not db_on:
            yield {"type": "warn", "message": "DB が使えないため、文書検索だけで調べます。"}

        system = step_system_prompt(specs, db_on=db_on, n_tables=len(cat) if cat else 0,
                                    overview=cat.overview() if db_on else "", max_steps=max_steps, cause=cause)
        hist = self._history(history)
        msgs = [{"role": "system", "content": system}] + hist + \
               [{"role": "user", "content": question + "\n\n" + REMIND}]
        trace: List[Dict[str, Any]] = []
        hits: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        results: Dict[int, Dict[str, Any]] = {}
        format_errors = repeats = 0
        reader = self.reader_factory() if db_on else None
        step_opts = {"temperature": 0.1 if temperature is None else temperature, "num_predict": self.step_predict}
        try:
            n = 0
            finished = False
            while n < max_steps:
                try:
                    data = self.llm.chat(self._fit(msgs, self._budget_chars(self.step_predict)), options=step_opts)
                except LLMError as e:
                    yield {"type": "error", "message": str(e)}
                    return
                text = data.get("content") or ""
                obj = extract_json(text)
                if obj is None or not (obj.get("action") or "final" in obj):
                    format_errors += 1
                    if format_errors > 2:
                        yield {"type": "warn", "message": "LLM が指示の形式 (JSON) で返せなかったため、"
                                                          "ここまでの情報で回答します。"}
                        break
                    msgs += [{"role": "assistant", "content": strip_think(text)[:600]},
                             {"role": "user", "content": "形式エラー: 説明や思考は書かず、"
                                                         '{"thought": …, "action": …, "args": …} の '
                                                         "JSON オブジェクト 1 つだけを返してください。"}]
                    continue
                action = str(obj.get("action") or ("answer" if "final" in obj else "")).strip()
                thought = str(obj.get("thought") or "").strip()
                if action in ANSWER_ACTIONS:
                    if thought:
                        trace.append({"note": thought})
                    finished = True
                    break
                args = obj.get("args") or {}
                if isinstance(args, str):
                    args = extract_json(args) or {"value": args}
                if not isinstance(args, dict):
                    args = {}
                n += 1
                yield {"type": "step", "n": n, "thought": thought, "tool": action, "args": args}
                t0 = time.time()
                sig = json.dumps([action, args], ensure_ascii=False, sort_keys=True)
                if action not in names:
                    result = {"error": "未知の道具です: %s (使えるのは %s)" % (action, ", ".join(sorted(names)))}
                elif sig in seen:
                    repeats += 1
                    prev = next((t for t in trace if t.get("n") == seen[sig]), {})
                    result = {"error": "手順 %d と同じ呼び出しです。同じことを繰り返さず、前の結果を踏まえて"
                                       "引数を変えるか、別の道具を使ってください。" % seen[sig],
                              "previous_result": (prev.get("result") or "")[:800]}
                else:
                    seen[sig] = n
                    result = self._execute(action, args, reader, cat, hits, results)
                ok = "error" not in result
                if ok and action in ("query", "sample_rows") and "rows" in result:
                    results[n] = result
                chart = result.pop("_chart", None)
                if chart:
                    yield {"type": "chart", "n": n, "title": chart["title"], "src": chart["src"],
                           "csv": chart.get("csv")}
                obs = format_obs(result)
                if len(obs) > self.obs_chars:
                    obs = obs[:self.obs_chars] + "…(以下省略)"
                yield {"type": "observation", "n": n, "tool": action, "ok": ok,
                       "summary": summarize(action, result), "detail": obs[:6000],
                       "elapsed_ms": int((time.time() - t0) * 1000)}
                trace.append({"n": n, "tool": action, "args": args, "thought": thought, "result": obs, "ok": ok})
                if repeats >= 2:
                    yield {"type": "warn", "message": "同じ呼び出しを繰り返したため、調査を打ち切って"
                                                      "ここまでの結果で回答します。"}
                    break
                msgs += [{"role": "assistant", "content": json.dumps(
                              {"thought": thought, "action": action, "args": args}, ensure_ascii=False)},
                         {"role": "user", "content": "観測 (手順 %d: %s)\n%s\n\n残り %d 手。%s"
                                                     % (n, action, obs, max_steps - n, REMIND)}]
            if not finished and n >= max_steps and repeats < 2:
                yield {"type": "warn", "message": "手数の上限 (%d 手) に達したため、"
                                                  "ここまでの結果で回答します。" % max_steps}
        finally:
            if reader is not None:
                reader.close()

        if hits:
            yield {"type": "sources", "hits": hits}
        yield from self._final(question, hist, trace, temperature, cause=cause)

    def _history(self, history: Optional[List[Dict[str, str]]], cap: int = 3000) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        used = 0
        for m in reversed([m for m in (history or []) if (m.get("content") or "").strip()]):
            c = m["content"] if len(m["content"]) <= 1200 else m["content"][:1200] + "…"
            if used + len(c) > cap:
                break
            out.insert(0, {"role": m.get("role", "user"), "content": c})
            used += len(c)
        return out

    def _final(self, question: str, hist: List[Dict[str, str]], trace: List[Dict[str, Any]],
               temperature: Optional[float], cause: bool = False) -> Iterator[Dict[str, Any]]:
        steps = [t for t in trace if "note" not in t]
        if steps and not any(t.get("ok") for t in steps):
            yield {"type": "content", "text": no_result_answer(steps)}
            return
        final_system = FINAL_SYSTEM + (CAUSE_FINAL if cause else "")
        budget = self._budget_chars(self.num_predict) - len(final_system) - len(question) - 500
        limits = allocate(steps, budget)
        lines = []
        for t in trace:
            if "note" in t:
                lines.append("### まとめのメモ (推測を含む)\n" + t["note"])
                continue
            per = limits.get(t["n"], 400)
            res = t["result"] if len(t["result"]) <= per else t["result"][:per] + "…(省略)"
            lines.append("### 手順 %d: %s %s — %s\n目的: %s\n結果: %s" % (
                t["n"], t["tool"], json.dumps(t["args"], ensure_ascii=False)[:400],
                "成功" if t.get("ok") else "失敗 (この結果は根拠にしない)", t.get("thought", ""), res))
        record = "\n\n".join(lines) if lines else "(道具は使っていない)"
        if steps and not data_obtained(steps):
            status = ("DB の値も文書も 1 件も取得できていない。数値・ロット名・件数を書いてはいけない。"
                      "分かったこと (候補の表や列など) と、取得できなかった理由だけを書く。")
        elif steps:
            status = "成功した手順の結果だけを根拠にする。"
        else:
            status = "道具は使っていない。一般的な内容だけで答え、データについては推測しない。"
        user = "## 質問\n%s\n\n## データ取得の状況\n%s\n\n## 調査の記録\n%s" % (question, status, record)
        msgs = [{"role": "system", "content": final_system}] + hist[-4:] + [{"role": "user", "content": user}]
        opts: Dict[str, Any] = {"num_predict": self.num_predict}
        if temperature is not None:
            opts["temperature"] = temperature
        answer: List[str] = []
        try:
            if self.stream:
                for ev in self.llm.chat_stream(msgs, options=opts):
                    if ev["type"] == "error":
                        yield {"type": "error", "message": ev["text"]}
                        return
                    if ev["type"] in ("content", "thinking"):
                        if ev["type"] == "content":
                            answer.append(ev["text"])
                        yield ev
                    elif ev["type"] == "done" and ev["text"] == "length":
                        yield {"type": "warn", "message": TRUNCATED_MSG}
            else:
                data = self.llm.chat(msgs, options=opts)
                answer.append(data.get("content") or "")
                yield {"type": "content", "text": answer[-1]}
        except LLMError as e:
            yield {"type": "error", "message": str(e)}
            return
        text = "".join(answer)
        if cause:
            text = _RANK_CELL_RE.sub(r"\1\2", text)
        bad = unsupported_numbers(text, question + "\n" + record + "\n".join(t["content"] for t in hist))
        if bad:
            yield {"type": "warn", "message": "回答に調査記録に無い数値が含まれています (要確認): " + ", ".join(bad)}


def format_obs(result: Dict[str, Any]) -> str:
    """観測を LLM に返す文字列。エラーは平文にして、直し方 (実在する列など) を先頭に置く
    — 小さいモデルは JSON の奥にあるヒントを読まずに同じ SQL を繰り返すため。"""
    if "error" not in result:
        return json.dumps(result, ensure_ascii=False, default=str)
    lines = ["エラー: %s" % result["error"]]
    hint = result.get("hint") or {}
    for key, h in hint.items():
        if isinstance(h, dict):
            lines.append("表 %s の実在する列: %s" % (key, ", ".join(h.get("columns") or [])))
    if hint.get("note"):
        lines.append("→ " + hint["note"])
    if result.get("previous_result"):
        lines.append("前回の結果: " + str(result["previous_result"]))
    return "\n".join(lines)


DATA_TOOLS = ("query", "make_chart", "search_docs", "sample_rows")


def allocate(steps: List[Dict[str, Any]], budget: int) -> Dict[int, int]:
    """最終回答に渡す記録の長さを手順ごとに割り振る。均等割りだと手数が多いときに肝心の集計結果まで
    切られ、LLM が見えない値を推測で書く。失敗・表の検索や定義は短く、値を返した手順に残りを回す。"""
    out: Dict[int, int] = {}
    data = []
    for t in steps:
        size = len(t.get("result") or "")
        if not t.get("ok"):
            out[t["n"]] = min(size, 300)
        elif t["tool"] in DATA_TOOLS:
            data.append((t["n"], size))
        else:
            out[t["n"]] = min(size, 900)
    remaining = max(0, budget - sum(out.values()))
    data.sort(key=lambda x: x[1])
    for i, (n, size) in enumerate(data):
        share = remaining // max(1, len(data) - i)
        out[n] = max(400, min(size, share))
        remaining -= min(size, share)
    return out


def data_obtained(steps: List[Dict[str, Any]]) -> bool:
    """回答の根拠になる値 (DB の行・文書) を 1 つでも取れたか。表の検索や定義だけでは足りない。"""
    for t in steps:
        if not t.get("ok"):
            continue
        if t["tool"] in ("query", "sample_rows", "make_chart"):
            return True
        if t["tool"] == "search_docs" and '"hits": [{' in t.get("result", ""):
            return True
    return False


_NUM_RE = re.compile(r"(?<![0-9A-Za-z])(?<!\d\.)\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> List[Tuple[str, int]]:
    t = unicodedata.normalize("NFKC", text or "")
    return [(m.group(0).replace(",", "").rstrip("."), m.end()) for m in _NUM_RE.finditer(t)]


def _derived(v: str, ev: List[float]) -> bool:
    """記録の 2 つの数値の 比・百分率・差・和 で説明できるか (表示桁で丸めて比べる)。"""
    try:
        target = float(v)
    except ValueError:
        return False
    digits = len(v.split(".")[1]) if "." in v else 0
    tol = 0.5 * 10 ** (-digits) + 1e-9
    for a in ev:
        for b in ev:
            if b and (abs(a / b - target) <= tol or abs(a / b * 100 - target) <= tol):
                return True
            if abs(a - b - target) <= tol or abs(a + b - target) <= tol:
                return True
    return False


def _rounded_from(v: str, ev_raw) -> bool:
    """記録のどれかの値を v と同じ桁に四捨五入すると v になるか。"""
    digits = len(v.split(".")[1])
    target = float(v)
    for e in ev_raw:
        if "." not in e or len(e.split(".")[1]) <= digits:
            continue
        try:
            if abs(round(float(e), digits) - target) < 10 ** (-digits - 3):
                return True
        except ValueError:
            continue
    return False


def unsupported_numbers(answer: str, evidence: str, limit: int = 8) -> List[str]:
    """回答の数値のうち、調査記録 (と質問) で説明できないもの。
    許すもの: 記録にある値・その丸め・記録の 2 値からの計算 (比・%・差・和)・
    ID や日付に埋もれた 4 桁以上の値・「100 %」。手順番号・1 桁・引用番号 [n] は見ない。"""
    ev_raw = {x for x, _ in _numbers(evidence)}
    text = re.sub(r"\s*[×xX*]\s*10(?:[⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|\^[-+]?\d+)", " ", answer or "")
    text = unicodedata.normalize("NFKC", re.sub(r"\[n?\d{1,2}\]|手順\s*\d+", " ", text))
    found = _numbers(text)
    ev_f = sorted({float(x) for x, _ in found if x in ev_raw and x.replace(".", "", 1).isdigit()})
    out: List[str] = []
    for v, end in found:
        if len(v.replace(".", "")) < 2 or v in ev_raw or v in out:
            continue
        if "." in v and any(e.startswith(v) for e in ev_raw):        # 12.34 → 12.3 (切り捨て)
            continue
        if "." in v and _rounded_from(v, ev_raw):                      # 18.3965 → 18.397 (四捨五入)
            continue
        if len(v) >= 4 and "." not in v and any(v in e for e in ev_raw):
            continue
        if v == "100" and text[end:end + 2].strip().startswith("%"):
            continue
        if _derived(v, ev_f):
            continue
        out.append(v)
    return out[:limit]


def no_result_answer(steps: List[Dict[str, Any]]) -> str:
    lines = ["調べましたが、答えの根拠になるデータを取得できませんでした。", "", "**試したこと**"]
    for t in steps[:8]:
        err = str(t.get("result") or "")
        if err.startswith("エラー: "):
            err = err[4:].splitlines()[0]
        lines.append("- 手順 %d `%s`: %s" % (t["n"], t["tool"], err[:160] or "結果なし"))
    lines += ["", "質問を具体的にする (表名・ロット・期間など) と、調べられることがあります。"]
    return "\n".join(lines)


_FROM_RE = re.compile(r'\b(?:from|join)\s+("[^"]+"|[A-Za-z_][\w$]*)', re.I)


def column_hint(sql: str, cat: Catalog, error: str) -> Dict[str, Any]:
    """列・表が無いエラーのとき、SQL が参照した表の実在する列を添える
    (小さいモデルは describe_table を飛ばして列名を推測しがちなので、エラーから直せるように)。"""
    if not re.search(r"no such column|no such table|syntax error|ambiguous", error, re.I):
        return {}
    out: Dict[str, Any] = {}
    for m in _FROM_RE.finditer(sql or ""):
        t = cat.get(m.group(1).strip('"'))
        if t is not None:
            out[t["name"]] = {"columns": [c["name"] for c in t["columns"]][:80]}
    if not out and re.search(r"no such table", error, re.I):
        return {"note": "その表はありません。find_tables で正しい表名を探してください。"}
    if out:
        out["note"] = "実在する列名だけを使うこと。"
    return out


def summarize(tool: str, r: Dict[str, Any]) -> str:
    """観測の 1 行要約 (画面の「調査の過程」に出す)。"""
    if "error" in r:
        return "エラー: " + str(r["error"])[:160]
    if tool == "search_docs":
        h = r.get("hits") or []
        return "文書 %d 件%s" % (len(h), ": " + ", ".join(
            "%s %s" % (x.get("cite", ""), re.split(r"[\\/]", x["path"])[-1]) for x in h[:3]) if h else "")
    if tool == "find_tables":
        t = r.get("tables") or []
        return "候補 %d 表%s" % (len(t), ": " + ", ".join(x["table"] for x in t[:5]) if t else "")
    if tool == "describe_table":
        return "%s (%s 列, %s 行)%s" % (r.get("table"), r.get("column_count"),
                                       "{:,}".format(r["est_rows"]) if r.get("est_rows") is not None else "?",
                                       " · " + r["description"][:60] if r.get("description") else "")
    if tool == "sample_rows":
        return "%d 行" % len(r.get("rows") or [])
    if tool == "explain":
        return "全件走査 %d 箇所" % len(r.get("full_scans") or [])
    if tool == "query":
        return "%d 行%s (%s ms)" % (r.get("row_count", 0), " (打ち切り)" if r.get("truncated") else "",
                                    r.get("elapsed_ms", "?"))
    if tool == "make_chart":
        return str(r.get("summary", "グラフを表示"))[:200]
    return "完了"
