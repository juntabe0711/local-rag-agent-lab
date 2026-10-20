"""エージェントのループを、台本どおりに返す偽の LLM で確かめる。"""
import json
import sqlite3

import pytest

from agent.agent import Agent, allocate, is_cause_question, unsupported_numbers
from agent.sqldb import Catalog, SQLiteReader


class ScriptedLLM:
    """chat() は台本を順に返す。chat_stream() は最終回答を返す。"""

    def __init__(self, steps, final="答え"):
        self.steps = list(steps)
        self.final = final
        self.seen = []

    def chat(self, messages, options=None):
        self.seen.append(messages)
        nxt = self.steps.pop(0) if self.steps else {"thought": "終わり", "action": "answer"}
        return {"content": nxt if isinstance(nxt, str) else json.dumps(nxt, ensure_ascii=False), "thinking": ""}

    def chat_stream(self, messages, options=None):
        self.final_messages = messages
        yield {"type": "thinking", "text": "考え中"}
        yield {"type": "content", "text": self.final}
        yield {"type": "done", "text": ""}


@pytest.fixture
def factory(tmp_path):
    p = tmp_path / "f.sqlite"
    conn = sqlite3.connect(p)
    conn.executescript("CREATE TABLE lots (lot_id TEXT, equipment TEXT, lpd INTEGER);")
    conn.executemany("INSERT INTO lots VALUES (?,?,?)",
                     [("L1", "POL-01", 8), ("L2", "POL-03", 21), ("L3", "POL-03", 23), ("L4", "POL-01", 9)])
    conn.commit()
    conn.close()
    r = SQLiteReader(p)
    cat = Catalog.build(r)
    r.close()
    return p, cat


def run(agent, q):
    return list(agent.run(q))


def test_full_loop_with_db_doc_and_chart(factory):
    path, cat = factory
    steps = [
        "<think>まず表を探す</think>```json\n" + json.dumps({"thought": "表を探す", "action": "find_tables",
                                                            "args": {"keyword": "lots"}}) + "\n```",
        {"thought": "装置別に平均", "action": "query",
         "args": {"sql": "SELECT equipment, avg(lpd) AS avg_lpd FROM lots GROUP BY equipment ORDER BY equipment"}},
        {"thought": "グラフ", "action": "make_chart", "args": {"type": "bar", "x": "equipment", "y": "avg_lpd",
                                                              "from_step": 2, "title": "装置別 LPD"}},
        {"thought": "事例を探す", "action": "search_docs", "args": {"query": "パッド交換 LPD"}},
        {"thought": "分かった", "action": "answer"},
    ]
    docs = [{"chunk_id": 7, "path": "sample_docs/TR.md", "text": "パッド交換後に LPD が増えた事例", "score": 0.9}]
    llm = ScriptedLLM(steps, final="POL-03 の平均は 22 で POL-01 は 8.5 です [1]。")
    ag = Agent(llm, search=lambda q, k, path_contains=None: docs, catalog=cat,
               reader_factory=lambda: SQLiteReader(path), stream=True)
    ev = run(ag, "装置別の LPD は？")
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "agent" and "chart" in kinds and "sources" in kinds
    obs = [e for e in ev if e["type"] == "observation"]
    assert [o["ok"] for o in obs] == [True, True, True, True]
    assert obs[1]["summary"].startswith("2 行")
    assert next(e for e in ev if e["type"] == "chart")["src"].startswith("data:image/png;base64,")
    assert "".join(e["text"] for e in ev if e["type"] == "content").startswith("POL-03")
    assert not [e for e in ev if e["type"] == "warn"]          # 22 / 8.5 は記録にある
    record = llm.final_messages[-1]["content"]
    assert "手順 2: query" in record and "成功" in record


def test_format_errors_then_gives_up(factory):
    path, cat = factory
    llm = ScriptedLLM(["ただの文章です", "まだ文章", "やっぱり文章"])
    ag = Agent(llm, catalog=cat, reader_factory=lambda: SQLiteReader(path))
    ev = run(ag, "何件？")
    assert any(e["type"] == "warn" and "JSON" in e["message"] for e in ev)
    assert any(e["type"] == "content" for e in ev)


def test_repeat_detection_and_failed_steps_answer_without_llm(factory):
    path, cat = factory
    bad = {"thought": "x", "action": "query", "args": {"sql": "SELECT nope FROM lots"}}
    llm = ScriptedLLM([bad, bad, bad])
    ag = Agent(llm, catalog=cat, reader_factory=lambda: SQLiteReader(path))
    ev = run(ag, "何件？")
    obs = [e for e in ev if e["type"] == "observation"]
    assert "実在する列" in obs[0]["detail"]                      # column_hint
    assert "同じ呼び出し" in obs[1]["detail"]
    assert any(e["type"] == "warn" and "繰り返した" in e["message"] for e in ev)
    content = "".join(e["text"] for e in ev if e["type"] == "content")
    assert content.startswith("調べましたが")                     # LLM に書かせない定型文
    assert not hasattr(llm, "final_messages")


def test_unknown_tool_and_cause_mode(factory):
    path, cat = factory
    llm = ScriptedLLM([{"thought": "x", "action": "rm_rf", "args": {}}])
    ag = Agent(llm, catalog=cat, reader_factory=lambda: SQLiteReader(path), max_steps=3, cause_max_steps=6)
    ev = run(ag, "LPD が増えた原因は？")
    assert ev[0]["cause"] is True and ev[0]["max_steps"] == 6
    assert "未知の道具" in [e for e in ev if e["type"] == "observation"][0]["summary"]
    assert "原因・要因の候補" in llm.seen[0][0]["content"]


def test_no_db_warns_and_uses_docs_only():
    llm = ScriptedLLM([{"thought": "挨拶", "action": "answer"}], final="こんにちは")
    ev = run(Agent(llm, search=lambda *a, **k: []), "こんにちは")
    assert ev[1]["type"] == "warn"
    assert "search_docs" in ev[0]["tools"] and "query" not in ev[0]["tools"]


def test_unsupported_numbers():
    ev = "平均 12.34 / 件数 80363 / 22655 / 日付 20260915"
    assert unsupported_numbers("平均は12.3、件数 80,363", ev) == []           # 丸め・桁区切り
    # 計算の材料は「回答にも書かれていて記録にもある値」だけ (偶然の一致で通さないため)
    assert unsupported_numbers("22,655 ÷ 80,363 ≈ 28.2 %", ev) == []
    assert unsupported_numbers("比率は 28.2 %", ev) == ["28.2"]
    assert unsupported_numbers("2026 年の値は 45.6 です", ev) == ["45.6"]     # 2026 は日付に埋もれている
    assert unsupported_numbers("[3] によると 手順 4 で", ev) == []
    assert unsupported_numbers("平均 18.397", "avg 18.396522") == []            # 四捨五入
    assert unsupported_numbers("平均 18.39", "avg 18.396522") == []             # 切り捨て
    assert unsupported_numbers("平均 18.41", "avg 18.396522") == ["18.41"]


def test_allocate_gives_data_steps_the_budget():
    steps = [{"n": 1, "tool": "find_tables", "ok": True, "result": "x" * 5000},
             {"n": 2, "tool": "query", "ok": True, "result": "y" * 5000},
             {"n": 3, "tool": "query", "ok": False, "result": "z" * 5000}]
    lim = allocate(steps, 4000)
    assert lim[1] == 900 and lim[3] == 300 and lim[2] == 4000 - 1200


def test_is_cause_question():
    assert is_cause_question("なぜ増えた？") and not is_cause_question("平均は？")


def test_truncated_final_answer_warns(factory):
    path, cat = factory

    class Cut(ScriptedLLM):
        def chat_stream(self, messages, options=None):
            yield {"type": "content", "text": "途中まで"}
            yield {"type": "done", "text": "length"}

    ev = run(Agent(Cut([{"thought": "x", "action": "answer"}]), catalog=cat,
                   reader_factory=lambda: SQLiteReader(path)), "平均は？")
    assert any(e["type"] == "warn" and "途中で切れ" in e["message"] for e in ev)
