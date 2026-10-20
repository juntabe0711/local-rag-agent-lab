import sqlite3

from ragcore import rag_eval
from ragcore.indexer import open_state_db


def test_summarize_arithmetic():
    ranks = [1, 3, None, 7]
    file_ranks = [1, 1, None, 2]
    s = rag_eval.summarize(ranks, file_ranks, top_k=5, latencies_ms=[10, 20, 30, 40])
    assert s["n_questions"] == 4
    assert s["recall_1"] == 0.25
    assert s["recall_5"] == 0.5          # 1 と 3
    assert s["recall_k"] == 0.5          # k=5 なので 7 は外れ
    assert s["mrr"] == round((1 + 1 / 3 + 1 / 7) / 4, 4)
    assert s["file_recall_1"] == 0.5
    assert s["file_recall_k"] == 0.75
    assert s["miss"] == 1
    assert s["latency_ms"] == 25.0


def test_summarize_empty():
    s = rag_eval.summarize([], [], top_k=10)
    assert s["n_questions"] == 0 and s["mrr"] is None


def test_rank_uses_path_and_chunk_index_not_chunk_id():
    item = {"path": "docs/A.md", "chunk_index": 2, "chunk_id": 999}
    hits = [{"path": "docs/B.md", "chunk_index": 2, "chunk_id": 999},
            {"path": "docs/a.md", "chunk_index": 0, "chunk_id": 1},
            {"path": "docs/A.md", "chunk_index": 2, "chunk_id": 5}]
    assert rag_eval.rank_of(hits, item) == 3
    assert rag_eval.file_rank_of(hits, item) == 2      # 大文字小文字を無視


def test_extract_question_variants():
    assert rag_eval.extract_question("「CZ法の引き上げ速度は？」") == "CZ法の引き上げ速度は？"
    assert rag_eval.extract_question("以下が質問です。\n1. 研磨圧力の標準値は何kPaか？") == "研磨圧力の標準値は何kPaか？"
    assert rag_eval.extract_question("```\nQ: SC-1 の温度は？\n```") == "SC-1 の温度は？"
    assert rag_eval.extract_question("   \n") == ""


def test_score_records_exception_as_miss():
    golden = [{"q": "a", "path": "x", "chunk_index": 0}, {"q": "boom", "path": "y", "chunk_index": 1}]

    def search(q, k):
        if q == "boom":
            raise RuntimeError("down")
        return [{"path": "x", "chunk_index": 0}]

    out = rag_eval.score(golden, search, top_k=10)
    assert out["n_questions"] == 2 and out["recall_1"] == 0.5 and out["miss"] == 1
    assert out["details"][1]["error"].startswith("RuntimeError")


def test_sample_chunks_one_per_file_and_length(tmp_path):
    db = tmp_path / "state.sqlite"
    conn = open_state_db(db)
    for f in range(5):
        for c in range(4):
            text = ("x" * 50) if c == 0 else ("本文" * 150)
            conn.execute("INSERT INTO chunks (file_id, chunk_index, num_chunks, path, ext, text) VALUES (?,?,?,?,?,?)",
                         (f"f{f}", c, 4, f"docs/{f}.md", ".md", text))
    conn.commit()
    conn.close()
    import random
    rows = rag_eval.sample_chunks(db, 10, rng=random.Random(1))
    assert len(rows) == 5                                 # 1 文書 1 問
    assert all(200 <= len(r["text"]) <= 2000 for r in rows)
    # 読み取り専用で開いている (書けない)
    conn = rag_eval._ro_conn(db)
    try:
        conn.execute("DELETE FROM chunks")
        raise AssertionError("should be read-only")
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def test_golden_roundtrip_skips_bad_lines(tmp_path):
    p = tmp_path / "g.jsonl"
    rag_eval.save_golden(p, [rag_eval.GoldenItem(q="q1", path="a", chunk_index=0)])
    p.write_text(p.read_text(encoding="utf-8") + "# comment\n{broken\n{\"q\": \"\", \"path\": \"b\"}\n", encoding="utf-8")
    items = rag_eval.load_golden(p)
    assert len(items) == 1 and items[0].key == ("a", 0)


def test_record_run_writes_db_and_csv(tmp_path):
    row = rag_eval.now_row(source="manual", label="t", golden="g.jsonl",
                           **rag_eval.summarize([1], [1], top_k=10))
    rid = rag_eval.record_run(tmp_path / "e.sqlite", row, csv_path=tmp_path / "e.csv")
    assert rid == 1
    runs = rag_eval.list_runs(tmp_path / "e.sqlite")
    assert runs[0]["label"] == "t" and runs[0]["file_recall_k"] == 1.0
    assert (tmp_path / "e.csv").read_text(encoding="utf-8-sig").startswith("run_at,")
