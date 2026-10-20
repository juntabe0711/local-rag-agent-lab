import sqlite3

import pytest

from agent.sqldb import Catalog, DBError, SQLiteReader, clean_select, run_tool


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE lots (lot_id TEXT PRIMARY KEY, grade TEXT, lpd INTEGER);
        CREATE TABLE secrets (name TEXT, api_key TEXT, db_password TEXT);
        CREATE INDEX idx_lots_grade ON lots(grade);
    """)
    conn.executemany("INSERT INTO lots VALUES (?,?,?)", [(f"L{i}", "prime" if i % 2 else "epi", i) for i in range(100)])
    conn.execute("INSERT INTO secrets VALUES ('svc', 'sk-123', 'hunter2')")
    conn.commit()
    conn.close()
    return p


@pytest.mark.parametrize("sql", [
    "SELECT 1", "select * from lots;", "WITH t AS (SELECT 1 AS a) SELECT a FROM t",
    "SELECT 'insert into x' AS s", 'SELECT "lot_id" FROM lots -- drop table lots',
    "SELECT replace(lot_id, 'L', '') FROM lots",
])
def test_clean_select_allows(sql):
    assert clean_select(sql)


@pytest.mark.parametrize("sql,needle", [
    ("DELETE FROM lots", "SELECT"), ("SELECT 1; DROP TABLE lots", "複数"),
    ("PRAGMA table_info(lots)", "SELECT"), ("SELECT * FROM lots WHERE 1 /* x", "引用符"),
    ("WITH d AS (DELETE FROM lots RETURNING *) SELECT * FROM d", "DELETE"),
    ("ATTACH DATABASE 'x.db' AS x", "SELECT"), ("SELECT load_extension('evil')", "load_extension"),
    ("SELECT api_key FROM secrets", "機密列"), ('SELECT "db_password" AS p FROM secrets', "機密列"),
    ("", "空"),
])
def test_clean_select_rejects(sql, needle):
    with pytest.raises(DBError) as e:
        clean_select(sql)
    assert needle in str(e.value)


def test_reader_is_read_only_and_redacts(db):
    r = SQLiteReader(db, max_rows=10)
    try:
        out = r.query("SELECT lot_id, lpd FROM lots ORDER BY lpd")
        assert out["row_count"] == 10 and out["truncated"] is True
        # SELECT * でも機密列は認可で断られる
        with pytest.raises(DBError) as e:
            r.query("SELECT * FROM secrets")
        assert "許可されていない" in str(e.value)
        # エンジン側でも書けない (検査をすり抜けた場合の保険)
        with pytest.raises(sqlite3.OperationalError):
            r.conn.execute("INSERT INTO lots VALUES ('X', 'x', 1)")
        s = r.sample_rows("secrets", 3)
        assert s["rows"][0][s["columns"].index("api_key")] == "***"
        assert s["rows"][0][s["columns"].index("name")] == "svc"
    finally:
        r.close()


def test_reader_timeout(db):
    r = SQLiteReader(db, timeout_sec=0.2)
    try:
        with pytest.raises(DBError) as e:
            r.query("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")
        assert "打ち切" in str(e.value)
    finally:
        r.close()


def test_catalog_and_tools(db, tmp_path):
    desc = tmp_path / "d.json"
    desc.write_text('{"tables": {"lots": {"description": "ロットとパーティクル", "columns": {"lpd": "LPD 個数"}}}}',
                    encoding="utf-8")
    r = SQLiteReader(db)
    try:
        cat = Catalog.build(r, desc)
        assert len(cat) == 2
        found = run_tool(r, cat, "find_tables", {"keyword": "パーティクル"})
        assert found["tables"][0]["table"] == "lots"
        brief = run_tool(r, cat, "describe_table", {"table": "lots"})
        assert brief["description_source"] == "manual" and brief["est_rows"] == 100
        grade = next(c for c in brief["columns"] if c["name"] == "grade")
        assert set(grade["common_vals"]) == {"prime", "epi"}
        assert any("idx_lots_grade" in x or "lots(grade)" in x for x in brief["indexes"])
        sec = run_tool(r, cat, "describe_table", {"table": "secrets"})
        assert next(c for c in sec["columns"] if c["name"] == "api_key").get("note")
        assert "error" in run_tool(r, cat, "describe_table", {"table": "nope"})
        plan = run_tool(r, cat, "explain", {"sql": "SELECT * FROM lots WHERE grade = 'epi'"})
        assert "USING INDEX" in plan["plan"]
        q = run_tool(r, cat, "query", {"sql": "SELECT grade, count(*) n FROM lots GROUP BY grade ORDER BY grade"})
        assert q["rows"] == [["epi", 50], ["prime", 50]]
    finally:
        r.close()
