"""索引のビルド (差分更新) と 2 段検索を、偽の埋め込みモデルで通しで確かめる。"""
import os
import time

import numpy as np
import pytest

from conftest import write_config
from ragcore import rag_eval
from ragcore.chunking import chunk_by_tokens
from ragcore.config import load_config
from ragcore.indexer import IndexBuilder
from ragcore.retriever import Retriever, SearchHit
from ragcore.vector_index import VectorIndex
from conftest import FakeTokenizer

DOCS = {
    "polish.md": "研磨工程ではパッドの交換後にならしを行う。研磨圧力は18kPaが標準である。" * 3,
    "clean.md": "洗浄工程のSC-1薬液温度は65度が標準で、66.5度を超えると鉄の汚染が増える。" * 3,
    "inspect.md": "表面検査ではLPDを45nm以上で数える。製品ごとに上限がある。" * 3,
}


def make_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def test_chunking_overlap_and_coverage():
    tok = FakeTokenizer()
    text = "".join(chr(0x3042 + (i % 80)) for i in range(100))
    chunks = chunk_by_tokens(text, tok, chunk_size=40, overlap=8)
    assert [len(c) for c in chunks] == [40, 40, 36]          # 0-40, 32-72, 64-100
    assert chunks[1][:8] == chunks[0][-8:]
    # 長い文字列 (ストリーム経路) でも重なりと網羅は同じ
    long = "あいうえお" * 4000
    ch = chunk_by_tokens(long, tok, 512, 64)
    assert all(len(c) <= 512 for c in ch) and sum(len(c) for c in ch) >= len(long)


@pytest.mark.parametrize("kind", ["flat", "flat_fp16", "flat_sq8"])
def test_build_incremental_and_search(tmp_path, fake_model, kind):
    docs = make_docs(tmp_path)
    cfg_path = write_config(tmp_path, docs=docs, index_type=kind)
    cfg, base = load_config(cfg_path)
    r1 = IndexBuilder(cfg, base, model=fake_model).build(progress=False)
    assert r1["processed"] == 3 and r1["total_chunks"] > 3

    # 2 回目: 変更なし → 埋め込みは呼ばれない
    calls = fake_model.calls
    r2 = IndexBuilder(cfg, base, model=fake_model).build(progress=False)
    assert r2["unchanged"] == 3 and r2["processed"] == 0 and fake_model.calls == calls

    # 1 つ変更・1 つ削除・1 つ追加
    time.sleep(0.01)
    (docs / "polish.md").write_text("研磨パッドはPAD-B系に切り替える。" * 4, encoding="utf-8")
    os.remove(docs / "inspect.md")
    (docs / "new.md").write_text("新しい文書。結晶の引き上げ速度について。" * 3, encoding="utf-8")
    r3 = IndexBuilder(cfg, base, model=fake_model).build(progress=False)
    assert r3["processed"] == 2 and r3["removed_files"] == 1

    retr = Retriever(cfg_path, model=fake_model, load_reranker=False)
    import sqlite3
    conn = sqlite3.connect(str(retr.state_db))
    paths = {p for (p,) in conn.execute("SELECT DISTINCT path FROM chunks")}
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert paths == {"docs/polish.md", "docs/clean.md", "docs/new.md"}
    assert retr.total == n_chunks                            # 索引と SQLite が一致

    hits = retr.search("SC-1薬液温度", top_k=3)
    assert hits and hits[0].path == "docs/clean.md" and hits[0].score_type == "cosine"
    # 絞り込み (IDSelector) は密検索そのものを絞る
    hits = retr.search("SC-1薬液温度", top_k=3, path_contains="polish")
    assert hits and all(h.path == "docs/polish.md" for h in hits)
    assert retr.search("x", top_k=3, ext_filter=[".pdf"]) == []


def test_rerank_then_mmr_then_min_score(tmp_path, fake_model):
    docs = make_docs(tmp_path)
    cfg_path = write_config(tmp_path, docs=docs)
    cfg, base = load_config(cfg_path)
    IndexBuilder(cfg, base, model=fake_model).build(progress=False)

    class FakeReranker:
        """inspect.md を最上位にする logit を返す (生の logit を受け取って sigmoid するか確かめる)。"""
        def predict(self, pairs, activation_fn=None, **_):
            assert activation_fn is not None
            return np.array([5.0 if "LPD" in p else -2.0 for _, p in pairs], dtype=np.float32)

    retr = Retriever(cfg_path, model=fake_model, reranker=FakeReranker())
    hits = retr.search("SC-1薬液温度", top_k=2)
    assert hits[0].path == "docs/inspect.md" and hits[0].score_type == "rerank"
    assert abs(hits[0].score - 1 / (1 + np.exp(-5.0))) < 1e-6
    assert hits[0].dense_score is not None

    retr.cfg["retrieval"]["min_score"] = 0.5                 # sigmoid(-2) ≈ 0.12 は落ちる
    assert all(h.path == "docs/inspect.md" for h in retr.search("SC-1", top_k=5))

    retr.cfg["retrieval"]["min_score"] = None
    retr.cfg["retrieval"]["mmr"] = {"enabled": True, "lambda": 0.0}   # 多様性だけ → 別の文書が混ざる
    hits = retr.search("LPD", top_k=3)
    assert len({h.path for h in hits}) >= 2


def test_fetch_k_rules():
    assert Retriever._fetch_k(5, 200, 10000, 0, has_filter=False, will_postprocess=True) == 200
    assert Retriever._fetch_k(5, 200, 100, 0, has_filter=False, will_postprocess=True) == 100
    assert Retriever._fetch_k(5, 200, 10000, 0, has_filter=False, will_postprocess=False) == 5
    assert Retriever._fetch_k(5, 200, 10000, 30, has_filter=True, will_postprocess=True) == 30


def test_eval_end_to_end_with_fake_retriever(tmp_path, fake_model):
    docs = make_docs(tmp_path)
    cfg_path = write_config(tmp_path, docs=docs)
    cfg, base = load_config(cfg_path)
    IndexBuilder(cfg, base, model=fake_model).build(progress=False)
    retr = Retriever(cfg_path, model=fake_model, load_reranker=False)
    golden = [{"q": DOCS["clean.md"][:30], "path": "docs/clean.md", "chunk_index": 0},
              {"q": DOCS["polish.md"][:30], "path": "docs/polish.md", "chunk_index": 0}]
    out = rag_eval.score(golden, lambda q, k: retr.search(q, top_k=k), top_k=10)
    assert out["file_recall_1"] == 1.0 and out["miss"] == 0


def test_vector_index_remove_from_untrained_buffer(tmp_path):
    vi = VectorIndex(tmp_path, kind="flat_sq8", dim=8, train_size=100)
    x = np.random.RandomState(0).rand(10, 8).astype(np.float32)
    vi.add(x, np.arange(10))
    assert vi.ntotal == 10 and not vi.trained
    assert vi.remove([1, 2]) == 2
    vi.finalize()
    vi.save()
    vi2 = VectorIndex.load(tmp_path)
    assert vi2.ntotal == 8 and vi2.kind == "flat_sq8"


def test_searchhit_roundtrip():
    h = SearchHit(1, 0.5, "f", "p", ".md", 0, 1, "t")
    assert h.to_dict()["path"] == "p"
