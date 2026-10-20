"""Web アプリ (FastAPI)。RAG チャット・エージェント・検索・評価結果を 1 つの画面で使う。

    python -m app.server                 # http://127.0.0.1:8000
    python -m app.server --port 9000

API (応答の逐次表示は NDJSON = 1 行 1 JSON):
    GET  /api/health                     索引・LLM・DB の状態
    POST /api/search   {query, top_k?, ext?, path?}            → {hits}
    POST /api/chat     {message, history?, use_rag?, top_k?}   → NDJSON: sources → thinking/content … → done
    POST /api/agent    {message, history?}                     → NDJSON: agent / step / observation / chart / …
    GET  /api/eval/runs                  評価 (scripts/eval_rag.py run) の記録
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException                              # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse          # noqa: E402
from fastapi.staticfiles import StaticFiles                            # noqa: E402
from pydantic import BaseModel, Field                                  # noqa: E402

from ragcore import rag_eval                                           # noqa: E402
from ragcore.config import data_path, load_config, resolve_path        # noqa: E402
from ragcore.llm import LLMClient, LLMError                            # noqa: E402
from ragcore.prompts import build_rag_messages                         # noqa: E402
from agent.agent import TRUNCATED_MSG                                  # noqa: E402

log = logging.getLogger("app")
STATIC = Path(__file__).resolve().parent / "static"


class SearchReq(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)
    ext: Optional[List[str]] = None
    path: Optional[str] = None


class ChatReq(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    use_rag: bool = True
    top_k: Optional[int] = Field(None, ge=1, le=20)


class AgentReq(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class State:
    """重いもの (モデル・索引・カタログ) は最初に使うときに 1 回だけ読む。"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg, self.base = load_config(config_path)
        self._lock = threading.Lock()
        self._retriever = None
        self._retriever_error = ""
        self._catalog = None

    @property
    def retriever(self):
        with self._lock:
            if self._retriever is None and not self._retriever_error:
                try:
                    from ragcore.retriever import Retriever
                    self._retriever = Retriever(self.config_path)
                except Exception as e:                          # noqa: BLE001
                    log.exception("retriever failed to load")
                    self._retriever_error = str(e)
            return self._retriever

    def llm(self, role: str = "chat") -> LLMClient:
        return LLMClient.from_config(self.cfg, role=role)

    def db_path(self) -> Path:
        return data_path(self.cfg, self.base, (self.cfg.get("agent") or {}).get("db_path", "factory.sqlite"))

    def reader(self):
        from agent.sqldb import SQLiteReader
        a = self.cfg.get("agent") or {}
        return SQLiteReader(self.db_path(), timeout_sec=float(a.get("query_timeout_sec", 20)),
                            max_rows=int(a.get("max_rows", 50)))

    @property
    def catalog(self):
        with self._lock:
            if self._catalog is None and self.db_path().is_file():
                from agent.sqldb import Catalog
                desc = (self.cfg.get("agent") or {}).get("descriptions")
                r = self.reader()
                try:
                    self._catalog = Catalog.build(r, resolve_path(self.base, desc) if desc else None)
                finally:
                    r.close()
            return self._catalog


def ndjson(events: Iterator[Dict[str, Any]]) -> Iterator[bytes]:
    try:
        for ev in events:
            yield (json.dumps(ev, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    except LLMError as e:
        yield (json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n").encode("utf-8")
    except Exception as e:                                      # noqa: BLE001
        log.exception("stream failed")
        yield (json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
               + "\n").encode("utf-8")


def create_app(config_path: str) -> FastAPI:
    st = State(config_path)
    app = FastAPI(title="local-rag-agent-lab", docs_url="/api/docs", redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        out: Dict[str, Any] = {}
        r = st.retriever
        out["retrieval"] = r.stats() if r else {"error": st._retriever_error or "not loaded"}
        try:
            out["llm"] = st.llm().health()
        except Exception as e:                                  # noqa: BLE001
            out["llm"] = {"error": str(e), "default_model": (st.cfg.get("llm") or {}).get("model")}
        cat = st.catalog
        out["db"] = {"tables": len(cat) if cat else 0, "path": st.db_path().name}
        return out

    @app.post("/api/search")
    def search(req: SearchReq):
        r = st.retriever
        if r is None:
            raise HTTPException(503, st._retriever_error or "retriever not available")
        hits = r.search(req.query, top_k=req.top_k, ext_filter=req.ext, path_contains=req.path)
        return {"hits": [h.to_dict() for h in hits]}

    @app.post("/api/chat")
    def chat(req: ChatReq):
        lcfg = st.cfg.get("llm") or {}
        top_k = req.top_k or int(lcfg.get("rag_top_k", 5))

        def gen():
            hits = []
            if req.use_rag:
                r = st.retriever
                if r is None:
                    yield {"type": "warn", "message": "索引が読めないため、文書を参照せずに答えます。"}
                else:
                    hits = r.search(req.message, top_k=top_k)
            yield {"type": "sources", "hits": [dict(h.to_dict(), n=i) for i, h in enumerate(hits, 1)]}
            history = [m for m in req.history[-8:] if m.get("role") in ("user", "assistant")]
            msgs = build_rag_messages(req.message, hits, system=lcfg.get("system_prompt") or "",
                                      rag_used=req.use_rag, history=history)
            for ev in st.llm().chat_stream(msgs, options={"num_predict": int(lcfg.get("num_predict", 8192))}):
                if ev["type"] == "error":
                    yield {"type": "error", "message": ev["text"]}
                    return
                if ev["type"] in ("content", "thinking"):
                    yield ev
                elif ev["type"] == "done" and ev["text"] == "length":
                    yield {"type": "warn", "message": TRUNCATED_MSG}
            yield {"type": "done"}

        return StreamingResponse(ndjson(gen()), media_type="application/x-ndjson")

    @app.post("/api/agent")
    def run_agent(req: AgentReq):
        from agent.agent import Agent
        a = st.cfg.get("agent") or {}
        lcfg = st.cfg.get("llm") or {}
        r = st.retriever

        def search(q, k, path_contains=None):
            return r.search(q, top_k=k, path_contains=path_contains)

        cat = st.catalog
        ag = Agent(st.llm("agent"), search=search if r else None, catalog=cat,
                   reader_factory=st.reader if cat else None,
                   max_steps=int(a.get("max_steps", 8)), cause_max_steps=int(a.get("cause_max_steps", 12)),
                   obs_chars=int(a.get("obs_chars", 3500)), step_predict=int(a.get("step_predict", 2048)),
                   num_ctx=int(lcfg.get("num_ctx", 16384)), num_predict=int(a.get("num_predict", 4096)),
                   est_chars_per_token=float(lcfg.get("est_chars_per_token", 1.5)),
                   chart_max_rows=int(a.get("chart_max_rows", 5000)))

        def gen():
            yield from ag.run(req.message, req.history)
            yield {"type": "done"}

        return StreamingResponse(ndjson(gen()), media_type="application/x-ndjson")

    @app.get("/api/eval/runs")
    def eval_runs(limit: int = 30):
        db = data_path(st.cfg, st.base, (st.cfg.get("rag_eval") or {}).get("db", "eval/rag_eval.sqlite"))
        return {"runs": rag_eval.list_runs(db, limit=limit)}

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="local-rag-agent-lab web app")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg, _ = load_config(args.config)
    web = cfg.get("web") or {}
    import uvicorn
    uvicorn.run(create_app(args.config), host=args.host or web.get("host", "127.0.0.1"),
                port=args.port or int(web.get("port", 8000)), log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
