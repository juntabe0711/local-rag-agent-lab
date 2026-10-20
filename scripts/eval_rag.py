#!/usr/bin/env python3
"""検索の精度を測る — 質問集を作り、採点する。

    # 1) 一度だけ: コーパスから正解つきの質問を 100 問合成する (LLM が要る)
    python scripts/eval_rag.py build-golden --n 100 --seed 42

    # 2) いつでも: 採点する (設定を変えたら --label を変えてもう一度)
    python scripts/eval_rag.py run --label "baseline"
    python scripts/eval_rag.py run --label "no-reranker" --no-reranker

数字の意味
  FILE recall@k   正しい「文書」が上位 k 件に入った割合。主指標
  CHUNK recall@k  正しい「チャンク」そのものが上位 k 件に入った割合 (厳しい)
  MRR             CHUNK の平均逆順位。並べ替え (リランカー) の変化が出る
点数を比べられるのは同じ質問集どうしだけ。作り直すときは --out で別名にする。
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

from _common import DEFAULT_CONFIG, setup_logging

from ragcore import rag_eval
from ragcore.config import data_path, load_config

log = logging.getLogger("eval_rag")
FAST_FAIL_AFTER = 3          # 最初から連続でこの回数失敗したら設定の問題とみなして止める


def _paths(cfg, base, args):
    ecfg = cfg.get("rag_eval") or {}
    pcfg = cfg.get("processing") or {}
    golden = Path(getattr(args, "out", None) or getattr(args, "golden", None) or "")
    golden = golden.resolve() if str(golden) not in ("", ".") else data_path(cfg, base, ecfg.get("golden", "eval/golden_v1.jsonl"))
    return {"golden": golden,
            "state_db": data_path(cfg, base, pcfg.get("state_db", "vector_db/state.sqlite")),
            "db": data_path(cfg, base, ecfg.get("db", "eval/rag_eval.sqlite")),
            "csv": data_path(cfg, base, ecfg.get("csv", "logs/rag_eval.csv"))}


def cmd_build_golden(args) -> int:
    from ragcore.llm import LLMClient, LLMError
    cfg, base = load_config(args.config)
    paths = _paths(cfg, base, args)
    out = paths["golden"]
    if out.exists() and not args.force:
        log.error("%s は既にあります。上書きすると過去の結果と比べられない新しい基準になります。"
                  "上書きするなら --force、別名なら --out。", out)
        return 2
    rows = rag_eval.sample_chunks(paths["state_db"], args.n, min_chars=args.min_chars, max_chars=args.max_chars,
                                  rng=random.Random(args.seed) if args.seed is not None else None)
    if not rows:
        log.error("条件に合うチャンクがありません (索引が空、または長さの条件で全部外れた)。")
        return 2
    log.info("Sampled %d chunk(s) from %d file(s)", len(rows), len({r["path"] for r in rows}))
    llm = LLMClient.from_config(cfg, role="agent")
    if args.model:
        llm.model = args.model
    if args.think != "default":
        llm.think = args.think == "on"
    log.info("Question writer: %s (api=%s, think=%s)", llm.model, llm.api, llm.think)
    items, skipped, consecutive = [], 0, 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        try:
            res = llm.chat([{"role": "user", "content": rag_eval.QUESTION_PROMPT.format(text=r["text"])}],
                           options={"temperature": args.temperature})
            q = rag_eval.extract_question(res["content"])
            bad = None if len(q) >= args.min_question_chars else f"unusable question {q!r}"
        except LLMError as exc:
            q, bad = "", f"generation failed ({exc})"
        if bad:
            log.warning("[%d/%d] %s — skipped", i, len(rows), bad)
            skipped += 1
            consecutive += 1
            if consecutive >= FAST_FAIL_AFTER and not items:
                log.error("最初の %d 件が連続で失敗しました。llm.base_url / llm.model を確認してください。", consecutive)
                return 3
            continue
        consecutive = 0
        items.append(rag_eval.GoldenItem(q=q, path=r["path"], chunk_index=r["chunk_index"],
                                         ext=r["ext"], chunk_id=r["chunk_id"]))
        if args.verbose or i % 10 == 0:
            log.info("[%d/%d] %s", i, len(rows), q[:70])
    if not items:
        log.error("全件の生成に失敗しました。何も書いていません。")
        return 3
    for it in items:
        it["generator"] = llm.model
    n = rag_eval.save_golden(out, items)
    log.info("WROTE %d question(s) to %s (skipped %d, %.1fs)", n, out, skipped, time.time() - t0)
    log.info("次: 変な質問があれば行ごと消してから  python scripts/eval_rag.py run --label baseline")
    return 0


def cmd_run(args) -> int:
    from ragcore.retriever import Retriever
    cfg, base = load_config(args.config)
    paths = _paths(cfg, base, args)
    golden = rag_eval.load_golden(paths["golden"])
    if not golden:
        log.error("%s に使える質問がありません — 先に build-golden を実行してください。", paths["golden"])
        return 2
    if args.limit > 0:
        golden = golden[:args.limit]
    top_k = args.top_k or int((cfg.get("rag_eval") or {}).get("top_k", 10))
    retr = Retriever(args.config, load_reranker=not args.no_reranker)
    if args.candidates:
        retr.cfg.setdefault("retrieval", {})["candidates"] = args.candidates
    if args.mmr:
        retr.cfg.setdefault("retrieval", {}).setdefault("mmr", {})["enabled"] = True
    log.info("Scoring %d question(s) at top_k=%d against %s", len(golden), top_k, paths["golden"].name)
    t0 = time.time()
    result = rag_eval.score(golden, lambda q, k: retr.search(q, top_k=k), top_k=top_k)
    details = result.pop("details", [])
    row = rag_eval.now_row(source=args.source, label=args.label or "", golden=paths["golden"].name,
                           chunks_total=rag_eval.corpus_size(paths["state_db"]),
                           duration_sec=round(time.time() - t0, 1), note=args.note or "", **result)
    report(row, details, show_misses=args.show_misses)
    if args.no_record:
        return 0
    rid = rag_eval.record_run(paths["db"], row, csv_path=paths["csv"])
    log.info("Recorded as run id=%d (+ %s)", rid, paths["csv"])
    return 0


def report(row, details, *, show_misses: bool) -> None:
    def pct(v):
        return "-" if v is None else f"{v * 100:.1f}%"
    print("=" * 68)
    print(f"questions={row['n_questions']}  top_k={row['top_k']}  label={row.get('label')!r}  "
          f"chunks={row.get('chunks_total')}")
    print(f"FILE  recall@1={pct(row['file_recall_1'])}  recall@{row['top_k']}={pct(row['file_recall_k'])}"
          "   <- 主指標 (正しい文書が上位に来たか)")
    print(f"CHUNK recall@1={pct(row['recall_1'])}  recall@5={pct(row['recall_5'])}  "
          f"recall@{row['top_k']}={pct(row['recall_k'])}  MRR={'-' if row['mrr'] is None else format(row['mrr'], '.3f')}")
    print(f"miss={row['miss']} (正解の文書が 1 件も出なかった質問)   latency="
          f"{'-' if row['latency_ms'] is None else format(row['latency_ms'], '.0f')} ms/query")
    print("=" * 68)
    if show_misses:
        for d in details:
            if d["file_rank"] is None:
                print("MISS  " + (d["q"] or "")[:80])
                print(f"      期待: {d['path']} #{d['chunk_index']}" + (f"  ({d['error']})" if d.get("error") else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-golden", help="コーパスから正解つきの質問を合成する (LLM を使う)")
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--out", default=None, help="出力 JSONL (既定: rag_eval.golden)")
    b.add_argument("--min-chars", type=int, default=200)
    b.add_argument("--max-chars", type=int, default=2000)
    b.add_argument("--min-question-chars", type=int, default=8)
    b.add_argument("--seed", type=int, default=None, help="抽出の乱数シード (再現用)")
    b.add_argument("--temperature", type=float, default=None)
    b.add_argument("--model", default="", help="質問を書かせるモデル (既定: llm.agent_model / llm.model)")
    b.add_argument("--think", choices=["default", "on", "off"], default="default",
                   help="推論モデルの思考 (Ollama のみ)。off にすると速い")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_build_golden)
    r = sub.add_parser("run", help="質問集で採点する")
    r.add_argument("--golden", default=None)
    r.add_argument("--top-k", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--label", default="")
    r.add_argument("--note", default="")
    r.add_argument("--source", default="manual", choices=["manual", "scheduled"])
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--show-misses", action="store_true")
    r.add_argument("--no-reranker", action="store_true", help="リランカーを外して測る (比較用)")
    r.add_argument("--candidates", type=int, default=0, help="retrieval.candidates を上書き (比較用)")
    r.add_argument("--mmr", action="store_true", help="MMR を有効にして測る (比較用)")
    r.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
