#!/usr/bin/env python3
"""コマンドラインで検索する (LLM は使わない)。

    python scripts/query.py "CZ法で酸素濃度が決まる仕組み" --top-k 5
"""
from __future__ import annotations

import argparse
import sys
import time

from _common import DEFAULT_CONFIG, setup_logging

from ragcore.retriever import Retriever


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="検索文 (省略すると対話モード)")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ext", nargs="*", help="拡張子で絞る (例: .md .pdf)")
    p.add_argument("--path", default=None, help="パスの部分一致で絞る")
    args = p.parse_args(argv)
    setup_logging()
    r = Retriever(args.config)

    def run(q: str) -> None:
        t0 = time.time()
        hits = r.search(q, top_k=args.top_k, ext_filter=args.ext, path_contains=args.path)
        print(f"\n{len(hits)} hits ({(time.time() - t0) * 1000:.0f} ms)")
        for i, h in enumerate(hits, 1):
            text = " ".join(h.text.split())
            print(f"[{i}] {h.score:.3f} ({h.score_type}, dense {h.dense_score:.3f})  "
                  f"{h.path}  #{h.chunk_index}/{h.num_chunks}")
            print("    " + text[:200] + ("…" if len(text) > 200 else ""))

    if args.query:
        run(args.query)
        return 0
    while True:
        try:
            q = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q:
            run(q)


if __name__ == "__main__":
    sys.exit(main())
