#!/usr/bin/env python3
"""索引を作る / 差分で更新する。

    python scripts/build_index.py              # 差分更新 (変わったファイルだけ)
    python scripts/build_index.py --rebuild    # 消して作り直す
"""
from __future__ import annotations

import argparse
import json
import sys

from _common import DEFAULT_CONFIG, setup_logging

from ragcore.config import load_config
from ragcore.indexer import IndexBuilder


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--rebuild", action="store_true", help="索引と状態 DB を消して作り直す")
    p.add_argument("--limit", type=int, default=0, help="先頭 N ファイルだけ (動作確認用)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    cfg, base = load_config(args.config)
    builder = IndexBuilder(cfg, base)
    if args.rebuild:
        builder.rebuild_reset()
    result = builder.build(limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
