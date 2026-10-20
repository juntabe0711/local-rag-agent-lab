"""scripts/ 共通: リポジトリのルートを import パスに入れ、ログと標準出力を整える。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = str(ROOT / "config.yaml")

for _stream in (sys.stdout, sys.stderr):          # 日本語 Windows (cp932) でも --help やログが化けないように
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
