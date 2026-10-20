"""設定ファイル (config.yaml) の読み込みとパス解決。

パスの基準は 2 つだけ:
  - config.yaml のあるフォルダ ... target.folders / paths.data_root / ローカルのモデルフォルダ
  - data_root (既定 ./data) ......... 索引・状態 DB・評価の質問集・ログ・エージェント用 DB

data_root は環境変数 RAG_DATA_ROOT で差し替えられる (Google ドライブなど同期フォルダの外に
索引を置きたいとき用)。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_config(path: str | Path = "config.yaml") -> Tuple[Dict[str, Any], Path]:
    """(cfg, base_dir) を返す。BOM 付きでも読める。"""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    load_dotenv(p.parent)
    return cfg, p.parent


def load_dotenv(folder: Path) -> None:
    """folder/.env を読む (python-dotenv があれば)。シェルで設定済みの値が勝つ。"""
    env = Path(folder) / ".env"
    if not env.is_file():
        return
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    _load(dotenv_path=env, override=False)


def resolve_path(base_dir: Path, p: str | Path) -> Path:
    q = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return q if q.is_absolute() else (Path(base_dir) / q).resolve()


def data_root(cfg: Dict[str, Any], base_dir: Path) -> Path:
    env = os.environ.get("RAG_DATA_ROOT", "").strip()
    raw = env or (cfg.get("paths") or {}).get("data_root") or "./data"
    return resolve_path(base_dir, raw)


def data_path(cfg: Dict[str, Any], base_dir: Path, p: str | Path) -> Path:
    """data_root 基準のパス (絶対パスならそのまま)。"""
    return resolve_path(data_root(cfg, base_dir), p)


def model_ref(base_dir: Path, name_or_path: str) -> str:
    """ローカルフォルダがあればその絶対パス、無ければ Hugging Face のリポジトリ名として返す。"""
    s = str(name_or_path or "").strip()
    if not s:
        raise ValueError("model name_or_path is empty")
    local = resolve_path(base_dir, s)
    return str(local) if local.is_dir() else s
