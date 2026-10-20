#!/usr/bin/env python3
"""公開データのコーパスを用意する: 日本語版ウィキペディアの記事をテキストで保存する。

    python scripts/fetch_wikipedia.py                 # 既定のカテゴリから約 400 記事
    python scripts/fetch_wikipedia.py --max 1000 --categories 半導体 機械学習

保存先は data/corpus/<記事名>.txt (1 行目に題名と URL)。記事の本文は CC BY-SA 4.0 なので、
リポジトリには含めず、各自がこのスクリプトで取得する (.gitignore 済み)。
同じカテゴリ・同じ件数なら同じ記事の集合になる (並び順を固定している)。ただし記事は日々
編集されるので、取得日が違えば本文は少し変わる — 取得日は data/corpus_manifest.json に残る。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from _common import ROOT, setup_logging

API = "https://ja.wikipedia.org/w/api.php"
UA = os.environ.get("WIKI_USER_AGENT") or "local-rag-agent-lab/0.1 (personal RAG evaluation demo; sequential, rate-limited)"
DELAY = 1.0              # 1 リクエストごとの待ち (秒)。API の利用ルールに沿って逐次・低速で取る
DEFAULT_CATEGORIES = ["半導体", "半導体製造", "結晶学", "流体力学", "電磁気学", "品質管理",
                      "機械学習", "自然言語処理", "統計学", "材料工学"]


def _get(sess: requests.Session, params: dict) -> dict:
    params = dict(params, format="json", formatversion=2)
    for attempt in range(8):
        r = sess.get(API, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            # 混んでいるときはサーバの指示 (Retry-After) に従って待つ
            wait = float(r.headers.get("Retry-After") or 0) or min(60.0, 5.0 * 2 ** attempt)
            print(f"\n  HTTP {r.status_code} — waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(DELAY)
        return r.json()
    r.raise_for_status()
    return {}


def category_titles(sess, category: str, depth: int, limit: int) -> list[str]:
    """カテゴリ内の記事名 (名前空間 0)。depth まで下位カテゴリもたどる。"""
    out, seen_cats = [], set()
    queue = [(category, 0)]
    while queue and len(out) < limit:
        cat, d = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont = {}
        while len(out) < limit:
            data = _get(sess, {"action": "query", "list": "categorymembers", "cmtitle": "Category:" + cat,
                               "cmlimit": 500, "cmtype": "page|subcat", **cont})
            for m in (data.get("query") or {}).get("categorymembers", []):
                if m.get("ns") == 0:
                    out.append(m["title"])
                elif m.get("ns") == 14 and d < depth:
                    queue.append((m["title"].split(":", 1)[1], d + 1))
            if "continue" not in data:
                break
            cont = data["continue"]
    return out[:limit]


def fetch_extracts(sess, titles: list[str]) -> dict[str, str]:
    """本文 (プレーンテキスト)。全文の抽出は 1 回 1 記事までなので 1 件ずつ取る。"""
    out = {}
    for t in titles:
        data = _get(sess, {"action": "query", "prop": "extracts", "explaintext": 1,
                           "exsectionformat": "plain", "redirects": 1, "titles": t})
        for page in (data.get("query") or {}).get("pages", []):
            text = page.get("extract") or ""
            if text:
                out[page["title"]] = text
    return out


def page_lengths(sess, titles: list[str]) -> dict[str, int]:
    """記事の長さ (ウィキ記法のバイト数)。1 回 50 記事まで取れるので、本文を取る前の足切りに使う。"""
    out = {}
    for i in range(0, len(titles), 50):
        data = _get(sess, {"action": "query", "prop": "info", "titles": "|".join(titles[i:i + 50])})
        for page in (data.get("query") or {}).get("pages", []):
            if not page.get("missing"):
                out[page["title"]] = int(page.get("length") or 0)
    return out


def safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", title)[:120]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "corpus"))
    p.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    p.add_argument("--per-category", type=int, default=80, help="1 カテゴリから集める記事名の上限")
    p.add_argument("--depth", type=int, default=1, help="下位カテゴリをたどる深さ")
    p.add_argument("--max", type=int, default=400, help="保存する記事数の上限")
    p.add_argument("--min-chars", type=int, default=1500, help="これより短い記事は捨てる")
    p.add_argument("--min-bytes", type=int, default=8000,
                   help="本文を取る前の足切り (ウィキ記法のバイト数。短い記事に無駄なリクエストを使わない)")
    args = p.parse_args(argv)
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA

    titles: list[str] = []
    for cat in args.categories:
        got = category_titles(sess, cat, args.depth, args.per_category)
        print(f"  Category:{cat}: {len(got)} titles")
        titles += [t for t in got if t not in titles]
    titles = sorted(set(titles))
    lengths = page_lengths(sess, titles)
    titles = [t for t in titles if lengths.get(t, 0) >= args.min_bytes]
    print(f"{len(titles)} candidate titles (>= {args.min_bytes} bytes)")

    saved, manifest = 0, []
    for title in titles:
        if saved >= args.max:
            break
        path = out / f"{safe_name(title)}.txt"
        if path.is_file():                         # 途中から再開できるように、取得済みは飛ばす
            text = path.read_text(encoding="utf-8").split("\n", 3)[-1]
        else:
            text = fetch_extracts(sess, [title]).get(title, "")
            if len(text) < args.min_chars:
                continue
            url = "https://ja.wikipedia.org/wiki/" + title.replace(" ", "_")
            path.write_text(f"{title}\n{url}\n\n{text}\n", encoding="utf-8")
        manifest.append({"title": title, "chars": len(text)})
        saved += 1
        print(f"  saved {saved}/{args.max}", end="\r", flush=True)
    (out.parent / (out.name + "_manifest.json")).write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "ja.wikipedia.org (CC BY-SA 4.0)", "categories": args.categories,
        "articles": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {saved} articles to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
