"""make_chart のグラフ描画 (matplotlib → PNG)。

LLM には Python を書かせない。LLM が出すのは「種類・x 列・y 列・系列」の指定 (spec) だけで、
描画はこのファイルの決まったコードが行う (任意のコード実行を避けるため)。

spec = {"type": "bar"|"barh"|"line"|"scatter"|"hist", "title": str, "x": 列名,
        "y": 列名 or [列名…], "series": 列名 (省略可), "xlabel": str, "ylabel": str}
"""
from __future__ import annotations

import base64
import datetime as _dt
import io
import math
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

TYPES = ("bar", "barh", "line", "scatter", "hist")
MAX_CATEGORIES = 40
MAX_SERIES = 8
JP_FONTS = ("Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Noto Sans CJK JP",
            "Noto Sans JP", "IPAexGothic", "Hiragino Sans")
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_LOCK = threading.Lock()          # pyplot はスレッド安全でない


class ChartError(ValueError):
    """指定の誤り (列が無い等)。LLM が読んで直せる文言にする。"""


def _col(name: Any, columns: Sequence[str], what: str) -> str:
    n = str(name or "").strip().strip('"')
    if n in columns:
        return n
    low = {c.lower(): c for c in columns}
    if n.lower() in low:
        return low[n.lower()]
    raise ChartError("%s の列 '%s' がデータにありません。使える列: %s" % (what, n, ", ".join(columns)))


def normalize(spec: Dict[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    t = str(spec.get("type") or "bar").lower().strip()
    t = {"column": "bar", "hbar": "barh", "histogram": "hist", "line_chart": "line"}.get(t, t)
    if t not in TYPES:
        raise ChartError("type は %s のどれか (指定: %s)" % (" / ".join(TYPES), t))
    ys = spec.get("y")
    ys = ys if isinstance(ys, list) else ([ys] if ys else [])
    out: Dict[str, Any] = {"type": t, "title": str(spec.get("title") or "")[:80],
                           "xlabel": str(spec.get("xlabel") or "")[:40],
                           "ylabel": str(spec.get("ylabel") or "")[:40]}
    if t == "hist":
        col = ys[0] if ys else spec.get("x")
        if not col:
            raise ChartError("hist は y (または x) に数値の列を 1 つ指定する")
        out["y"], out["x"] = [_col(col, columns, "y")], None
    else:
        if not spec.get("x"):
            raise ChartError("x (横軸の列) を指定する")
        if not ys:
            raise ChartError("y (縦軸の列) を指定する")
        out["x"] = _col(spec["x"], columns, "x")
        out["y"] = [_col(y, columns, "y") for y in ys[:MAX_SERIES]]
    out["series"] = _col(spec["series"], columns, "series") if spec.get("series") else None
    return out


def check_data(spec: Dict[str, Any], columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """y に数値が 1 つも無い指定を断る (文字列の列を y にすると空のグラフになる)。"""
    idx = {c: i for i, c in enumerate(columns)}
    numeric = [c for c in columns if any(not math.isnan(_num(r[idx[c]])) for r in rows[:200])]
    for y in spec["y"]:
        if not any(not math.isnan(_num(r[idx[y]])) for r in rows):
            raise ChartError("y の列 '%s' に数値がありません。数値の列を y にする (候補: %s)。"
                             "文字列の列は x か series に使う。" % (y, ", ".join(numeric) or "なし"))


def _num(v: Any) -> float:
    if v is None or isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return math.nan


def _date(v: Any) -> Optional[_dt.datetime]:
    if isinstance(v, _dt.datetime):
        d = v
    elif isinstance(v, _dt.date):
        d = _dt.datetime(v.year, v.month, v.day)
    else:
        s = str(v if v is not None else "").strip().replace("/", "-")
        d = None
        if re.fullmatch(r"\d{4}-\d{1,2}", s):
            y, m = s.split("-")
            d = _dt.datetime(int(y), int(m), 1) if 1 <= int(m) <= 12 else None
        elif re.match(r"\d{4}-\d{1,2}-\d{1,2}", s):
            s = re.sub(r"^(\d{4})-(\d{1,2})-(\d{1,2})",
                       lambda m: "%s-%02d-%02d" % (m.group(1), int(m.group(2)), int(m.group(3))), s)
            for cand in (s.replace(" ", "T", 1), s[:19].replace(" ", "T", 1), s[:10]):
                try:
                    d = _dt.datetime.fromisoformat(cand.rstrip("Z"))
                    break
                except ValueError:
                    continue
        if d is None:
            return None
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    return d if 1990 <= d.year <= 2100 else None


def _x_kind(values: List[Any]) -> str:
    vals = [v for v in values if v not in (None, "")]
    if vals and all(_date(v) for v in vals):
        return "date"
    if vals and all(not math.isnan(_num(v)) for v in vals):
        return "num"
    return "cat"


def _setup_font(plt) -> None:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for f in JP_FONTS:
        if f in have:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False


def render_png(spec: Dict[str, Any], columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    """検証済み spec (normalize の戻り値) で PNG を描く。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with _LOCK:
        _setup_font(plt)
        idx = {c: i for i, c in enumerate(columns)}
        fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=110)
        try:
            ax.set_prop_cycle(color=list(SERIES_COLORS))
            t = spec["type"]
            if t == "hist":
                vals = [v for v in (_num(r[idx[spec["y"][0]]]) for r in rows) if not math.isnan(v)]
                if not vals:
                    raise ChartError("hist の列に数値がありません")
                ax.hist(vals, bins=min(40, max(8, int(math.sqrt(len(vals))))), color=SERIES_COLORS[0], alpha=.85)
                ax.set_xlabel(spec["xlabel"] or spec["y"][0])
                ax.set_ylabel(spec["ylabel"] or "件数")
            else:
                groups = _groups(spec, idx, rows)
                if t in ("bar", "barh"):
                    _bars(ax, groups, horizontal=(t == "barh"))
                else:
                    _lines(ax, groups, scatter=(t == "scatter"))
                ax.set_xlabel(spec["xlabel"] or ("" if t == "barh" else spec["x"]))
                ax.set_ylabel(spec["ylabel"] or ("" if t == "barh" else " / ".join(spec["y"])))
                if len(groups) > 1:
                    ax.legend(fontsize=8, loc="best")
            if spec["title"]:
                ax.set_title(spec["title"], fontsize=12)
            ax.grid(True, alpha=.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            return buf.getvalue()
        finally:
            plt.close(fig)


def _groups(spec, idx, rows) -> List[Tuple[str, List[Any], List[float]]]:
    """[(系列名, x の値, y の値)]。series 指定なら系列ごと、y が複数なら列ごと。"""
    x = spec["x"]
    if spec["series"]:
        by: Dict[str, Tuple[List[Any], List[float]]] = {}
        for r in rows:
            key = str(r[idx[spec["series"]]])
            if key not in by and len(by) >= MAX_SERIES:
                continue
            by.setdefault(key, ([], []))
            by[key][0].append(r[idx[x]])
            by[key][1].append(_num(r[idx[spec["y"][0]]]))
        return [(k, v[0], v[1]) for k, v in by.items()]
    return [(y, [r[idx[x]] for r in rows], [_num(r[idx[y]]) for r in rows]) for y in spec["y"]]


def _bars(ax, groups, horizontal: bool) -> None:
    cats: List[str] = []
    for _, xs, _ in groups:
        for v in xs:
            if str(v) not in cats:
                cats.append(str(v))
    cats = cats[:MAX_CATEGORIES]
    n = len(groups)
    width = 0.8 / n
    for i, (name, xs, ys) in enumerate(groups):
        m = {str(a): b for a, b in zip(xs, ys)}
        pos = [c + (i - (n - 1) / 2) * width for c in range(len(cats))]
        vals = [m.get(c, math.nan) for c in cats]
        if horizontal:
            ax.barh(pos, vals, height=width, label=name)
        else:
            ax.bar(pos, vals, width=width, label=name)
    ticks = list(range(len(cats)))
    if horizontal:
        ax.set_yticks(ticks)
        ax.set_yticklabels(cats, fontsize=8)
        ax.invert_yaxis()
    else:
        ax.set_xticks(ticks)
        long = len(cats) > 8 or max((len(c) for c in cats), default=0) > 6
        ax.set_xticklabels(cats, fontsize=8, rotation=45 if long else 0, ha="right" if long else "center")


def _lines(ax, groups, scatter: bool) -> None:
    kind = _x_kind([v for _, xs, _ in groups for v in xs])
    for name, xs, ys in groups:
        if kind == "date":
            xv: List[Any] = [_date(v) for v in xs]
        elif kind == "num":
            xv = [_num(v) for v in xs]
        else:
            xv = [str(v) for v in xs]
        pts = [(a, b) for a, b in zip(xv, ys) if a is not None and not math.isnan(b)]
        if kind != "cat":
            pts.sort(key=lambda p: p[0])
        if not pts:
            continue
        a, b = zip(*pts)
        if scatter:
            ax.scatter(a, b, s=12, alpha=.7, label=name)
        else:
            ax.plot(a, b, marker="o" if len(a) <= 40 else None, markersize=3, linewidth=1.4, label=name)
    if kind == "date":
        ax.figure.autofmt_xdate()
    elif kind == "cat":
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")


def describe(spec: Dict[str, Any], columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """LLM に返す 1 行の要約 (描いたもの・点数・y の範囲)。"""
    idx = {c: i for i, c in enumerate(columns)}
    parts = ["%s のグラフ「%s」を表示した (%d 行)" % (spec["type"], spec["title"] or "無題", len(rows))]
    for y in spec["y"][:4]:
        vals = [v for v in (_num(r[idx[y]]) for r in rows) if not math.isnan(v)]
        if vals:
            parts.append("%s: 最小 %s / 最大 %s" % (y, _fmt(min(vals)), _fmt(max(vals))))
    return "。".join(parts)


def _fmt(v: float) -> str:
    return ("%d" % v) if float(v).is_integer() else ("%.4g" % v)


def to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
