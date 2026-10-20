"""ファイルからテキストを取り出す。

対応: PDF / Word / Excel / PowerPoint (表・グラフの数値・ノートを含む) / Jupyter / テキスト系。
重い依存 (pypdf, python-docx, openpyxl, python-pptx, nbformat) は使うときだけ import する。
"""
from __future__ import annotations

import codecs
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_STREAM_DECODE_THRESHOLD = 8 * 1024 * 1024   # 8 MB 以上はストリームで decode
_STREAM_READ_CHUNK = 1024 * 1024


def _decode_fallback(raw: bytes) -> str:
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        return raw.decode("utf-8", errors="replace")
    best = from_bytes(raw).best()
    return str(best) if best is not None else raw.decode("utf-8", errors="replace")


def _read_text_streaming(path: Path) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts: list[str] = []
    try:
        with path.open("rb") as f:
            while True:
                buf = f.read(_STREAM_READ_CHUNK)
                if not buf:
                    break
                parts.append(decoder.decode(buf))
            tail = decoder.decode(b"", final=True)
            if tail:
                parts.append(tail)
    except UnicodeDecodeError:
        return _decode_fallback(path.read_bytes())
    return "".join(parts)


def extract_text_file(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size == 0:
        return ""
    if size >= _STREAM_DECODE_THRESHOLD:
        return _read_text_streaming(path)
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _decode_fallback(raw)


# pypdf が埋め込みフォントを Unicode にできないときに出す "(cid:NNN)"
_CID_RE = re.compile(r"\(cid:\d+\)")


def sanitize_text(text: str) -> str:
    """NUL・制御文字・DEL を落とす (\\t \\n \\r と CJK を含む印字可能文字は残す)。"""
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in "\t\n\r":
            out.append(ch)
            continue
        o = ord(ch)
        if o < 32 or o == 0x7F or (0x80 <= o <= 0x9F):
            continue
        out.append(ch)
    return "".join(out)


def looks_garbled(text: str, *, min_len: int = 40, threshold: float = 0.10) -> bool:
    """抽出結果が文字化け (CID フォント・置換文字だらけ) に見えるか。空も True。"""
    if not text or not text.strip():
        return True
    if _CID_RE.search(text):
        return True
    n = len(text)
    if n < min_len:
        return False
    bad = 0
    for ch in text:
        o = ord(ch)
        if o == 0xFFFD or (0xE000 <= o <= 0xF8FF) or (0xF0000 <= o <= 0xFFFFD) or o < 9:
            bad += 1
    return (bad / n) > threshold


def _extract_pdf_pypdf(path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_pdf_fitz(path: Path) -> str:
    import fitz  # PyMuPDF (任意)
    parts = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            t = page.get_text("text")
            if t:
                parts.append(t)
    return "\n".join(parts)


def _extract_pdf_pdfminer(path: Path) -> str:
    from pdfminer.high_level import extract_text as _pm_extract  # 任意
    return _pm_extract(str(path)) or ""


def extract_pdf(path: Path) -> str:
    """まず pypdf。文字化けに見えるときだけ PyMuPDF → pdfminer を試す
    (正常な PDF の抽出結果を変えないため — 変わると内容ハッシュが変わり再埋め込みになる)。"""
    text = _extract_pdf_pypdf(path)
    if not looks_garbled(text):
        return text
    for fn in (_extract_pdf_fitz, _extract_pdf_pdfminer):
        try:
            alt = fn(path)
        except ImportError:
            continue
        except Exception as e:
            log.debug("PDF fallback %s failed for %s: %s", fn.__name__, path, e)
            continue
        if alt and not looks_garbled(alt):
            log.info("PDF %s: pypdf output garbled -> recovered with %s", path, fn.__name__)
            return sanitize_text(alt)
    return sanitize_text(text)


def extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text for c in row.cells if c.text]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(str(path), data_only=True, read_only=True)
    try:
        parts = []
        for sheet in wb.worksheets:
            parts.append(f"=== Sheet: {sheet.title} ===")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    parts.append("\t".join(cells))
        return "\n".join(parts)
    finally:
        wb.close()


def _pptx_table_lines(shape) -> list[str]:
    lines = ["[Table]"]
    for row in shape.table.rows:
        cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
        if any(cells):
            lines.append("\t".join(cells))
    return lines if len(lines) > 1 else []


def _pptx_chart_lines(shape) -> list[str]:
    """グラフの題と系列ごとの「項目=値」。数値はグラフ内の埋め込みブックにしか無いため。"""
    chart = shape.chart
    lines = ["[Chart]"]
    try:
        if chart.has_title:
            title = (chart.chart_title.text_frame.text or "").strip()
            if title:
                lines.append(f"Title: {title}")
    except Exception:
        pass
    try:
        cats = [str(c) for c in chart.plots[0].categories]
    except Exception:
        cats = []
    try:
        for series in chart.series:
            name = str(getattr(series, "name", "") or "series")
            values = ["" if v is None else str(v) for v in series.values]
            if cats and len(cats) == len(values):
                body = ", ".join(f"{c}={v}" for c, v in zip(cats, values))
            else:
                body = ", ".join(values)
            if body:
                lines.append(f"{name}: {body}")
    except Exception:
        pass
    return lines if len(lines) > 1 else []


def extract_pptx(path: Path) -> str:
    """スライドの文字・表・グラフの数値・ノートをスライド順に。"""
    from pptx import Presentation
    prs = Presentation(str(path))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"=== Slide {i} ===")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    txt = "".join(run.text for run in para.runs)
                    if txt:
                        parts.append(txt)
            try:
                if getattr(shape, "has_table", False):
                    parts.extend(_pptx_table_lines(shape))
            except Exception:
                log.debug("pptx table extract failed", exc_info=True)
            try:
                if getattr(shape, "has_chart", False):
                    parts.extend(_pptx_chart_lines(shape))
            except Exception:
                log.debug("pptx chart extract failed", exc_info=True)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
            if notes:
                parts.append(f"[Notes] {notes}")
    return "\n".join(parts)


def extract_ipynb(path: Path) -> str:
    import json
    nb = json.loads(path.read_text(encoding="utf-8-sig"))
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") in ("markdown", "code"):
            src = cell.get("source")
            src = "".join(src) if isinstance(src, list) else (src or "")
            if src.strip():
                parts.append(src)
    return "\n\n".join(parts)


EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".xlsx": extract_xlsx,
    ".pptx": extract_pptx,
    ".ipynb": extract_ipynb,
}


def extract(path: Path) -> str | None:
    """失敗したら None (呼び出し側はそのファイルを飛ばす)。"""
    fn = EXTRACTORS.get(path.suffix.lower(), extract_text_file)
    try:
        return fn(path)
    except Exception as e:
        log.warning("Failed to extract %s: %s", path, e)
        return None
