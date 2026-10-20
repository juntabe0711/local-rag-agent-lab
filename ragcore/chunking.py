"""埋め込みモデル自身のトークナイザでチャンクに分ける (既定 512 トークン・重なり 64)。

文字数ではなくトークン数で切るので、日本語 (≈1 トークン/字) と英語 (≈0.3 トークン/字) が混ざっても
1 チャンクがモデルの入力長に収まる。巨大なファイルは文字の窓ごとにトークン化して流す
(一度に数千万トークンを Python のリストにしない)。
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# 1 回にトークン化する文字数。bge-m3 の model_max_length (8192) を どの言語でも超えない長さ
_TOKENIZE_CHAR_WINDOW = 8000
_CHUNK_COUNT_WARN = 2000


def _slide_tokens(tokens: list[int], tokenizer, chunk_size: int, overlap: int) -> list[str]:
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return [decoded] if decoded else []
    step = max(1, chunk_size - overlap)
    out: list[str] = []
    for start in range(0, len(tokens), step):
        sub = tokens[start:start + chunk_size]
        if not sub:
            break
        decoded = tokenizer.decode(sub, skip_special_tokens=True).strip()
        if decoded:
            out.append(decoded)
        if start + chunk_size >= len(tokens):
            break
    return out


def chunk_by_tokens(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Rust の高速トークナイザは孤立サロゲートを受け付けない (PDF 抽出でまれに混ざる)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

    if len(text) <= _TOKENIZE_CHAR_WINDOW:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return _slide_tokens(tokens, tokenizer, chunk_size, overlap)

    chunks: list[str] = []
    buffer: list[int] = []
    step = max(1, chunk_size - overlap)
    pos, n = 0, len(text)
    while pos < n:
        window = text[pos:pos + _TOKENIZE_CHAR_WINDOW]
        pos += _TOKENIZE_CHAR_WINDOW
        buffer.extend(tokenizer.encode(window, add_special_tokens=False))
        while len(buffer) >= chunk_size:
            decoded = tokenizer.decode(buffer[:chunk_size], skip_special_tokens=True).strip()
            if decoded:
                chunks.append(decoded)
            del buffer[:step]
    if buffer:
        decoded = tokenizer.decode(buffer, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
    if len(chunks) >= _CHUNK_COUNT_WARN:
        log.warning("File produced %d chunks — check whether it should be indexed.", len(chunks))
    return chunks
