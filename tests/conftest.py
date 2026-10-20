"""テスト共通: GPU・モデル・LLM を使わずに動かすための偽物。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTokenizer:
    """1 文字 = 1 トークン。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class FakeEmbedder:
    """文字 2-gram のハッシュで作る決まったベクトル (同じ文字を含むほど近い)。"""
    dim = 64

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, normalize_embeddings=True, **_):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for a, b in zip(t, t[1:] + " "):
                h = int(hashlib.md5((a + b).encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if normalize_embeddings and n:
                out[i] /= n
        return out


@pytest.fixture
def fake_model():
    return FakeEmbedder()


def write_config(tmp: Path, *, docs: Path, index_type: str = "flat", reranker: bool = False) -> Path:
    cfg = f"""
paths: {{data_root: "{(tmp / 'data').as_posix()}"}}
model: {{name_or_path: "unused", device: "cpu", fp16: false}}
target: {{folders: ["{docs.as_posix()}"], include_extensions: [".txt", ".md"]}}
chunking: {{chunk_size: 40, chunk_overlap: 8}}
retrieval: {{candidates: 50, min_score: null, reranker: {{enabled: {str(reranker).lower()}}}, mmr: {{enabled: false}}}}
vector_db: {{path: "vector_db", index_type: "{index_type}", train_size: 1000}}
processing: {{embed_batch_chunks: 7, state_db: "vector_db/state.sqlite"}}
llm: {{base_url: "http://127.0.0.1:9/v1", model: "fake"}}
rag_eval: {{golden: "eval/golden.jsonl", top_k: 10}}
"""
    p = tmp / "config.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p
