"""RAG チャットのプロンプト。検索ヒットを番号付きのコンテキストにして messages を作る。

指示文は厳しすぎると (「記載が無ければ答えない」)、関連する情報があっても答えを拒むため、
- 直接の記述があればそれを根拠に
- 無くても関連情報から判断できる範囲で
- それでも足りない部分は一般知識で補い、どこからが文書に基づかないかを明示
という段階的な書き方にしている。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

DEFAULT_SYSTEM = ("あなたは文書検索アシスタントです。日本語で、正確かつ簡潔に答えてください。"
                  "根拠にした文書の番号を [1] のように示してください。")

_INSTRUCTIONS = (
    "以下のコンテキストを丁寧に読み、質問に答えてください。\n"
    "- コンテキストに直接の記述があれば、それを根拠に具体的に答えてください。\n"
    "- 直接の記述が無くても、関連する情報・周辺情報・類似事例から判断できる範囲で答えてください。\n"
    "- 複数のチャンクを横断して情報を組み合わせ、要点をまとめて答えてください。\n"
    "- コンテキストで答えられない部分は、そこで打ち切らず、あなた自身の一般知識で回答を続けてください。"
    "その際は「文書には該当が無いため一般知識で補足します」のように、どこからが文書に基づかない内容かを"
    "必ず明示してください。\n"
    "- コンテキストを根拠にした箇所には [番号] を明記してください (一般知識で補った箇所には付けないでください)。\n\n"
)

_NO_CONTEXT = (
    "文書の検索では関連する記述が見つかりませんでした。\n"
    "その旨を一言添えたうえで、あなた自身の一般知識で質問に答えてください。\n\n"
)

_RAG_DISABLED = (
    "今回は文書を参照しない設定です。あなた自身の一般知識で質問に答えてください。\n"
    "文書に基づく回答ではない点だけ一言添えてください。\n\n"
)


def _hit(h, name):
    return h.get(name) if isinstance(h, dict) else getattr(h, name)


def format_context(hits: Sequence) -> str:
    return "\n\n".join(
        f"[{i}] {_hit(h, 'path')} (chunk {_hit(h, 'chunk_index')}/{_hit(h, 'num_chunks')}, "
        f"score={float(_hit(h, 'score')):+.3f})\n{_hit(h, 'text')}"
        for i, h in enumerate(hits, 1))


def build_rag_messages(query: str, hits: Sequence, system: str = DEFAULT_SYSTEM,
                       rag_used: bool = True, history: Sequence[Dict[str, str]] = ()) -> List[Dict[str, str]]:
    if hits:
        user = _INSTRUCTIONS + "## コンテキスト\n" + format_context(hits) + "\n\n## 質問\n" + query
    else:
        user = (_NO_CONTEXT if rag_used else _RAG_DISABLED) + "## 質問\n" + query
    return ([{"role": "system", "content": system}] + list(history) +
            [{"role": "user", "content": user}])
