from ragcore.llm import _ThinkSplitter, extract_json, strip_think
from ragcore.prompts import build_rag_messages


def _run(chunks):
    s = _ThinkSplitter()
    out = []
    for c in chunks:
        out += list(s.feed(c))
    out += list(s.flush())
    merged = {"content": "", "thinking": ""}
    for e in out:
        merged[e["type"]] += e["text"]
    return merged


def test_think_splitter_across_chunk_boundaries():
    m = _run(["<thi", "nk>考え", "中</th", "ink>答え", "です<", "b>"])
    assert m == {"content": "答えです<b>", "thinking": "考え中"}


def test_think_splitter_plain():
    assert _run(["普通の", "答え"]) == {"content": "普通の答え", "thinking": ""}


def test_extract_json_tolerates_noise():
    assert extract_json('<think>{"x":1}</think>前置き {"a": "}", "b": {"c": 2}} 後') == {"a": "}", "b": {"c": 2}}
    assert extract_json("```json\n{\"action\": \"answer\"}\n```") == {"action": "answer"}
    assert extract_json("JSON なし") is None
    assert strip_think("a<think>b") == "a"


def test_rag_messages_modes():
    hits = [{"path": "a.md", "chunk_index": 0, "num_chunks": 2, "score": 0.9, "text": "本文"}]
    m = build_rag_messages("質問", hits, system="sys")
    assert m[0] == {"role": "system", "content": "sys"} and "[1] a.md" in m[-1]["content"]
    assert "見つかりませんでした" in build_rag_messages("質問", [], system="s")[-1]["content"]
    assert "参照しない設定" in build_rag_messages("質問", [], system="s", rag_used=False)[-1]["content"]


class _FakeResp:
    def __init__(self, lines=None, payload=None, status=200):
        self.lines, self.payload, self.status_code = lines or [], payload, status
        self.text = "err"

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)

    def json(self):
        return self.payload

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_ollama_native_body_and_stream():
    from ragcore.llm import LLMClient
    c = LLMClient("http://x:11434", model="m", api="ollama", num_ctx=16384, think=False)
    sent = {}

    def post(url, json=None, **kw):
        sent.update(url=url, body=json)
        if json["stream"]:
            return _FakeResp(lines=[b'{"message":{"thinking":"t"},"done":false}',
                                    b'{"message":{"content":"<think>x</think>A"},"done":false}',
                                    b'{"message":{"content":"B"},"done":true}'])
        return _FakeResp(payload={"message": {"content": "ok", "thinking": "hm"}})

    c._sess.post = post
    assert c.chat([{"role": "user", "content": "q"}], options={"num_predict": 5}) == {"content": "ok", "thinking": "hm"}
    assert sent["url"].endswith("/api/chat")
    assert sent["body"]["options"] == {"num_ctx": 16384, "num_predict": 5} and sent["body"]["think"] is False
    evs = list(c.chat_stream([{"role": "user", "content": "q"}]))
    text = {"content": "", "thinking": ""}
    for e in evs:
        if e["type"] in text:
            text[e["type"]] += e["text"]
    assert text == {"content": "AB", "thinking": "tx"} and evs[-1]["type"] == "done"


def test_client_openai_body_and_sse():
    from ragcore.llm import LLMClient
    c = LLMClient("http://x/v1", model="m", api="openai")
    sent = {}

    def post(url, json=None, **kw):
        sent.update(url=url, body=json)
        return _FakeResp(lines=[b'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}',
                                b'data: {"choices":[{"delta":{"content":"Hi"}}]}', b"data: [DONE]"])

    c._sess.post = post
    evs = [e for e in c.chat_stream([], options={"num_predict": 9, "temperature": 0.1})]
    assert sent["url"].endswith("/chat/completions") and sent["body"]["max_tokens"] == 9
    assert "options" not in sent["body"]
    assert [(e["type"], e["text"]) for e in evs[:2]] == [("thinking", "r"), ("content", "Hi")]
