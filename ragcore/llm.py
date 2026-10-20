"""LLM サーバのクライアント。2 つの API に対応する。

    api: openai   OpenAI 互換 (vLLM / LM Studio / llama.cpp server / Ollama の /v1)
                  POST {base_url}/chat/completions   stream=true → SSE (data: {...})
    api: ollama   Ollama ネイティブ
                  POST {base_url}/api/chat           stream=true → NDJSON (1 行 1 JSON)

Ollama は OpenAI 互換の口だと文脈長 (num_ctx) を指定できず、既定の 4096 トークンで動く。
エージェントのプロンプトはそれより長いので、Ollama を使うときはネイティブの口 (api: ollama) にして
num_ctx を渡す (思考の on/off = think もここで渡せる)。

推論モデルの思考は、サーバによって reasoning_content (vLLM) / reasoning / thinking (Ollama) /
本文中の <think>…</think> のどれかで返る。どれでも "thinking" として扱う。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional

import requests

CONNECT_TIMEOUT = 5.0


class LLMError(RuntimeError):
    """LLM サーバ由来の失敗。画面にはこの文言をそのまま出す。"""


class LLMClient:
    def __init__(self, base_url: str, *, model: str, api: str = "openai", api_key: str = "",
                 timeout: int = 600, num_ctx: int = 16384, think: Optional[bool] = None,
                 extra_body: Optional[Dict[str, Any]] = None):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api = (api or "openai").lower()
        if self.api not in ("openai", "ollama"):
            raise ValueError(f"llm.api must be 'openai' or 'ollama', got {api!r}")
        self.api_key = api_key or ""
        self.timeout = int(timeout)
        self.num_ctx = int(num_ctx)
        self.think = think
        self.extra_body = dict(extra_body or {})
        self._sess = requests.Session()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], *, role: str = "chat") -> "LLMClient":
        """role="agent" なら llm.agent_model (空なら llm.model) を使う。環境変数 LLM_MODEL があれば最優先。"""
        lcfg = cfg.get("llm") or {}
        model = (os.environ.get("LLM_MODEL") or
                 (lcfg.get("agent_model") if role == "agent" else "") or lcfg.get("model") or "")
        key = os.environ.get(lcfg.get("api_key_env") or "LLM_API_KEY", "") or "none"
        api = lcfg.get("api") or "openai"
        default_url = "http://localhost:11434" if api == "ollama" else "http://localhost:8000/v1"
        return cls(lcfg.get("base_url") or default_url, model=model, api=api, api_key=key,
                   timeout=int(lcfg.get("timeout", 600)), num_ctx=int(lcfg.get("num_ctx", 16384)),
                   think=lcfg.get("think"), extra_body=lcfg.get("extra_body") or {})

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _url(self) -> str:
        return self.base_url + ("/api/chat" if self.api == "ollama" else "/chat/completions")

    def _body(self, messages: List[Dict[str, Any]], options: Optional[Dict[str, Any]],
              stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        opts = {k: v for k, v in (options or {}).items() if v is not None}
        if self.api == "ollama":
            body["options"] = dict({"num_ctx": self.num_ctx}, **opts)
            if self.think is not None:
                body["think"] = bool(self.think)
        else:
            if "num_predict" in opts:             # Ollama の語を OpenAI の語に直す
                opts["max_tokens"] = opts.pop("num_predict")
            body.update(opts)
        body.update(self.extra_body)
        return body

    def _post(self, body: Dict[str, Any], stream: bool):
        try:
            r = self._sess.post(self._url(), json=body, headers=self._headers(), stream=stream,
                                timeout=(CONNECT_TIMEOUT, self.timeout))
        except requests.RequestException as e:
            raise LLMError(f"LLM サーバに接続できません ({self.base_url}): {e}") from e
        if r.status_code != 200:
            text = r.text[:500]
            r.close()
            raise LLMError(f"LLM HTTP {r.status_code}: {text}")
        return r

    def health(self) -> Dict[str, Any]:
        if self.api == "ollama":
            r = self._sess.get(self.base_url + "/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name") for m in (r.json().get("models") or [])]
        else:
            r = self._sess.get(self.base_url + "/models", headers=self._headers(), timeout=3)
            r.raise_for_status()
            names = [m.get("id") for m in (r.json().get("data") or [])]
        return {"api": self.api, "models": names, "default_model": self.model,
                "model_available": self.model in names or f"{self.model}:latest" in names}

    def chat(self, messages: List[Dict[str, Any]], *, options: Optional[Dict[str, Any]] = None
             ) -> Dict[str, str]:
        """→ {"content": 本文 (<think> 除去済み), "thinking": 思考}"""
        r = self._post(self._body(messages, options, False), stream=False)
        try:
            data = r.json()
            msg = data["message"] if self.api == "ollama" else data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError(f"LLM の応答を読めません: {r.text[:300]}") from e
        raw = msg.get("content") or ""
        thinking = msg.get("thinking") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        m = _THINK_RE.search(raw)
        if m:
            thinking = (thinking + "\n" + m.group(0)[7:-8]).strip()
        return {"content": strip_think(raw), "thinking": thinking}

    def chat_stream(self, messages: List[Dict[str, Any]], *, options: Optional[Dict[str, Any]] = None
                    ) -> Iterator[Dict[str, str]]:
        """{"type": "content"|"thinking"|"error", "text": …} を逐次返す。最後に {"type": "done", "text": 終了理由}。
        本文に混ざる <think>…</think> は thinking に振り分ける。"""
        r = self._post(self._body(messages, options, True), stream=True)
        splitter = _ThinkSplitter()
        reason = ""
        try:
            with r:
                for raw in r.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    if self.api == "openai":
                        if not raw.startswith(b"data:"):
                            continue
                        raw = raw[5:].strip()
                        if raw == b"[DONE]":
                            break
                    try:
                        obj = json.loads(raw.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    if obj.get("error"):
                        yield {"type": "error", "text": str(obj["error"])}
                        return
                    if self.api == "ollama":
                        deltas = [obj.get("message") or {}]
                    else:
                        deltas = [ch.get("delta") or {} for ch in obj.get("choices") or []]
                    for d in deltas:
                        think = d.get("thinking") or d.get("reasoning_content") or d.get("reasoning")
                        if think:
                            yield {"type": "thinking", "text": think}
                        if d.get("content"):
                            yield from splitter.feed(d["content"])
                    if self.api == "ollama":
                        if obj.get("done"):
                            reason = str(obj.get("done_reason") or "")
                            break
                    else:
                        for ch in obj.get("choices") or []:
                            reason = str(ch.get("finish_reason") or reason)
            yield from splitter.flush()
            # text = 終了理由。"length" なら出力の上限で切れた (推論モデルは思考も上限を食う)
            yield {"type": "done", "text": reason}
        except requests.RequestException as e:
            raise LLMError(f"LLM サーバとの通信が切れました ({self.base_url}): {e}") from e


class _ThinkSplitter:
    """ストリームの本文から <think>…</think> を thinking に分ける (タグがチャンクをまたいでもよい)。"""

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> Iterator[Dict[str, str]]:
        self.buf += text
        while self.buf:
            tag = "</think>" if self.inside else "<think>"
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    yield {"type": "thinking" if self.inside else "content", "text": self.buf[:i]}
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            keep = 0                       # タグの頭だけ届いているかもしれない分は持ち越す
            for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-n:]):
                    keep = n
                    break
            out = self.buf[:len(self.buf) - keep]
            if out:
                yield {"type": "thinking" if self.inside else "content", "text": out}
            self.buf = self.buf[len(self.buf) - keep:]
            break

    def flush(self) -> Iterator[Dict[str, str]]:
        if self.buf:
            yield {"type": "thinking" if self.inside else "content", "text": self.buf}
            self.buf = ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str) -> str:
    """<think>…</think> を落とす (閉じていない思考も落とす)。"""
    t = _THINK_RE.sub("", text or "")
    if "<think>" in t.lower():
        t = t[:t.lower().index("<think>")]
    return t.strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """出力から最初の JSON オブジェクトを取り出す。コードフェンス・地の文・<think> 混入に耐える。"""
    t = strip_think(text)
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = t.find("{", start + 1)
    return None
