from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Dict, Any, Tuple, List

import requests

from .config import CODEX_BASE_URL, CODEX_MODEL_NAME, CODEX_PROXY, CODEX_TIMEOUT, CODEX_REASONING_EFFORT


class CodexClient:
    def __init__(self, base_url: str, model: str, proxy: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = os.getenv("CODEX_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise RuntimeError("空响应")
        try:
            return json.loads(text)
        except Exception:
            pass
        # 兼容 "JSON后面还拼接了其他文本/事件" 的情况，只解出第一个 JSON 对象
        decoder = json.JSONDecoder()
        first_brace = text.find("{")
        if first_brace >= 0:
            try:
                obj, _ = decoder.raw_decode(text[first_brace:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        m = re.search(r"```json\s*(\{[\s\S]*\})\s*```", text)
        if m:
            return json.loads(m.group(1))
        m2 = re.search(r"(\{[\s\S]*\})", text)
        if m2:
            return json.loads(m2.group(1))
        raise RuntimeError("Codex 输出中未找到可解析 JSON")

    def _parse_sse_text(self, raw_text: str) -> str:
        js = None
        collected = []
        for line in raw_text.splitlines():
            if line.startswith("data: "):
                chunk = line[6:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                    js = obj
                    t = obj.get("type", "")
                    if t == "response.output_text.delta":
                        collected.append(obj.get("delta", ""))
                    elif t == "response.completed":
                        resp = obj.get("response", {})
                        if resp.get("output_text"):
                            collected.append(resp["output_text"])
                        for item in resp.get("output", []) or []:
                            for c in item.get("content", []) or []:
                                txt = c.get("text")
                                if isinstance(txt, str) and txt.strip():
                                    collected.append(txt)
                    elif t == "response.output_item.done":
                        item = obj.get("item", {}) or {}
                        for c in item.get("content", []) or []:
                            txt = c.get("text")
                            if isinstance(txt, str) and txt.strip():
                                collected.append(txt)
                except Exception:
                    continue
        if collected:
            return "".join(collected)
        if js and "response" in js:
            resp = js.get("response", {})
            if resp.get("output_text"):
                return resp["output_text"]
            parts = []
            for item in resp.get("output", []) or []:
                for c in item.get("content", []) or []:
                    txt = c.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
            if parts:
                return "\n".join(parts)
        return ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing CODEX_API_KEY/OPENAI_API_KEY for Codex call.")

        chat_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        resp_payload = {
            "model": self.model,
            "reasoning": {"effort": CODEX_REASONING_EFFORT},
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "stream": False,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempts: List[Tuple[str, Dict[str, Any], str]] = [
            (f"{self.base_url}/v1/chat/completions", chat_payload, "chat"),
            (f"{self.base_url}/chat/completions", chat_payload, "chat"),
            (f"{self.base_url}/v1/responses", resp_payload, "resp"),
        ]

        errs = []
        proxy_modes = [("proxy", self.proxies)]
        if self.proxies:
            proxy_modes.append(("direct", None))

        for mode_name, proxies in proxy_modes:
            for url, payload, kind in attempts:
                for retry in range(3):
                    r = None
                    try:
                        r = requests.post(url, json=payload, headers=headers, proxies=proxies, timeout=self.timeout)
                        r.raise_for_status()

                        content_type = r.headers.get("Content-Type", "")
                        if "text/event-stream" in content_type or r.text.lstrip().startswith("event:"):
                            text = self._parse_sse_text(r.text)
                            return self._extract_json(text)

                        js = r.json()
                        if kind == "chat" and "choices" in js:
                            content = js["choices"][0]["message"].get("content", "")
                            return self._extract_json(content)
                        if kind == "resp":
                            if "output_text" in js and js["output_text"]:
                                return self._extract_json(js["output_text"])
                            if "output" in js:
                                parts = []
                                for item in js.get("output", []):
                                    for c in item.get("content", []):
                                        if c.get("type") == "output_text":
                                            parts.append(c.get("text", ""))
                                if parts:
                                    return self._extract_json("\n".join(parts))
                        return self._extract_json(r.text)
                    except Exception as e:
                        body = ""
                        try:
                            body = (r.text or "")[:300] if r is not None else ""
                        except Exception:
                            pass
                        errs.append(f"[{mode_name}][retry={retry+1}] {url} -> {type(e).__name__}: {e} | {body}")
                        # 对网络类错误做短暂退避重试
                        if retry < 2:
                            time.sleep(1.2 * (retry + 1))
                            continue

        raise RuntimeError("Codex API call failed on all endpoints: " + " || ".join(errs))


def call_codex_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    client = CodexClient(
        base_url=CODEX_BASE_URL,
        model=CODEX_MODEL_NAME,
        proxy=CODEX_PROXY,
        timeout=CODEX_TIMEOUT,
    )
    return client.generate_json(system_prompt, user_prompt)
