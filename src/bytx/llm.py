"""Minimal HTTP client for a local Ollama server: embeddings and chat.

Deliberately thin -- just enough for rag.py and summarize.py. No streaming;
responses are small enough (chunk embeddings, a summary or an answer) that
buffering the whole response is simpler and fine.
"""

from __future__ import annotations

import httpx

from .config import OllamaConfig


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: OllamaConfig, timeout: float = 120.0):
        self.cfg = cfg
        self._client = httpx.Client(base_url=cfg.host, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def is_reachable(self) -> bool:
        try:
            resp = self._client.get("/api/version")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self, model: str) -> bool:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.cfg.host}: {exc}") from exc
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        # Ollama tags include a version suffix (e.g. "bge-m3:latest"); match
        # either the exact name or the bare model name before ':'.
        bare = model.split(":")[0]
        return model in names or any(n == model or n.split(":")[0] == bare for n in names)

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        model = model or self.cfg.embed_model
        try:
            resp = self._client.post("/api/embed", json={"model": model, "input": texts})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Embedding request failed (model={model}): {exc}") from exc
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise OllamaError(f"Ollama returned no embeddings: {data}")
        return embeddings

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        num_ctx: int | None = None,
        temperature: float | None = None,
    ) -> str:
        model = model or self.cfg.chat_model
        options: dict = {"num_ctx": num_ctx if num_ctx is not None else self.cfg.num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        try:
            resp = self._client.post(
                "/api/chat",
                json={"model": model, "messages": messages, "stream": False, "options": options},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Chat request failed (model={model}): {exc}") from exc
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content")
        if content is None:
            raise OllamaError(f"Ollama returned no message content: {data}")
        return content
