from __future__ import annotations

import httpx
import pytest

from bytx.config import OllamaConfig
from bytx.llm import OllamaClient, OllamaError


def make_cfg() -> OllamaConfig:
    return OllamaConfig(host="http://127.0.0.1:11434", embed_model="bge-m3", chat_model="gemma3:4b", num_ctx=8192)


def make_client(handler) -> OllamaClient:
    """Builds an OllamaClient whose internal httpx.Client is wired to a
    MockTransport instead of the network.
    """
    cfg = make_cfg()
    client = OllamaClient(cfg)
    client._client = httpx.Client(base_url=cfg.host, transport=httpx.MockTransport(handler))
    return client


def test_is_reachable_true_on_200():
    def handler(request):
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.13.1"})

    client = make_client(handler)
    assert client.is_reachable() is True


def test_is_reachable_false_on_error_status():
    def handler(request):
        return httpx.Response(500)

    client = make_client(handler)
    assert client.is_reachable() is False


def test_is_reachable_false_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    assert client.is_reachable() is False


def test_has_model_exact_match():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "bge-m3:latest"}, {"name": "gemma3:4b"}]})

    client = make_client(handler)
    assert client.has_model("gemma3:4b") is True
    assert client.has_model("bge-m3") is True  # bare name matches versioned tag
    assert client.has_model("qwen3:8b") is False


def test_has_model_raises_ollama_error_when_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)
    with pytest.raises(OllamaError):
        client.has_model("gemma3:4b")


def test_embed_returns_vectors():
    def handler(request):
        assert request.url.path == "/api/embed"
        body = request.read()
        assert b"bge-m3" in body
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = make_client(handler)
    result = client.embed(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_uses_override_model():
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    client = make_client(handler)
    client.embed(["x"], model="other-embed")
    assert seen["body"]["model"] == "other-embed"


def test_embed_raises_on_missing_embeddings_key():
    def handler(request):
        return httpx.Response(200, json={"error": "model not found"})

    client = make_client(handler)
    with pytest.raises(OllamaError):
        client.embed(["hello"])


def test_embed_raises_on_http_error():
    def handler(request):
        return httpx.Response(404)

    client = make_client(handler)
    with pytest.raises(OllamaError):
        client.embed(["hello"])


def test_chat_returns_content():
    def handler(request):
        assert request.url.path == "/api/chat"
        import json

        body = json.loads(request.read())
        assert body["model"] == "gemma3:4b"
        assert body["stream"] is False
        assert body["options"]["num_ctx"] == 8192
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "উত্তর"}})

    client = make_client(handler)
    result = client.chat([{"role": "user", "content": "প্রশ্ন"}])
    assert result == "উত্তর"


def test_chat_passes_custom_num_ctx_and_temperature():
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = make_client(handler)
    client.chat([{"role": "user", "content": "x"}], num_ctx=4096, temperature=0.2)
    assert seen["body"]["options"]["num_ctx"] == 4096
    assert seen["body"]["options"]["temperature"] == 0.2


def test_chat_raises_on_missing_content():
    def handler(request):
        return httpx.Response(200, json={"message": {}})

    client = make_client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "x"}])


def test_chat_raises_on_http_error():
    def handler(request):
        return httpx.Response(500)

    client = make_client(handler)
    with pytest.raises(OllamaError):
        client.chat([{"role": "user", "content": "x"}])


def test_context_manager_closes_client():
    cfg = make_cfg()
    with OllamaClient(cfg) as client:
        assert client is not None
    # closing twice should not raise
    client.close()
