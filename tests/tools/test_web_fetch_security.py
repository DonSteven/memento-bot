"""Tests for web_fetch SSRF protection and untrusted content marking."""

from __future__ import annotations

import json
import socket
from unittest.mock import patch

import pytest

from nanobot.agent.tools.web import WebFetchTool


def _fake_resolve_private(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.169.254", 0))]


def _fake_resolve_public(hostname, port, family=0, type_=0):
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_ip():
    tool = WebFetchTool()
    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_private):
        result = await tool.execute(url="http://169.254.169.254/computeMetadata/v1/")
    data = json.loads(result)
    assert "error" in data
    assert "private" in data["error"].lower() or "blocked" in data["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_blocks_localhost():
    tool = WebFetchTool()
    def _resolve_localhost(hostname, port, family=0, type_=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
    with patch("nanobot.security.network.socket.getaddrinfo", _resolve_localhost):
        result = await tool.execute(url="http://localhost/admin")
    data = json.loads(result)
    assert "error" in data


@pytest.mark.asyncio
async def test_web_fetch_result_contains_untrusted_flag():
    """When fetch succeeds, result JSON must include untrusted=True and the banner."""
    tool = WebFetchTool()

    fake_html = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"

    import httpx

    class FakeResponse:
        status_code = 200
        url = "https://example.com/page"
        text = fake_html
        headers = {"content-type": "text/html"}
        def raise_for_status(self): pass
        def json(self): return {}

    async def _fake_get(self, url, **kwargs):
        return FakeResponse()

    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public), \
         patch("httpx.AsyncClient.get", _fake_get):
        result = await tool.execute(url="https://example.com/page")

    data = json.loads(result)
    assert data.get("untrusted") is True
    assert "[External content" in data.get("text", "")


@pytest.mark.asyncio
async def test_web_fetch_blocks_private_redirect_before_returning_image(monkeypatch):
    tool = WebFetchTool()

    class FakeStreamResponse:
        headers = {"content-type": "image/png"}
        url = "http://127.0.0.1/secret.png"
        content = b"\x89PNG\r\n\x1a\n"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self):
            return self.content

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers=None):
            return FakeStreamResponse()

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)

    with patch("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public):
        result = await tool.execute(url="https://example.com/image.png")

    data = json.loads(result)
    assert "error" in data
    assert "redirect blocked" in data["error"].lower()


@pytest.mark.asyncio
async def test_readability_rejects_binary_body(monkeypatch):
    import httpx

    async def fake_get(*args, **kwargs):
        return httpx.Response(200, content=b"%PDF binary data", headers={"content-type": "application/pdf"},
                              request=httpx.Request("GET", "https://example.com/file.pdf"))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr("nanobot.security.network.validate_resolved_url", lambda url: (True, ""))
    result = await WebFetchTool()._fetch_readability("https://example.com/file.pdf", "text", 50000)
    assert "Unsupported content type" in json.loads(result)["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type, body, extractor",
    [
        (
            "application/xhtml+xml; charset=utf-8",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Release</title></head>'
            '<body><p>Version 2 was released in May 2024.</p></body></html>',
            "readability",
        ),
        (
            "application/xml",
            '<?xml version="1.0"?><release>Version 2 was released in May 2024.</release>',
            "raw",
        ),
        (
            "Application/Atom+XML; charset=UTF-8",
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            '<entry><title>Version 2 was released in May 2024.</title></entry></feed>',
            "raw",
        ),
    ],
)
async def test_web_fetch_extracts_xml_text_when_jina_is_unavailable(
    monkeypatch, content_type, body, extractor
):
    from unittest.mock import AsyncMock

    import httpx

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=body, headers={"content-type": content_type})
    )
    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        "nanobot.agent.tools.web.httpx.AsyncClient",
        lambda **kwargs: client_class(transport=transport, **kwargs),
    )
    monkeypatch.setattr("nanobot.security.network.socket.getaddrinfo", _fake_resolve_public)
    tool = WebFetchTool()
    tool._fetch_jina = AsyncMock(return_value=None)

    result = json.loads(await tool.execute("https://example.com/release", extractMode="text"))

    assert "error" not in result
    assert "Version 2 was released in May 2024." in result["text"]
    assert result["extractor"] == extractor
    assert result["untrusted"] is True and result["truncated"] is False
    assert result["finalUrl"] == "https://example.com/release"
    if extractor == "readability":
        assert "<html" not in result["text"]
    tool._fetch_jina.assert_awaited_once()
