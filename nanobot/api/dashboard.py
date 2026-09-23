"""Local Dashboard REST API served by the existing gateway event loop."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
from loguru import logger

from nanobot.agent.memory_db import MEMORY_CLASSES, MemoryRevisionConflictError
from nanobot.agent.memory_sync import MarkdownValidationError, MemoryConflictError
from nanobot.observability.store import ObservabilityStore

AGENT_KEY = web.AppKey("agent_loop", object)
CRON_KEY = web.AppKey("cron_service", object)
STORE_KEY = web.AppKey("observability_store", ObservabilityStore)
HOSTS_KEY = web.AppKey("allowed_hosts", set)
STATIC_KEY = web.AppKey("static_dir", Path)


def error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


@web.middleware
async def guard(request: web.Request, handler):
    allowed = request.app[HOSTS_KEY]
    try:
        host = request.url.host
    except ValueError:
        return error(400, "invalid_host", "Invalid Host header")
    if host not in allowed:
        return error(400, "invalid_host", "Host is not allowed")
    if request.method not in {"GET", "HEAD", "OPTIONS"} or request.headers.get("Upgrade", "").lower() == "websocket":
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != request.host:
                return error(403, "invalid_origin", "Origin must match this Dashboard")
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except sqlite3.Error:
        logger.exception("Dashboard store unavailable")
        return error(503, "store_unavailable", "Dashboard storage is unavailable")
    except Exception:
        logger.exception("Dashboard request failed")
        return error(500, "internal_error", "Dashboard request failed")


async def overview(request: web.Request) -> web.Response:
    store: ObservabilityStore = request.app[STORE_KEY]
    data = await asyncio.to_thread(store.get_overview)
    jobs = await asyncio.to_thread(request.app[CRON_KEY].list_jobs, True)
    data["enabled_tasks"] = sum(job.enabled for job in jobs)
    return web.json_response(data)


async def runs(request: web.Request) -> web.Response:
    raw_limit = request.query.get("limit", "50")
    if not raw_limit.isascii() or not raw_limit.isdecimal():
        return error(400, "invalid_limit", "limit must be an integer from 1 to 100")
    limit = int(raw_limit)
    if not 1 <= limit <= 100:
        return error(400, "invalid_limit", "limit must be an integer from 1 to 100")
    cursor = request.query.get("cursor")
    try:
        data = await asyncio.to_thread(request.app[STORE_KEY].list_runs,
                                       limit, cursor)
    except ValueError:
        return error(400, "invalid_cursor", "Invalid pagination cursor")
    return web.json_response(data)


async def run_detail(request: web.Request) -> web.Response:
    detail = await asyncio.to_thread(request.app[STORE_KEY].get_run,
                                     request.match_info["run_id"])
    if detail is None:
        return error(404, "not_found", "Run not found")
    return web.json_response(detail)


def memory_service(request: web.Request):
    service = getattr(request.app[AGENT_KEY], "memory_service", None)
    if service is None or getattr(service, "database", None) is None:
        return None
    return service


async def memory(request: web.Request) -> web.Response:
    service = memory_service(request)
    if service is None:
        return error(503, "memory_unavailable", "Memory service is unavailable")
    snapshot = await asyncio.to_thread(service.database.read_snapshot)
    counts = {main_class: 0 for main_class in MEMORY_CLASSES}
    for record in snapshot.memories:
        counts[record.main_class] += 1
    return web.json_response({
        "revision": snapshot.revision,
        "counts": counts,
        "records": [asdict(record) for record in snapshot.memories],
    })


async def memory_search(request: web.Request) -> web.Response:
    service = memory_service(request)
    if service is None or getattr(service, "embedder", None) is None:
        return error(503, "memory_unavailable", "Memory search is unavailable")
    try:
        body = await request.json()
    except (ValueError, web.HTTPException):
        return error(400, "invalid_body", "Expected a JSON object")
    if not isinstance(body, dict):
        return error(400, "invalid_body", "Expected a JSON object")
    query = body.get("query")
    limit = body.get("limit", service.config.dynamic_top_k)
    if not isinstance(query, str) or not query.strip() or len(query) > 512:
        return error(400, "invalid_query", "query must contain 1 to 512 characters")
    if type(limit) is not int or not 1 <= limit <= 100:
        return error(400, "invalid_limit", "limit must be an integer from 1 to 100")
    try:
        await service.sync_markdown()
        core = await asyncio.to_thread(service.database.read_core_memories)
        hits = await service.search_dynamic(query, limit=limit)
    except MarkdownValidationError as exc:
        return error(409, "invalid_markdown", str(exc))
    except (MemoryConflictError, MemoryRevisionConflictError) as exc:
        return error(409, "memory_conflict", str(exc))
    except RuntimeError:
        logger.exception("Dashboard memory search failed")
        return error(502, "embedding_failed", "Memory embedding or search service failed")
    return web.json_response({
        "core": [asdict(record) for record in core],
        "dynamic": [asdict(hit) for hit in hits],
    })


async def dashboard_index(request: web.Request) -> web.StreamResponse:
    index = request.app[STATIC_KEY] / "index.html"
    if not index.is_file():
        return error(503, "ui_not_built", "Dashboard UI is not built; run npm run build in dashboard/")
    return web.FileResponse(index)


async def dashboard_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/dashboard/")


async def dashboard_asset(request: web.Request) -> web.StreamResponse:
    assets = (request.app[STATIC_KEY] / "assets").resolve()
    path = (assets / request.match_info["name"]).resolve()
    if not path.is_relative_to(assets) or not path.is_file():
        return error(404, "not_found", "Asset not found")
    return web.FileResponse(path)


def create_dashboard_app(agent_loop, cron_service, observability_store: ObservabilityStore,
                         *, host: str = "127.0.0.1",
                         static_dir: Path | None = None) -> web.Application:
    app = web.Application(middlewares=[guard], client_max_size=16 * 1024)
    app[AGENT_KEY] = agent_loop
    app[CRON_KEY] = cron_service
    app[STORE_KEY] = observability_store
    app[HOSTS_KEY] = {"127.0.0.1", "localhost", "::1", host}
    app[STATIC_KEY] = static_dir or Path(__file__).resolve().parents[2] / "dashboard" / "dist"
    app.router.add_get("/api/dashboard/overview", overview)
    app.router.add_get("/api/dashboard/runs", runs)
    app.router.add_get("/api/dashboard/runs/{run_id}", run_detail)
    app.router.add_get("/api/dashboard/memory", memory)
    app.router.add_post("/api/dashboard/memory/search", memory_search)
    app.router.add_get("/dashboard", dashboard_redirect)
    app.router.add_get("/dashboard/", dashboard_index)
    app.router.add_get("/dashboard/assets/{name:.*}", dashboard_asset)
    return app
