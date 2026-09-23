"""Local Dashboard REST API served by the existing gateway event loop."""

from __future__ import annotations

import asyncio
import sqlite3
from urllib.parse import urlsplit

from aiohttp import web
from loguru import logger

from nanobot.observability.store import ObservabilityStore

AGENT_KEY = web.AppKey("agent_loop", object)
CRON_KEY = web.AppKey("cron_service", object)
STORE_KEY = web.AppKey("observability_store", ObservabilityStore)
HOSTS_KEY = web.AppKey("allowed_hosts", set)


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


def create_dashboard_app(agent_loop, cron_service, observability_store: ObservabilityStore,
                         *, host: str = "127.0.0.1") -> web.Application:
    app = web.Application(middlewares=[guard], client_max_size=16 * 1024)
    app[AGENT_KEY] = agent_loop
    app[CRON_KEY] = cron_service
    app[STORE_KEY] = observability_store
    app[HOSTS_KEY] = {"127.0.0.1", "localhost", "::1", host}
    app.router.add_get("/api/dashboard/overview", overview)
    app.router.add_get("/api/dashboard/runs", runs)
    app.router.add_get("/api/dashboard/runs/{run_id}", run_detail)
    return app
