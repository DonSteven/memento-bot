import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.api.dashboard import create_dashboard_app
from nanobot.observability.store import ObservabilityStore


@pytest.mark.asyncio
async def test_dashboard_index_redirect_and_assets(tmp_path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text('<script src="/dashboard/assets/app.js"></script>')
    (assets / "app.js").write_text("window.dashboardLoaded=true")
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    app = create_dashboard_app(object(), object(), store, static_dir=dist)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/dashboard", allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/dashboard/"
        response = await client.get("/dashboard/")
        assert response.status == 200
        assert "/dashboard/assets/app.js" in await response.text()
        response = await client.get("/dashboard/assets/app.js")
        assert response.status == 200
        assert "dashboardLoaded" in await response.text()
        assert (await client.get("/dashboard/assets/missing.js")).status == 404


@pytest.mark.asyncio
async def test_missing_build_keeps_rest_available(tmp_path):
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    app = create_dashboard_app(object(), object(), store, static_dir=tmp_path / "missing")
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/dashboard/")
        assert response.status == 503
        assert (await response.json())["error"]["code"] == "ui_not_built"
        assert (await client.get("/api/dashboard/runs")).status == 200
