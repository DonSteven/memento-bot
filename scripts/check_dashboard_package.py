"""Check built Dashboard resources in an archive or an installed Python package."""

from __future__ import annotations

import argparse
import asyncio
import tarfile
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts: list[str] = []
        self.styles: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if tag == "link" and attributes.get("rel") == "stylesheet" and attributes.get("href"):
            self.styles.append(attributes["href"])

    def local_assets(self, html: str) -> list[str]:
        self.feed(html)
        if not self.scripts or not self.styles:
            raise AssertionError("Dashboard index must reference JavaScript and CSS")
        assets = self.scripts + self.styles
        if any(not asset.startswith("/dashboard/assets/") or ".." in asset for asset in assets):
            raise AssertionError(f"Unexpected Dashboard asset reference: {assets}")
        if any(not asset.endswith(".js") for asset in self.scripts):
            raise AssertionError(f"Unexpected JavaScript reference: {self.scripts}")
        if any(not asset.endswith(".css") for asset in self.styles):
            raise AssertionError(f"Unexpected CSS reference: {self.styles}")
        return assets


def check_archive(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            index = next((name for name in names if name.endswith(
                "nanobot/api/dashboard_static/index.html")), None)
            if index is None:
                raise AssertionError(f"{path}: Dashboard index missing")
            html = archive.read(index).decode("utf-8")
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
            index = next((name for name in names if name.endswith(
                "nanobot/api/dashboard_static/index.html")), None)
            if index is None:
                raise AssertionError(f"{path}: Dashboard index missing")
            html = archive.extractfile(index).read().decode("utf-8")
    prefix = index.removesuffix("index.html")
    for reference in AssetReferences().local_assets(html):
        member = prefix + reference.removeprefix("/dashboard/")
        if member not in names:
            raise AssertionError(f"{path}: missing referenced asset {member}")
    print(f"Archive Dashboard assets OK: {path}")


async def check_installed(source_root: Path) -> None:
    from aiohttp.test_utils import TestClient, TestServer

    import nanobot
    from nanobot.api.dashboard import create_dashboard_app
    from nanobot.observability.store import ObservabilityStore

    package = Path(nanobot.__file__).resolve()
    if package.is_relative_to(source_root.resolve()):
        raise AssertionError(f"Imported source checkout instead of installed package: {package}")

    class FakeCron:
        def list_jobs(self, include_disabled=False):
            return []

    with tempfile.TemporaryDirectory() as directory:
        store = ObservabilityStore(Path(directory) / "dashboard.db")
        store.initialize()
        app = create_dashboard_app(object(), FakeCron(), store)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/dashboard/")
            if response.status != 200 or "text/html" not in response.headers.get("Content-Type", ""):
                raise AssertionError(f"Installed Dashboard index returned {response.status}")
            references = AssetReferences().local_assets(await response.text())
            for reference in references:
                response = await client.get(reference)
                expected = "javascript" if reference.endswith(".js") else "text/css"
                content_type = response.headers.get("Content-Type", "")
                if response.status != 200 or expected not in content_type:
                    raise AssertionError(f"{reference}: HTTP {response.status}, {content_type}")
                if not await response.read():
                    raise AssertionError(f"{reference}: empty resource")
    print(f"Installed Dashboard HTTP resources OK: {package}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--installed", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.archive:
        check_archive(args.archive)
    else:
        if args.source_root is None:
            parser.error("--installed requires --source-root")
        asyncio.run(check_installed(args.source_root))


if __name__ == "__main__":
    main()
