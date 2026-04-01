# Phase 0 test content:
# - verifies the CLI agent command forwards config.memory.mode into AgentLoop
# - keeps Phase 0 CLI wiring isolated from the repository's original CLI tests
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_0/cli/test_agent_memory_mode.py

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from nanobot.bus.events import OutboundMessage
from nanobot.cli.commands import app
from nanobot.config.schema import Config


runner = CliRunner()


def test_agent_passes_memory_mode_to_loop(tmp_path: Path) -> None:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "default-workspace")
    config.memory.mode = "shadow"

    with patch("nanobot.config.loader.load_config", return_value=config), \
         patch("nanobot.cli.commands.sync_workspace_templates"), \
         patch("nanobot.cli.commands._make_provider", return_value=object()), \
         patch("nanobot.cli.commands._print_agent_response"), \
         patch("nanobot.bus.queue.MessageBus"), \
         patch("nanobot.cron.service.CronService"), \
         patch("nanobot.agent.loop.AgentLoop") as mock_agent_loop_cls:

        agent_loop = MagicMock()
        agent_loop.channels_config = None
        agent_loop.process_direct = AsyncMock(
            return_value=OutboundMessage(channel="cli", chat_id="direct", content="mock-response"),
        )
        agent_loop.close_mcp = AsyncMock(return_value=None)
        mock_agent_loop_cls.return_value = agent_loop

        result = runner.invoke(app, ["agent", "-m", "hello"])

    assert result.exit_code == 0
    assert mock_agent_loop_cls.call_args.kwargs["memory_mode"] == "shadow"
