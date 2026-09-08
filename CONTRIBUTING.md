# Contributing to Memento Bot

Memento Bot is independently maintained from [HKUDS/nanobot](https://github.com/HKUDS/nanobot).
For this project, open issues and pull requests in
[DonSteven/memento-bot](https://github.com/DonSteven/memento-bot), targeting `main`.

## Development Setup

```bash
git clone https://github.com/DonSteven/memento-bot.git
cd memento-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,web_knowledge]'
python -m pytest
ruff check nanobot/
```

Use focused tests for the affected functionality. See [Quick Start](README.md#quick-start)
for runtime credentials and [evaluation](README.md#evaluation--reliability) for extension tests.

## Code Style

We care about more than passing lint. We want Memento Bot to stay small, calm, and readable.

When contributing, please aim for code that feels:

- Simple: prefer the smallest change that solves the real problem
- Clear: optimize for the next reader, not for cleverness
- Decoupled: keep boundaries clean and avoid unnecessary new abstractions
- Honest: do not hide complexity, but do not create extra complexity either
- Durable: choose solutions that are easy to maintain, test, and extend

In practice:

- Line length: 100 characters (`ruff`)
- Target: Python 3.11+
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored)
- Async: uses `asyncio` throughout; pytest with `asyncio_mode = "auto"`
- Prefer readable code over magical code
- Prefer focused patches over broad rewrites
- If a new abstraction is introduced, it should clearly reduce complexity rather than move it around
