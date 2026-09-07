# Python SDK

Use nanobot programmatically — load config, run the agent, get results.

## Quick Start

```python
import asyncio
from nanobot import Nanobot

async def main():
    bot = Nanobot.from_config()
    result = await bot.run("What time is it in Tokyo?")
    print(result.content)

asyncio.run(main())
```

## API

### `Nanobot.from_config(config_path?, *, workspace?)`

Create a `Nanobot` from a config file.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `str \| Path \| None` | `None` | Path to `config.json`. Defaults to `~/.nanobot/config.json`. |
| `workspace` | `str \| Path \| None` | `None` | Override workspace directory from config. |

Raises `FileNotFoundError` if an explicit path doesn't exist.

### `await bot.run(message, *, session_key?, hooks?)`

Run the agent once. Returns a `RunResult`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `str` | *(required)* | The user message to process. |
| `session_key` | `str` | `"sdk:default"` | Session identifier for conversation isolation. Different keys get independent history. |
| `hooks` | `list[AgentHook] \| None` | `None` | Lifecycle hooks for this run only. |

```python
# Isolated sessions — each user gets independent conversation history
await bot.run("hi", session_key="user-alice")
await bot.run("hi", session_key="user-bob")
```

### `RunResult`

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The agent's final text response. |
| `stop_reason` | `str` | Completion status, including `context_limit` when a model call was stopped. |
| `tools_used` | `list[str]` | Tool names invoked during the run. |
| `messages` | `list[dict]` | Raw message history (for debugging). |

## Hooks

Hooks let you observe or modify the agent loop without touching internals.

Subclass `AgentHook` and override any method:

| Method | When |
|--------|------|
| `before_iteration(ctx)` | Before each LLM call |
| `on_stream(ctx, delta)` | On each streamed token |
| `on_stream_end(ctx)` | When streaming finishes |
| `before_execute_tools(ctx)` | Before tool execution (inspect `ctx.tool_calls`) |
| `after_iteration(ctx, response)` | After each LLM response |
| `finalize_content(ctx, content)` | Transform final output text |

### Example: Audit Hook

```python
from nanobot.agent import AgentHook, AgentHookContext

class AuditHook(AgentHook):
    def __init__(self):
        self.calls = []

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            self.calls.append(tc.name)
            print(f"[audit] {tc.name}({tc.arguments})")

hook = AuditHook()
result = await bot.run("List files in /tmp", hooks=[hook])
print(f"Tools used: {hook.calls}")
```

### Composing Hooks

Pass multiple hooks — they run in order, errors in one don't block others:

```python
result = await bot.run("hi", hooks=[AuditHook(), MetricsHook()])
```

Under the hood this uses `CompositeHook` for fan-out with error isolation.

### `finalize_content` Pipeline

Unlike the async methods (fan-out), `finalize_content` is a pipeline — each hook's output feeds the next:

```python
class Censor(AgentHook):
    def finalize_content(self, ctx, content):
        return content.replace("secret", "***") if content else content
```

## Full Example

```python
import asyncio
from nanobot import Nanobot
from nanobot.agent import AgentHook, AgentHookContext

class TimingHook(AgentHook):
    async def before_iteration(self, ctx: AgentHookContext) -> None:
        import time
        ctx.metadata["_t0"] = time.time()

    async def after_iteration(self, ctx, response) -> None:
        import time
        elapsed = time.time() - ctx.metadata.get("_t0", 0)
        print(f"[timing] iteration took {elapsed:.2f}s")

async def main():
    bot = Nanobot.from_config(workspace="/my/project")
    result = await bot.run(
        "Explain the main function",
        hooks=[TimingHook()],
    )
    print(result.content)

asyncio.run(main())
```

## Memory, knowledge, and context limits

`Nanobot.from_config()` uses the same memory and knowledge services as CLI, gateway, and HTTP.
Successful archives and valid edits to `memory/MEMORY.md` appear in the next prepared context.
Core memory is always included in full; dynamic memory is selected by query. `kb_search` performs
one bounded online supplement when local evidence is insufficient.

`RunResult.stop_reason` reports `context_limit` when the necessary request cannot fit the configured
`agents.defaults.contextWindowTokens` after reserving `maxTokens` for output. `content` explains the
limit; the blocked model call is not made. Core memory is never truncated. Optional dynamic records
and knowledge evidence may be removed as complete units. If evidence is reduced, its previous
sufficiency assessment is invalidated. The check also runs after tools and after `before_iteration`
hooks, for both streaming and non-streaming calls.

HTTP exposes the same condition as status 400 with `error.type="context_limit"`. Configure the actual
model context limit; token counts use a provider counter when available, otherwise local estimation,
which is not a guarantee of exact provider-side accounting, particularly for multimodal requests.
The hard stop prevents requests already known to exceed the configured limit.

For old workspaces, follow [offline conversion and switching](MEMORY_KNOWLEDGE_UPGRADE.md). An offline
copy marked `REBUILD_REQUIRED` must finish its separately authorized embedding rebuild before use.
