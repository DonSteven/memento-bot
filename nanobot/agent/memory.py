"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.memory_db import MemoryDatabase, parse_memory_markdown
from nanobot.agent.memory_pipeline import ShadowMemoryPipeline
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]

"""
它把任意值规范成字符串，方便后面写入 MEMORY.md 或 HISTORY.md。
"""
def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) 
    # dumps返回json字符串，不更改文件。ensure_ascii=False允许非ASCII字符直接输出而不是转义
    # dump将python对象直接写入文件，没有返回值

"""
它负责把不同 Provider 可能返回的 tool arguments 统一成 dict[str, Any]。
因为不同模型/SDK 对工具参数的返回格式不完全一致，args 可能是：

JSON 字符串
Python dict
list，且第一个元素才是真正的参数对象
"""
def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
        # loads将json字符串解析为Python对象，返回python对象
        # load输入是json文件，返回Python对象
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

"""这是一个错误关键词集合，用来匹配“当前 Provider 不支持强制 tool_choice”这类报错"""
_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)

"""检测 Provider 错误信息中是否包含上述关键词"""
def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    """连续失败多少次后放弃 LLM 归纳，直接原样存档消息到 HISTORY.md"""
    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path, mode: str = "legacy"):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.mode = mode
        self.v2_db = MemoryDatabase(workspace) if mode == "v2" else None
        self.shadow_pipeline = ShadowMemoryPipeline(workspace) if mode == "shadow" else None
        self._consecutive_failures = 0

    def read_long_term(self) -> str: # 读取 Memory.md 内容
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None: # 把 content 写入 Memory.md，覆盖原内容
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None: # 往 History.md 追加一条记录
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def get_retrieval_context(self, query: str, retrieval_mode: str = "full_view", limit: int = 5) -> str:
        if retrieval_mode != "fts":
            return self.get_memory_context()
        if self.v2_db is None or not self.v2_db.db_path.exists():
            return self.get_memory_context()

        results = self.v2_db.query_canonical_memories(query, limit=limit)
        if not results:
            return ""

        lines = ["## Retrieved Memory"]
        for item in results:
            main_class = str(item.get("main_class") or "")
            sub_class = str(item.get("sub_class") or "").strip()
            text = str(item.get("text") or "").strip()
            label = f"{main_class}/{sub_class}" if sub_class else main_class
            lines.append(f"- [{label}] {text}")
        return "\n".join(lines)

    def _persist_v2(self, *, entry: str, update: str, messages: list[dict]) -> None:
        """把一次 LLM 产出的整份 MEMORY.md 更新结果，作为一个新的 v2 快照，完整写入数据库，并重建可读视图"""
        if self.v2_db is None:
            raise RuntimeError("v2 database is not initialized")

        timestamp = datetime.now().isoformat(timespec="seconds")
        event_id = f"v2_evt_{hashlib.sha1(f'{entry}|{timestamp}'.encode('utf-8')).hexdigest()[:12]}"
        canonical_memories = parse_memory_markdown(update)
        extracted = {
            "memory_update": update,
            "message_count": len(messages),
        }

        self.v2_db.insert_raw_event(
            event_id=event_id,
            ts=timestamp,
            session_key="v2",
            history_text=entry,
            plain_text=self._format_messages(messages),
            extracted=extracted,
            candidate_type="v2_snapshot",
        )
        self.v2_db.replace_canonical_snapshot(canonical_memories, event_id=event_id)
        self.v2_db.write_views()

    # 把一组消息 messages 格式化成一段纯文本，供后面的 memory consolidation prompt 使用。
    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """
        列表中每个元素是一个字典，比如：
        {
            "timestamp": "2026-03-27T10:20:33",
            "role": "user",
            "content": "帮我总结今天的工作",
            "tools_used": ["read_file", "web_search"]
        }
        """
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate the provided message chunk into MEMORY.md + HISTORY.md."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""
                Process this conversation and call the save_memory tool with your consolidation.

                ## Current Long-term Memory
                {current_memory or "(empty)"}

                ## Conversation to Process
                {self._format_messages(messages)}
                """

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            # args 格式不合法 
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            # args 虽然已经是 dict 了，但缺少约定的必填字段。
            if "history_entry" not in args or "memory_update" not in args:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = args["history_entry"]
            update = args["memory_update"]

            if entry is None or update is None:
                logger.warning("Memory consolidation: save_memory payload contains null required fields")
                return self._fail_or_raw_archive(messages)
            
            entry = _ensure_text(entry).strip()
            if not entry:
                # history_entry 虽然不为 None，但转成字符串并去掉空白后是空的
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            update = _ensure_text(update)
            if self.mode == "v2":
                try:
                    self._persist_v2(entry=entry, update=update, messages=messages)
                except Exception:
                    logger.exception("V2 memory persistence failed, falling back to legacy file writes")
                    self.append_history(entry)
                    if update != current_memory:
                        self.write_long_term(update)
            else:
                self.append_history(entry)
                if update != current_memory:
                    self.write_long_term(update)

            if self.shadow_pipeline is not None:
                try:
                    shadow_ok = await self.shadow_pipeline.ingest_chunk(messages, provider, model)
                    if not shadow_ok:
                        logger.warning("Shadow memory pipeline returned no structured write for this chunk")
                except Exception:
                    logger.exception("Shadow memory pipeline failed after legacy consolidation")

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
        mode: str = "legacy",
    ):
        self.mode = mode
        self.store = MemoryStore(workspace, mode=mode)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session, # 当前会话对象，有完整消息历史 session.messages
        tokens_to_remove: int, # 希望这次至少移走多少 token
    ) -> tuple[int, int] | None: # 形如 (idx, removed_tokens)
                                    # idx：切分位置
                                    # removed_tokens：如果切到这里，预计能移走多少 token
                                    # 如果找不到合适的切分点，就返回 None
        """
        在一个 session 的历史消息里，找一个“适合做记忆压缩的切分点”，
        让系统能一次性移走足够多的旧 token，而且尽量只在“用户回合边界”上切，不把一轮对话从中间劈开
        """
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None # last boundary允许这两种值，初始化为None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            """
            当扫描到一个新的 user 消息时，把这里当成一个合法切分边界。
            系统想尽量按“用户回合”切，而不是把 assistant/tool 消息截成半截。
            切在一个新的 user message 前面，意味着把前面的整段对话作为一个完整块拿去 consolidation。
            """
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    """
    它主要服务于 memory consolidation 的决策。MemoryConsolidator 需要知道：
    “当前 prompt 会不会太大，是否需要把旧消息压缩进 MEMORY/HISTORY 里。”
    所以它先模拟一次真实请求大小，再决定是否要归档旧历史。
    """
    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]: # （token数，估算方式）
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0) # 只取尚未被 consolidation 掉的历史，保证历史从合法的 tool-call 边界开始，不会留下孤立的 tool message
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        # session.key 的格式是 "channel:chat_id"，比如 "telegram:123456" 或 "cli:direct"
        # 为什么要拆这个：后面构造 prompt 时，ContextBuilder 会把当前时间、channel、chat_id 注入 runtime context，这些信息也会占 token
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        """
        不是直接拿 history 去估 token，而是构造一份“完整 prompt”。
        这里的 self._build_messages 是从外面注入进来的，实际上就是 ContextBuilder.build_messages()
        """
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True # 没有消息需要归档，直接返回成功
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        先估算当前 prompt 有多大
        如果还没超安全预算，就不动
        如果太大，就找一段“安全可切”的旧消息
        把那段消息做 consolidation
        更新 session.last_consolidated
        重新估算 prompt
        如果还是太大，继续下一轮
        """
        if not session.messages or self.context_window_tokens <= 0: # session.messages为空，或者
            return

        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < budget: # 还没超预算，不需要 consolidation
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
