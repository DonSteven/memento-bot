from __future__ import annotations

from nanobot.agent.memory_db import MemoryDatabase
from nanobot.agent.memory_service import MemoryService
from nanobot.config.schema import MemoryConfig


class FixedEmbedding:
    dimension = 3

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [self.vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)

    @staticmethod
    def vector(text: str) -> list[float]:
        lowered = text.lower()
        if "nanobot" in lowered or "语义" in text or "向量" in text:
            return [1.0, 0.0, 0.0]
        if "coffee" in lowered or "咖啡" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def memory_database(workspace) -> MemoryDatabase:
    return MemoryDatabase(workspace, vec_backend="array")


class TestMemoryService(MemoryService):
    __test__ = False

    def __init__(self, workspace, provider, model="test-model", **kwargs):
        embedder = kwargs.pop("embedder", FixedEmbedding())
        database = kwargs.pop("database", memory_database(workspace))
        config = kwargs.pop(
            "config",
            MemoryConfig.model_validate(
                {
                    "embedding": {"dimensions": embedder.dimension, "model": "fixed-test"},
                    "vector_similarity_threshold": 0.5,
                }
            ),
        )
        super().__init__(
            workspace,
            provider,
            model,
            database=database,
            embedder=embedder,
            config=config,
            **kwargs,
        )


def memory_service(workspace, provider, model="test-model", **kwargs) -> MemoryService:
    return TestMemoryService(workspace, provider, model, **kwargs)


def agent_loop(bus, provider, workspace, *args, **kwargs):
    from nanobot.agent.loop import AgentLoop

    model = kwargs.get("model") or provider.get_default_model()
    kwargs.setdefault("memory_service", memory_service(workspace, provider, model))
    return AgentLoop(bus, provider, workspace, *args, **kwargs)


class TestAgentLoop:
    __test__ = False

    def __new__(cls, bus, provider, workspace, *args, **kwargs):
        return agent_loop(bus, provider, workspace, *args, **kwargs)
