"""Bounded, same-process Dashboard invalidation broadcasts."""

from __future__ import annotations

import asyncio


class DashboardEvents:
    def __init__(self, queue_size: int = 32):
        self.queue_size = queue_size
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event_type: str, *, run_id: str | None = None,
                job_id: str | None = None) -> None:
        event = {"type": event_type}
        if run_id is not None:
            event["run_id"] = run_id
        if job_id is not None:
            event["job_id"] = job_id
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)

    def close(self) -> None:
        for queue in tuple(self._subscribers):
            self._subscribers.discard(queue)
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
