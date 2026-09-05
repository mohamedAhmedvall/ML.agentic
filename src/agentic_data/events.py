from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RuntimeEvent:
    type: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None
    time: str = field(default_factory=_now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "type": self.type,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "payload": self.payload,
        }


class EventBus:
    """Small synchronous event bus used by the runtime and consumable by CLI/UI/MCP."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[RuntimeEvent], None]] = []

    def subscribe(self, callback: Callable[[RuntimeEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(
        self,
        event_type: str,
        run_id: str,
        payload: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(event_type, run_id, payload or {}, node_id=node_id)
        for callback in tuple(self._subscribers):
            callback(event)
        return event
