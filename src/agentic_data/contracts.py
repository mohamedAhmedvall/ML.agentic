from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeState(StrEnum):
    BLOCKED = "blocked"
    READY = "ready"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Harness:
    model: str
    provider: str = "chatgpt_host"
    tools: tuple[str, ...] = ()
    approval: str = "never"
    max_retries: int = 0
    timeout_seconds: int = 300
    network: str = "deny"

    def __post_init__(self) -> None:
        if self.provider not in {"chatgpt_host", "github_copilot", "ollama"}:
            raise ValueError("unsupported provider")
        if self.max_retries < 0:
            raise ValueError("max_retries must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.network not in {"deny", "tool_allowlist"}:
            raise ValueError("unsupported network policy")


@dataclass(frozen=True)
class AgentNode:
    id: str
    role: str
    depends_on: tuple[str, ...] = ()
    harness: Harness = field(default_factory=lambda: Harness(model="default"))


@dataclass(frozen=True)
class Workflow:
    id: str
    objective: str
    nodes: tuple[AgentNode, ...]

    def node_map(self) -> dict[str, AgentNode]:
        mapping = {node.id: node for node in self.nodes}
        if len(mapping) != len(self.nodes):
            raise ValueError("node ids must be unique")
        for node in self.nodes:
            unknown = set(node.depends_on) - mapping.keys()
            if unknown:
                raise ValueError(f"node {node.id} has unknown dependencies: {sorted(unknown)}")
        return mapping


@dataclass(frozen=True)
class NodeResult:
    node_id: str
    state: NodeState
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 1

