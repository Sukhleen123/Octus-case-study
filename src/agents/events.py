"""
Trace event system for multi-agent observability.

Events are emitted by each agent and collected by the orchestrator.
The UI renders them as an expandable step-by-step trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    RETRIEVAL_RESULTS = "retrieval_results"
    CITATIONS_EMITTED = "citations_emitted"


@dataclass
class TraceEvent:
    """A single trace event emitted during agent execution."""
    event_type: EventType
    agent_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "agent_name": self.agent_name,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


def agent_start(agent_name: str, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.AGENT_START, agent_name, payload=kwargs)


def agent_end(agent_name: str, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.AGENT_END, agent_name, payload=kwargs)


def tool_call_start(agent_name: str, tool: str, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.TOOL_CALL_START, agent_name, payload={"tool": tool, **kwargs})


def tool_call_end(agent_name: str, tool: str, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.TOOL_CALL_END, agent_name, payload={"tool": tool, **kwargs})


def retrieval_results(agent_name: str, count: int, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.RETRIEVAL_RESULTS, agent_name, payload={"count": count, **kwargs})


def citations_emitted(agent_name: str, count: int, **kwargs) -> TraceEvent:
    return TraceEvent(EventType.CITATIONS_EMITTED, agent_name, payload={"count": count, **kwargs})
