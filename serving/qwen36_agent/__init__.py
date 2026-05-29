"""Qwen3.6 agent-serving policy layer.

This package intentionally lives above the FlashRT execution contract:
sessions, prefix reuse, tool-calling, streaming, and scheduling are serving
policy.  The lower exec layer remains Buffer/Graph/Plan/Event only.
"""

from .prefix import PrefixMatch, longest_common_prefix
from .service import AgentRequest, AgentResult, AgentService
from .session import PrefixPlan, SessionRecord, SessionRegistry
from .tool_stream import StreamEvent, ToolCallStreamParser

__all__ = [
    "AgentRequest",
    "AgentResult",
    "AgentService",
    "PrefixMatch",
    "PrefixPlan",
    "SessionRecord",
    "SessionRegistry",
    "StreamEvent",
    "ToolCallStreamParser",
    "longest_common_prefix",
]
