"""Incremental Qwen tool-call parser for streaming OpenAI deltas."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    payload: Any


class ToolCallStreamParser:
    """Parse text/tool-call blocks without leaking partial tool JSON.

    Text before a tool block can be emitted immediately.  Once a
    ``<tool_call>`` block starts, bytes are buffered until the closing tag and
    then emitted as an OpenAI-shaped tool call.
    """

    def __init__(self):
        self._buf = ""
        self._in_tool = False
        self._tool_idx = 0

    def feed(self, text: str) -> List[StreamEvent]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> List[StreamEvent]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> List[StreamEvent]:
        out: List[StreamEvent] = []
        while self._buf:
            if self._in_tool:
                close_idx = self._buf.find(TOOL_CALL_CLOSE)
                if close_idx < 0:
                    if final:
                        out.append(StreamEvent("text", TOOL_CALL_OPEN + self._buf))
                        self._buf = ""
                        self._in_tool = False
                    break
                raw = self._buf[:close_idx]
                self._buf = self._buf[close_idx + len(TOOL_CALL_CLOSE):]
                tool = self._parse_tool_call(raw)
                if tool is None:
                    out.append(
                        StreamEvent(
                            "text",
                            TOOL_CALL_OPEN + raw + TOOL_CALL_CLOSE,
                        )
                    )
                else:
                    out.append(StreamEvent("tool_call", tool))
                self._in_tool = False
                continue

            open_idx = self._buf.find(TOOL_CALL_OPEN)
            if open_idx < 0:
                emit_len = len(self._buf) if final else self._safe_text_len()
                if emit_len > 0:
                    out.append(StreamEvent("text", self._buf[:emit_len]))
                    self._buf = self._buf[emit_len:]
                break
            if open_idx > 0:
                out.append(StreamEvent("text", self._buf[:open_idx]))
            self._buf = self._buf[open_idx + len(TOOL_CALL_OPEN):]
            self._in_tool = True
        return out

    def _safe_text_len(self) -> int:
        """Leave a possible split '<tool_call>' suffix buffered."""

        max_hold = min(len(self._buf), len(TOOL_CALL_OPEN) - 1)
        for hold in range(max_hold, 0, -1):
            if TOOL_CALL_OPEN.startswith(self._buf[-hold:]):
                return len(self._buf) - hold
        return len(self._buf)

    def _parse_tool_call(self, raw: str) -> Optional[Dict[str, Any]]:
        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[^\n]*\n", "", s)
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        parsed = self._parse_json_tool_call(s)
        if parsed is None:
            parsed = self._parse_qwen_xml_tool_call(s)
        if parsed is None:
            return None
        name, args = parsed
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        idx = self._tool_idx
        self._tool_idx += 1
        return {
            "index": idx,
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        }

    @staticmethod
    def _parse_json_tool_call(s: str) -> Optional[tuple[str, Any]]:
        try:
            obj = json.loads(s)
        except Exception:
            return None
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None
        args = obj.get("arguments", obj.get("parameters", {}))
        return name, args

    @staticmethod
    def _parse_qwen_xml_tool_call(s: str) -> Optional[tuple[str, Dict[str, str]]]:
        """Parse Qwen's native tool-call format from its chat template:

        <function=name>
        <parameter=arg>
        value
        </parameter>
        </function>
        """
        m = re.fullmatch(
            r"\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*",
            s,
            flags=re.DOTALL,
        )
        if m is None:
            return None
        name = m.group(1).strip()
        body = m.group(2)
        if not name:
            return None
        args: Dict[str, str] = {}
        for pm in re.finditer(
                r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
                body,
                flags=re.DOTALL):
            key = pm.group(1).strip()
            if key:
                args[key] = pm.group(2).strip()
        if not args and body.strip():
            return None
        return name, args
