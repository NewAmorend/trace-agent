"""Adapter for DeerFlow JSONL run event files.

DeerFlow (by ByteDance) stores agent run events in JSONL format at
``.deer-flow/threads/{thread_id}/runs/{run_id}.jsonl`` when configured
with ``run_events.backend: jsonl``.

Each line is a JSON object with the following canonical shape::

    {
        "thread_id": "uuid",
        "run_id": "uuid",
        "event_type": "run.start|llm.human.input|llm.ai.response|...",
        "category": "message|trace|lifecycle|outputs|error|middleware",
        "content": "<dict or string>",
        "metadata": {"caller": "lead_agent|subagent:name|middleware:name", ...},
        "seq": 42,
        "created_at": "ISO-8601"
    }

This adapter maps DeerFlow events onto the unified ``Step`` / ``Trajectory``
model so that the rest of the pipeline (classifier, analyzer, tree) works
unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from adapters.base import BaseAdapter, infer_task_from_source
from models import Step, Trajectory

_DEERFLOW_EVENT_TYPES = frozenset({
    "run.start",
    "run.end",
    "run.error",
    "llm.human.input",
    "llm.ai.response",
    "llm.tool.result",
    "llm.error",
})

_DEERFLOW_CATEGORIES = frozenset({
    "message",
    "trace",
    "lifecycle",
    "outputs",
    "error",
    "middleware",
})


class DeerFlowAdapter(BaseAdapter):
    """Adapter for DeerFlow JSONL run event files."""

    def detect(self, data: object) -> bool:
        if not isinstance(data, list) or not data:
            return False
        deerflow_count = 0
        for line in data[:20]:
            if not isinstance(line, dict):
                continue
            et = line.get("event_type", "")
            cat = line.get("category", "")
            if et in _DEERFLOW_EVENT_TYPES or cat in _DEERFLOW_CATEGORIES:
                deerflow_count += 1
            if et.startswith("middleware:"):
                deerflow_count += 1
        return deerflow_count >= 2

    def transform(self, data: list[dict], source_path: str = "") -> Trajectory:
        steps: list[Step] = []
        step_id = 0
        thread_id: str | None = None
        task = "Unknown task"
        has_failure = False
        failure_message: str | None = None
        caller_stack: list[str] = []

        for record in data:
            event_type = record.get("event_type", "")
            category = record.get("category", "")
            content = record.get("content", "")
            metadata = record.get("metadata", {})
            caller = metadata.get("caller", "lead_agent")

            if event_type == "run.start":
                thread_id = record.get("thread_id") or thread_id
                continue

            if event_type == "llm.human.input":
                text = _extract_msg_text(content)
                if text and task == "Unknown task":
                    task = text[:500]
                continue

            if event_type == "run.end":
                continue

            if event_type == "run.error":
                has_failure = True
                if isinstance(content, str):
                    failure_message = content
                elif isinstance(content, dict):
                    failure_message = content.get("message", str(content))
                continue

            if event_type == "llm.error":
                has_failure = True
                error_text = content if isinstance(content, str) else str(content)
                step_id += 1
                steps.append(Step(
                    step_id=step_id,
                    thought=None,
                    action="llm_error",
                    observation=error_text,
                    diff=None,
                    item_type="error",
                    status="error",
                ))
                continue

            if event_type == "llm.ai.response":
                msg = content if isinstance(content, dict) else {}
                text = _extract_msg_text(msg)
                tool_calls = msg.get("tool_calls") or []
                usage = metadata.get("usage") or {}
                latency_ms = metadata.get("latency_ms")

                if tool_calls:
                    for tc in tool_calls:
                        tc_name = tc.get("name", "unknown_tool")
                        tc_args = tc.get("args", {})
                        tc_id = tc.get("id", "")

                        step_id += 1
                        action = tc_name
                        if tc_args:
                            action += f" {_truncate(json.dumps(tc_args, default=str), 500)}"

                        steps.append(Step(
                            step_id=step_id,
                            event_id=tc_id,
                            thought=text if text else None,
                            action=action,
                            observation=None,
                            diff=None,
                            item_type="tool_call",
                        ))

                        if caller.startswith("subagent:"):
                            caller_stack.append(caller)
                elif text and text.strip():
                    step_id += 1
                    steps.append(Step(
                        step_id=step_id,
                        thought=None,
                        action="",
                        observation=text,
                        diff=None,
                        item_type="agent_message",
                    ))

                continue

            if event_type == "llm.tool.result":
                msg = content if isinstance(content, dict) else {}
                result_text = _extract_tool_result(msg)
                tool_name = msg.get("name", "")
                tool_call_id = msg.get("tool_call_id", "")
                status = msg.get("status")

                step_id += 1
                action = tool_name if tool_name else "tool_result"

                steps.append(Step(
                    step_id=step_id,
                    event_id=tool_call_id,
                    thought=None,
                    action=action,
                    observation=result_text,
                    diff=None,
                    item_type="tool_result",
                    status=status,
                ))

                if caller_stack:
                    caller_stack.pop()
                continue

            if event_type.startswith("middleware:"):
                tag = event_type[len("middleware:"):]
                mw_content = content if isinstance(content, dict) else {}
                mw_name = mw_content.get("name", "")
                mw_action = mw_content.get("action", "")
                mw_changes = mw_content.get("changes", {})
                mw_hook = mw_content.get("hook", "")

                step_id += 1
                action = f"middleware:{tag}"
                if mw_action:
                    action += f" {mw_action}"

                obs_parts = []
                if mw_name:
                    obs_parts.append(f"[{mw_name}]")
                if mw_hook:
                    obs_parts.append(f"hook={mw_hook}")
                if mw_changes:
                    obs_parts.append(_truncate(json.dumps(mw_changes, default=str), 500))
                observation = " ".join(obs_parts) if obs_parts else None

                steps.append(Step(
                    step_id=step_id,
                    thought=None,
                    action=action,
                    observation=observation,
                    diff=None,
                    item_type="middleware",
                ))
                continue

        final_status = "failed" if has_failure else "success"

        if task == "Unknown task":
            task = infer_task_from_source(source_path)

        return Trajectory(
            source_path=source_path,
            task=task,
            final_status=final_status,
            steps=steps,
            thread_id=thread_id,
            failure_message=failure_message,
        )


def _extract_msg_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""

    text = content.get("content", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text", "")
                if t:
                    parts.append(t)
        return " ".join(parts).strip()
    return ""


def _extract_tool_result(msg: dict) -> str | None:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts) if parts else None
    return None


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
