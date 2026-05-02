"""Adapter for Claude Code CLI --output-format stream-json."""

import json
from adapters.base import BaseAdapter
from models import Step, Trajectory


class ClaudeAdapter(BaseAdapter):
    """Handles Claude Code CLI stream-json output (claude -p ... --output-format stream-json)."""

    def detect(self, data: object) -> bool:
        if not isinstance(data, list):
            return False
        for line in data[:10]:
            if (
                isinstance(line, dict)
                and line.get('type') == 'system'
                and line.get('subtype') == 'init'
            ):
                return True
        return False

    def transform(self, data: list[dict], source_path: str = "") -> Trajectory:
        steps: list[Step] = []
        step_id = 0
        pending_thought: str | None = None
        thread_id: str | None = None
        has_failure = False
        failure_message: str | None = None
        tool_use_step_idx: dict[str, int] = {}

        for line in data:
            event_type = line.get('type', '')

            if event_type == 'system' and line.get('subtype') == 'init':
                thread_id = line.get('session_id') or thread_id
                continue

            if event_type == 'assistant':
                message = line.get('message', {})
                content_blocks = message.get('content', [])
                msg_id = message.get('id')

                for block in content_blocks:
                    block_type = block.get('type', '')

                    if block_type == 'thinking':
                        pending_thought = block.get('thinking') or None
                        continue

                    if block_type == 'text':
                        text = block.get('text', '').strip()
                        if text:
                            step_id += 1
                            steps.append(Step(
                                step_id=step_id,
                                event_id=msg_id,
                                thought=pending_thought,
                                action='',
                                observation=text,
                                diff=None,
                                item_type='agent_message',
                            ))
                            pending_thought = None
                        continue

                    if block_type == 'tool_use':
                        tool_name = block.get('name', '')
                        tool_id = block.get('id', '')
                        tool_input = block.get('input', {})

                        step_id += 1
                        if tool_name == 'Bash':
                            action = tool_input.get('command', '')
                            item_type = 'command_execution'
                            diff = None
                        elif tool_name in ('Write', 'Edit', 'MultiEdit', 'str_replace_based_edit_tool'):
                            path = (
                                tool_input.get('file_path')
                                or tool_input.get('path')
                                or '?'
                            )
                            action = f"apply_patch {path}"
                            item_type = 'file_change'
                            verb = 'write' if tool_name == 'Write' else 'edit'
                            diff = f"{verb}: {path}"
                        else:
                            action = tool_name
                            if tool_input:
                                action += f" {json.dumps(tool_input)}"
                            item_type = 'command_execution'
                            diff = None

                        steps.append(Step(
                            step_id=step_id,
                            event_id=tool_id,
                            thought=pending_thought,
                            action=action,
                            observation=None,
                            diff=diff,
                            item_type=item_type,
                        ))
                        tool_use_step_idx[tool_id] = len(steps) - 1
                        pending_thought = None
                continue

            if event_type == 'tool_result':
                tool_use_id = line.get('tool_use_id', '')
                obs = _extract_tool_result_text(line.get('content', []))
                if tool_use_id in tool_use_step_idx:
                    steps[tool_use_step_idx[tool_use_id]].observation = obs
                continue

            if event_type == 'result':
                if line.get('subtype') != 'success':
                    has_failure = True
                    failure_message = line.get('result') or line.get('subtype')
                continue

        return Trajectory(
            source_path=source_path,
            task=_infer_task_from_source(source_path),
            final_status='failed' if has_failure else 'success',
            steps=steps,
            thread_id=thread_id,
            failure_message=failure_message,
        )


def _extract_tool_result_text(content: list) -> str | None:
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get('type') == 'text':
            parts.append(item.get('text', ''))
        elif isinstance(item, str):
            parts.append(item)
    return '\n'.join(parts) if parts else None


def _infer_task_from_source(source_path: str) -> str:
    if not source_path:
        return 'Unknown task'
    parts = source_path.replace('\\', '/').split('/')
    stem = parts[-1].rsplit('.', 1)[0]
    if len(parts) >= 2:
        parent = parts[-2]
        name = f"{parent} {stem}"
    else:
        name = stem
    return name.replace('_', ' ').replace('-', ' ')
