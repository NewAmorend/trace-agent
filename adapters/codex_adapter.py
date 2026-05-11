"""Adapter for OpenAI Codex CLI session JSONL format."""

import json
from adapters.base import BaseAdapter, infer_task_from_source
from models import Step, Trajectory


class CodexAdapter(BaseAdapter):
    """Handles Codex CLI rollout JSONL files (rollout-*.jsonl) and exec streams."""

    def detect(self, data: object) -> bool:
        if isinstance(data, list):
            return any(
                isinstance(line, dict) and _is_codex_line(line)
                for line in data[:10]
            )
        return False

    def transform(self, data: list[dict], source_path: str = "") -> Trajectory:
        lines = data
        steps = []
        step_id = 0
        pending_thought = None
        task = 'Unknown task'
        thread_id = None
        has_failure = False
        failure_message = None

        for line in lines:
            event_type = line.get('type', '')

            if event_type == 'thread.started':
                thread_id = line.get('thread_id') or line.get('id') or thread_id

            if event_type == 'turn.failed':
                has_failure = True
                error = line.get('error') or {}
                if isinstance(error, dict):
                    failure_message = error.get('message') or failure_message
                elif isinstance(error, str):
                    failure_message = error

            # Only process completed items for stable data
            if event_type != 'item.completed':
                continue

            item = line.get('item', {})
            if not item:
                continue

            details = item.get('details', item)  # exec stream uses flat structure
            item_type = details.get('type', '')

            if item_type in ('user_message', 'message'):
                text = _extract_text(details)
                if text and task == 'Unknown task':
                    task = text[:500]
                continue

            if item_type == 'reasoning':
                text = details.get('text', '')
                if text:
                    pending_thought = text
                continue

            if item_type == 'agent_message':
                text = details.get('text', '')
                if text and text.strip():
                    step_id += 1
                    steps.append(Step(
                        step_id=step_id,
                        event_id=item.get('id'),
                        thought=pending_thought,
                        action='',
                        observation=text,
                        diff=None,
                        item_type=item_type,
                    ))
                    pending_thought = None
                continue

            if item_type == 'command_execution':
                step_id += 1
                command = details.get('command', '')
                output = details.get('aggregated_output', '')
                status = details.get('status', '')
                exit_code = details.get('exit_code')

                steps.append(Step(
                    step_id=step_id,
                    event_id=item.get('id'),
                    thought=pending_thought,
                    action=command,
                    observation=output if output else None,
                    diff=None,
                    item_type=item_type,
                    exit_code=exit_code,
                    status=status,
                ))
                pending_thought = None
                continue

            if item_type == 'file_change':
                step_id += 1
                changes = details.get('changes', [])
                paths = ', '.join(c.get('path', '?') for c in changes)
                change_kinds = [c.get('kind', 'update') for c in changes]

                diff_parts = []
                for change in changes:
                    kind = change.get('kind', 'update')
                    path = change.get('path', '?')
                    diff_parts.append(f"{kind}: {path}")

                steps.append(Step(
                    step_id=step_id,
                    event_id=item.get('id'),
                    thought=pending_thought,
                    action=f"apply_patch {paths}" if paths else "apply_patch",
                    observation=f"Changes: {'; '.join(diff_parts)}" if diff_parts else "Patch applied",
                    diff='\n'.join(diff_parts) if diff_parts else None,
                    item_type=item_type,
                    status=details.get('status'),
                ))
                pending_thought = None
                continue

            if item_type == 'mcp_tool_call':
                step_id += 1
                server = details.get('server', '')
                tool = details.get('tool', '')
                arguments = details.get('arguments', {})
                result = details.get('result')
                error = details.get('error')

                action = f"{server}/{tool}" if server else tool
                if arguments and isinstance(arguments, dict):
                    action += f" {json.dumps(arguments)}"

                obs = None
                if error:
                    if isinstance(error, dict):
                        obs = error.get('message', '')
                    elif isinstance(error, str):
                        obs = error
                    has_failure = True
                elif result:
                    content = result.get('content', [])
                    obs = json.dumps(content) if content else None

                steps.append(Step(
                    step_id=step_id,
                    event_id=item.get('id'),
                    thought=pending_thought,
                    action=action,
                    observation=obs,
                    diff=None,
                    item_type=item_type,
                ))
                pending_thought = None
                continue

            if item_type == 'error':
                step_id += 1
                message = details.get('message', 'Unknown error')
                has_failure = True
                failure_message = message
                steps.append(Step(
                    step_id=step_id,
                    event_id=item.get('id'),
                    thought=pending_thought,
                    action='',
                    observation=message,
                    diff=None,
                    item_type=item_type,
                ))
                pending_thought = None
                continue

            if item_type == 'web_search':
                step_id += 1
                query = details.get('query', '')
                steps.append(Step(
                    step_id=step_id,
                    event_id=item.get('id'),
                    thought=pending_thought,
                    action=f"web_search {query}",
                    observation=None,
                    diff=None,
                    item_type=item_type,
                ))
                pending_thought = None
                continue

        final_status = 'failed' if has_failure else 'success'

        if task == 'Unknown task':
            task = infer_task_from_source(source_path)

        return Trajectory(
            source_path=source_path,
            task=task,
            final_status=final_status,
            steps=steps,
            thread_id=thread_id,
            failure_message=failure_message,
        )


def _is_codex_line(line: dict) -> bool:
    t = line.get('type', '')
    return t in (
        'thread.started', 'turn.started', 'turn.completed', 'turn.failed',
        'item.started', 'item.updated', 'item.completed', 'error',
    ) or 'item' in line and isinstance(line['item'], dict)


def _extract_text(details: dict) -> str:
    """Best-effort extraction for Codex user/message item payloads."""
    text = details.get('text')
    if isinstance(text, str):
        return text.strip()

    content = details.get('content')
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get('text') or item.get('content')
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts).strip()

    return ""
