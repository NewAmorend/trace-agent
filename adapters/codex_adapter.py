"""Adapter for OpenAI Codex CLI session JSONL format."""

import json
from adapters.base import BaseAdapter
from models import Step


class CodexAdapter(BaseAdapter):
    """Handles Codex CLI rollout JSONL files (rollout-*.jsonl) and exec streams."""

    def detect(self, data: object) -> bool:
        if isinstance(data, list):
            return any(
                isinstance(line, dict) and _is_codex_line(line)
                for line in data[:10]
            )
        return False

    def transform(self, data: list[dict]) -> tuple[str, str, list[Step]]:
        lines = data
        steps = []
        step_id = 0
        pending_thought = None
        task = 'Unknown task'
        has_failure = False
        last_turn_had_error = False

        for line in lines:
            event_type = line.get('type', '')

            # Extract task from first turn context or thread.started
            if event_type == 'thread.started' and not task:
                pass  # thread_id available but not the task prompt

            # Track turn-level failure
            if event_type == 'turn.failed':
                has_failure = True
                last_turn_had_error = True

            if event_type == 'turn.started':
                last_turn_had_error = False

            # Only process completed items for stable data
            if event_type != 'item.completed':
                continue

            item = line.get('item', {})
            if not item:
                continue

            details = item.get('details', item)  # exec stream uses flat structure
            item_type = details.get('type', '')

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
                        thought=pending_thought,
                        action='',
                        observation=text,
                        diff=None,
                    ))
                    pending_thought = None
                continue

            if item_type == 'command_execution':
                step_id += 1
                command = details.get('command', '')
                output = details.get('aggregated_output', '')
                status = details.get('status', '')

                if status == 'failed' or details.get('exit_code', 0) not in (0, None):
                    last_turn_had_error = True

                steps.append(Step(
                    step_id=step_id,
                    thought=pending_thought,
                    action=command,
                    observation=output if output else None,
                    diff=None,
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
                    thought=pending_thought,
                    action=f"apply_patch {paths}" if paths else "apply_patch",
                    observation=f"Changes: {'; '.join(diff_parts)}" if diff_parts else "Patch applied",
                    diff='\n'.join(diff_parts) if diff_parts else None,
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
                    obs = error.get('message', '')
                    last_turn_had_error = True
                elif result:
                    content = result.get('content', [])
                    obs = json.dumps(content) if content else None

                steps.append(Step(
                    step_id=step_id,
                    thought=pending_thought,
                    action=action,
                    observation=obs,
                    diff=None,
                ))
                pending_thought = None
                continue

            if item_type == 'error':
                step_id += 1
                message = details.get('message', 'Unknown error')
                last_turn_had_error = True
                steps.append(Step(
                    step_id=step_id,
                    thought=pending_thought,
                    action='',
                    observation=message,
                    diff=None,
                ))
                pending_thought = None
                continue

            if item_type == 'web_search':
                step_id += 1
                query = details.get('query', '')
                steps.append(Step(
                    step_id=step_id,
                    thought=pending_thought,
                    action=f"web_search {query}",
                    observation=None,
                    diff=None,
                ))
                pending_thought = None
                continue

        final_status = 'failed' if has_failure else 'success'

        # Try to extract task from first agent_message
        for step in steps:
            if step.observation and step.action == '' and not step.thought:
                task = step.observation[:200]
                break

        return task, final_status, steps


def _is_codex_line(line: dict) -> bool:
    t = line.get('type', '')
    return t in (
        'thread.started', 'turn.started', 'turn.completed', 'turn.failed',
        'item.started', 'item.updated', 'item.completed', 'error',
    ) or 'item' in line and isinstance(line['item'], dict)
