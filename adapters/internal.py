"""Adapter for the internal trajectory JSON format."""

from adapters.base import BaseAdapter
from models import Step


class InternalAdapter(BaseAdapter):
    """Handles the original {task, final_status, steps} JSON format."""

    def detect(self, data: object) -> bool:
        return (
            isinstance(data, dict)
            and 'steps' in data
            and isinstance(data['steps'], list)
        )

    def transform(self, data: dict) -> tuple[str, str, list[Step]]:
        task = data.get('task', 'Unknown task')
        final_status = data.get('final_status', 'unknown')
        raw_steps = data['steps']

        steps = []
        for raw_step in raw_steps:
            if 'step_id' not in raw_step:
                continue
            steps.append(Step(
                step_id=raw_step['step_id'],
                thought=raw_step.get('thought'),
                action=raw_step.get('action', ''),
                observation=raw_step.get('observation'),
                diff=raw_step.get('diff'),
            ))

        steps.sort(key=lambda s: s.step_id)
        return task, final_status, steps
