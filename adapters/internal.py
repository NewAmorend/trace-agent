"""Adapter for the internal trajectory JSON format."""

from adapters.base import BaseAdapter
from models import Step, Trajectory


class InternalAdapter(BaseAdapter):
    """Handles the original {task, final_status, steps} JSON format."""

    def detect(self, data: object) -> bool:
        return (
            isinstance(data, dict)
            and 'steps' in data
            and isinstance(data['steps'], list)
        )

    def transform(self, data: dict, source_path: str = "") -> Trajectory:
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
                item_type=raw_step.get('item_type', ''),
                exit_code=raw_step.get('exit_code'),
                status=raw_step.get('status'),
            ))

        steps.sort(key=lambda s: s.step_id)
        return Trajectory(
            source_path=source_path,
            task=task,
            final_status=final_status,
            steps=steps,
        )
