"""Parser for trajectory JSON files."""

import json
from models import Step


def load_trajectory(path: str) -> tuple[str, str, list[Step]]:
    """
    Load trajectory from JSON file.

    Returns:
        tuple of (task, final_status, list of Steps)

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if JSON is invalid or missing required fields
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    # Validate required fields
    if 'task' not in data:
        raise ValueError("Missing required field: 'task'")
    if 'final_status' not in data:
        raise ValueError("Missing required field: 'final_status'")
    if 'steps' not in data:
        raise ValueError("Missing required field: 'steps'")

    task = data['task']
    final_status = data['final_status']
    raw_steps = data['steps']

    # Build Step objects, tolerating missing fields
    steps = []
    for raw_step in raw_steps:
        if 'step_id' not in raw_step:
            continue  # Skip steps without ID

        step = Step(
            step_id=raw_step['step_id'],
            thought=raw_step.get('thought'),
            action=raw_step.get('action', ''),
            observation=raw_step.get('observation'),
            diff=raw_step.get('diff')
        )
        steps.append(step)

    # Sort by step_id
    steps.sort(key=lambda s: s.step_id)

    return task, final_status, steps
