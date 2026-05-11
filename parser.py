"""Parser for trajectory files (JSON and JSONL)."""

import json
from pathlib import Path

from models import Step, Trajectory
from adapters import get_adapter


def load_trajectory(path: str) -> Trajectory:
    """
    Load trajectory from a JSON or JSONL file.

    Uses registered adapters to detect the format and transform data.

    Returns:
        Trajectory

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if format is unrecognized or data is invalid
    """
    input_path = Path(path)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    # Try JSON first (single object or array)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try JSONL (one JSON object per line)
        data = _parse_jsonl(raw)

    adapter = get_adapter(data)
    if adapter is None:
        raise ValueError(
            f"Unrecognized trajectory format in {path}. "
            f"This evaluator currently supports Codex JSONL session/exec streams."
        )

    trajectory = adapter.transform(data, str(input_path))

    if not trajectory.steps:
        raise ValueError(f"No valid steps found in {path}")

    return trajectory


def load_trajectory_legacy(path: str) -> tuple[str, str, list[Step]]:
    """Compatibility wrapper for older callers."""
    trajectory = load_trajectory(path)
    return trajectory.task, trajectory.final_status, trajectory.steps


def _parse_jsonl(raw: str) -> list[dict]:
    """Parse a JSONL string into a list of dicts."""
    lines = []
    for line_number, line in enumerate(raw.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return lines
