"""Parser for trajectory files (JSON and JSONL)."""

import json
from models import Step
from adapters import get_adapter


def load_trajectory(path: str) -> tuple[str, str, list[Step]]:
    """
    Load trajectory from a JSON or JSONL file.

    Uses registered adapters to detect the format and transform data.

    Returns:
        tuple of (task, final_status, list of Steps)

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if format is unrecognized or data is invalid
    """
    try:
        with open(path, 'r') as f:
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
            f"No registered adapter could handle the data."
        )

    task, final_status, steps = adapter.transform(data)

    if not steps:
        raise ValueError(f"No valid steps found in {path}")

    return task, final_status, steps


def _parse_jsonl(raw: str) -> list[dict]:
    """Parse a JSONL string into a list of dicts."""
    lines = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return lines
