"""Parser for trajectory files (JSON and JSONL)."""

import json
from pathlib import Path

from models import Step, Trajectory  # noqa: F401  (Trajectory used in type hints)
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
        with open(input_path, 'r') as f:
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

    sidecar = _load_sidecar(input_path)
    if sidecar:
        _enrich_with_sidecar(trajectory, sidecar)

    return trajectory


def _load_sidecar(trajectory_path: Path) -> dict | None:
    """Look for `<trajectory>.sidecar.json` next to the trajectory file."""
    candidates = [
        trajectory_path.with_suffix(trajectory_path.suffix + ".sidecar.json"),
        trajectory_path.with_name(trajectory_path.stem + ".sidecar.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
    return None


def _enrich_with_sidecar(trajectory: 'Trajectory', sidecar: dict) -> None:
    """Stamp task identity and verified-grade fields onto the trajectory."""
    metadata = {
        key: sidecar.get(key)
        for key in (
            "instance_id", "repo", "base_commit",
            "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS",
        )
        if sidecar.get(key) is not None
    }
    if metadata:
        trajectory.task_metadata = metadata
        instance_id = metadata.get("instance_id")
        repo = metadata.get("repo")
        if instance_id:
            label = instance_id if not repo else f"{repo}#{instance_id}"
            trajectory.task = label

    if "verified_pass" in sidecar:
        trajectory.verified_pass = sidecar.get("verified_pass")
    trajectory.grader_status = sidecar.get("grader_status")
    grader_details = {
        key: sidecar.get(key)
        for key in (
            "fail_to_pass_results", "pass_to_pass_results",
            "agent_diff", "grader_message",
        )
        if sidecar.get(key) is not None
    }
    if grader_details:
        trajectory.grader_details = grader_details


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
