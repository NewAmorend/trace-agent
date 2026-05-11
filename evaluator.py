"""Evaluation orchestration for Codex trajectories."""

from pathlib import Path

from analyzer import locate_failure, score_suspicious_steps
from classifier import normalize_steps
from models import BatchSummary, EvalMetrics, EvalResult, NormalizedStep
from parser import load_trajectory
from patterns import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_SUSPICIOUS_STEPS_HIGH
from test_signals import looks_like_test_failure
from tree import build_trace_tree, render_trace_tree


SUPPORTED_SUFFIXES = {".jsonl", ".json"}


def evaluate_file(path: str | Path) -> EvalResult:
    """Evaluate one Codex trajectory file."""
    trajectory = load_trajectory(str(path))
    normalized = normalize_steps(trajectory.steps)
    scored = score_suspicious_steps(
        normalized,
        trajectory.task,
        trajectory.final_status,
    )
    tree_nodes = build_trace_tree(scored)
    tree_md = render_trace_tree(tree_nodes)
    diagnosis = locate_failure(scored, trajectory.final_status)
    metrics = compute_metrics(scored)

    return EvalResult(
        source_path=trajectory.source_path,
        task=trajectory.task,
        final_status=trajectory.final_status,
        normalized_steps=scored,
        tree_md=tree_md,
        diagnosis=diagnosis,
        metrics=metrics,
        failure_message=trajectory.failure_message,
    )


def discover_inputs(path: str | Path) -> list[Path]:
    """Return trajectory files from a file or directory path."""
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    files = [
        child for child in sorted(input_path.rglob("*"))
        if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No trajectory files found under {path}")
    return files


def compute_metrics(steps: list[NormalizedStep]) -> EvalMetrics:
    """Compute aggregate metrics for a scored trajectory."""
    total = len(steps)
    command_steps = sum(1 for s in steps if s.item_type == "command_execution")
    file_change_steps = sum(1 for s in steps if s.action_type == "edit_file")
    test_steps = sum(1 for s in steps if s.action_type == "run_test")
    failed_test_steps = sum(
        1 for s in steps
        if s.action_type == "run_test" and looks_like_test_failure(s.observation)
    )
    state_changes = sum(1 for s in steps if s.state_change)
    suspicious_steps = sum(1 for s in steps if s.suspicious_score > 0)
    max_score = max((s.suspicious_score for s in steps), default=0.0)

    return EvalMetrics(
        total_steps=total,
        command_steps=command_steps,
        file_change_steps=file_change_steps,
        test_steps=test_steps,
        failed_test_steps=failed_test_steps,
        state_changes=state_changes,
        suspicious_steps=suspicious_steps,
        max_suspicious_score=max_score,
        risk_level=_risk_level(max_score, suspicious_steps),
    )


def summarize_batch(results: list[EvalResult]) -> BatchSummary:
    """Summarize many evaluation results."""
    summary = BatchSummary(total=len(results))
    for result in results:
        if result.final_status.lower() == "success":
            summary.succeeded += 1
        else:
            summary.failed += 1

        if result.metrics.risk_level == "high":
            summary.high_risk += 1
        elif result.metrics.risk_level == "medium":
            summary.medium_risk += 1
        else:
            summary.low_risk += 1

        error_type = result.diagnosis.error_type or "unknown"
        summary.common_error_types[error_type] = (
            summary.common_error_types.get(error_type, 0) + 1
        )

    return summary


def _risk_level(max_score: float, suspicious_steps: int) -> str:
    if max_score >= RISK_THRESHOLD_HIGH or suspicious_steps >= RISK_THRESHOLD_SUSPICIOUS_STEPS_HIGH:
        return "high"
    if max_score >= RISK_THRESHOLD_MEDIUM or suspicious_steps:
        return "medium"
    return "low"
