"""Report generation for trajectory analysis."""

import json
import os
from dataclasses import asdict
from pathlib import Path

from models import BatchSummary, Diagnosis, EvalResult, NormalizedStep


def write_outputs(
    output_dir: str,
    task: str,
    final_status: str,
    steps: list[NormalizedStep],
    tree_md: str,
    diagnosis: Diagnosis
):
    """Write the legacy analysis output files."""
    os.makedirs(output_dir, exist_ok=True)

    # Write normalized_steps.json
    normalized_data = []
    for step in steps:
        step_dict = {
            'step_id': step.step_id,
            'event_id': step.event_id,
            'thought': step.thought,
            'action': step.action,
            'observation': step.observation,
            'diff': step.diff,
            'item_type': step.item_type,
            'exit_code': step.exit_code,
            'status': step.status,
            'action_type': step.action_type,
            'stage': step.stage,
            'state_change': step.state_change,
            'suspicious_score': step.suspicious_score,
            'suspicious_reasons': step.suspicious_reasons
        }
        normalized_data.append(step_dict)

    with open(os.path.join(output_dir, 'normalized_steps.json'), 'w') as f:
        json.dump(normalized_data, f, indent=2)

    # Write trace_tree.md
    with open(os.path.join(output_dir, 'trace_tree.md'), 'w') as f:
        f.write(tree_md)

    # Write diagnosis.json
    diagnosis_data = {
        'critical_step_id': diagnosis.critical_step.step_id if diagnosis.critical_step else None,
        'failure_stage': diagnosis.failure_stage,
        'error_type': diagnosis.error_type,
        'replay_branch_step': diagnosis.replay_branch_step,
        'replay_hint': diagnosis.replay_hint
    }

    with open(os.path.join(output_dir, 'diagnosis.json'), 'w') as f:
        json.dump(diagnosis_data, f, indent=2)

    # Write diagnosis.md
    diagnosis_md = format_diagnosis_md(task, final_status, steps, diagnosis)
    with open(os.path.join(output_dir, 'diagnosis.md'), 'w') as f:
        f.write(diagnosis_md)


def write_eval_result(output_dir: str | Path, result: EvalResult) -> None:
    """Write all output files for a single trajectory evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_outputs(
        str(output_path),
        result.task,
        result.final_status,
        result.normalized_steps,
        result.tree_md,
        result.diagnosis,
    )

    eval_data = {
        "source_path": result.source_path,
        "task": result.task,
        "final_status": result.final_status,
        "failure_message": result.failure_message,
        "metrics": asdict(result.metrics),
        "diagnosis": _diagnosis_dict(result.diagnosis),
    }
    with open(output_path / "eval_result.json", "w") as f:
        json.dump(eval_data, f, indent=2)

    with open(output_path / "eval_summary.md", "w") as f:
        f.write(format_eval_summary_md(result))


def write_batch_summary(output_dir: str | Path, summary: BatchSummary, results: list[EvalResult]) -> None:
    """Write batch-level summary files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": asdict(summary),
        "results": [
            {
                "source_path": result.source_path,
                "task": result.task,
                "final_status": result.final_status,
                "risk_level": result.metrics.risk_level,
                "max_suspicious_score": result.metrics.max_suspicious_score,
                "suspicious_steps": result.metrics.suspicious_steps,
                "error_type": result.diagnosis.error_type,
                "critical_step_id": (
                    result.diagnosis.critical_step.step_id
                    if result.diagnosis.critical_step else None
                ),
            }
            for result in results
        ],
    }
    with open(output_path / "batch_summary.json", "w") as f:
        json.dump(data, f, indent=2)

    with open(output_path / "batch_summary.md", "w") as f:
        f.write(format_batch_summary_md(summary, results))


def format_diagnosis_md(task: str, final_status: str, steps: list[NormalizedStep], diagnosis: Diagnosis) -> str:
    """Format diagnosis as Markdown."""
    lines = [
        "# Codex Trajectory Diagnosis",
        "",
        "## Task",
        task,
        "",
        "## Final Status",
        final_status,
        "",
    ]

    if diagnosis.critical_step:
        lines.extend([
            "## Failure Stage",
            diagnosis.failure_stage,
            "",
            "## Critical Step",
            f"Step {diagnosis.critical_step.step_id}",
            f"- Action: `{diagnosis.critical_step.action}`",
            f"- Type: {diagnosis.critical_step.action_type}",
            f"- Stage: {diagnosis.critical_step.stage}",
            f"- Suspicious Score: {diagnosis.critical_step.suspicious_score:.2f}",
            ""
        ])

    # Suspicious steps table
    suspicious = [s for s in steps if s.suspicious_score > 0]
    if suspicious:
        lines.extend([
            "## Suspicious Steps",
            "",
            "| Step ID | Stage | Action Type | Score | Reasons | Action |",
            "|---------|-------|-------------|-------|---------|--------|"
        ])

        for step in suspicious:
            reasons_text = "; ".join(step.suspicious_reasons) if step.suspicious_reasons else ""
            # Escape pipes in action
            action_escaped = step.action.replace('|', '\\|')
            lines.append(
                f"| {step.step_id} | {step.stage} | {step.action_type} | "
                f"{step.suspicious_score:.2f} | {reasons_text} | {action_escaped} |"
            )

        lines.append("")

    # Replay suggestion
    if diagnosis.replay_branch_step:
        lines.extend([
            "## Replay Suggestion",
            "",
            f"Branch at Step {diagnosis.replay_branch_step}",
            "",
            diagnosis.replay_hint,
            ""
        ])

    return "\n".join(lines)


def format_eval_summary_md(result: EvalResult) -> str:
    """Format a compact evaluation summary."""
    metrics = result.metrics
    lines = [
        "# Agent Trajectory Eval Summary",
        "",
        "## Source",
        result.source_path,
        "",
        "## Outcome",
        f"- Final status: {result.final_status}",
        f"- Risk level: {metrics.risk_level}",
        f"- Max suspicious score: {metrics.max_suspicious_score:.2f}",
        f"- Suspicious steps: {metrics.suspicious_steps}",
        "",
        "## Metrics",
        f"- Total steps: {metrics.total_steps}",
        f"- Commands: {metrics.command_steps}",
        f"- File changes: {metrics.file_change_steps}",
        f"- Test runs: {metrics.test_steps}",
        f"- Failed test runs: {metrics.failed_test_steps}",
        f"- State changes: {metrics.state_changes}",
        "",
    ]

    if result.failure_message:
        lines.extend(["## Codex Failure Message", result.failure_message, ""])

    if result.diagnosis.critical_step:
        lines.extend([
            "## Critical Step",
            f"Step {result.diagnosis.critical_step.step_id}: `{result.diagnosis.critical_step.action}`",
            "",
            result.diagnosis.replay_hint,
            "",
        ])

    return "\n".join(lines)


def format_batch_summary_md(summary: BatchSummary, results: list[EvalResult]) -> str:
    """Format batch summary as Markdown."""
    lines = [
        "# Agent Trajectory Eval Batch Summary",
        "",
        "## Totals",
        f"- Total trajectories: {summary.total}",
        f"- Codex success: {summary.succeeded}",
        f"- Codex failed: {summary.failed}",
        f"- High risk: {summary.high_risk}",
        f"- Medium risk: {summary.medium_risk}",
        f"- Low risk: {summary.low_risk}",
        "",
        "## Error Types",
    ]

    if summary.common_error_types:
        for error_type, count in sorted(
            summary.common_error_types.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"- {error_type}: {count}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Trajectories",
        "",
        "| Source | Status | Risk | Score | Suspicious | Error Type |",
        "|--------|--------|------|-------|------------|------------|",
    ])

    for result in results:
        source = Path(result.source_path).name
        lines.append(
            f"| {source} | {result.final_status} | {result.metrics.risk_level} | "
            f"{result.metrics.max_suspicious_score:.2f} | "
            f"{result.metrics.suspicious_steps} | {result.diagnosis.error_type} |"
        )

    lines.append("")
    return "\n".join(lines)


def _diagnosis_dict(diagnosis: Diagnosis) -> dict:
    return {
        'critical_step_id': diagnosis.critical_step.step_id if diagnosis.critical_step else None,
        'failure_stage': diagnosis.failure_stage,
        'error_type': diagnosis.error_type,
        'replay_branch_step': diagnosis.replay_branch_step,
        'replay_hint': diagnosis.replay_hint,
    }
