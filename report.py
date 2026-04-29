"""Report generation for trajectory analysis."""

import json
import os
from models import NormalizedStep, Diagnosis


def write_outputs(
    output_dir: str,
    task: str,
    final_status: str,
    steps: list[NormalizedStep],
    tree_md: str,
    diagnosis: Diagnosis
):
    """Write all analysis output files."""
    os.makedirs(output_dir, exist_ok=True)

    # Write normalized_steps.json
    normalized_data = []
    for step in steps:
        step_dict = {
            'step_id': step.step_id,
            'thought': step.thought,
            'action': step.action,
            'observation': step.observation,
            'diff': step.diff,
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
