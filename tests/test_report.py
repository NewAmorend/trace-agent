"""Unit tests for report.py."""

import json
import tempfile
import unittest
from pathlib import Path

from models import Diagnosis, EvalMetrics, EvalResult
from report import (
    format_batch_summary_md,
    format_diagnosis_md,
    format_eval_summary_md,
    write_eval_result,
)
from tests._helpers import make_normalized_step


def make_eval_result(
    *,
    final_status: str = "success",
    risk_level: str = "low",
    suspicious_steps: int = 0,
    critical: bool = False,
) -> EvalResult:
    steps = [
        make_normalized_step(step_id=1, action="ls", action_type="other"),
    ]
    if critical:
        steps.append(
            make_normalized_step(
                step_id=2,
                action="apply_patch tests/x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
            )
        )

    diagnosis = Diagnosis()
    if critical:
        diagnosis.critical_step = steps[1]
        diagnosis.failure_stage = "patching"
        diagnosis.error_type = "test manipulation / verification bypass"
        diagnosis.replay_branch_step = 2
        diagnosis.replay_hint = "Investigate impl, not tests."

    metrics = EvalMetrics(
        total_steps=len(steps),
        suspicious_steps=suspicious_steps,
        max_suspicious_score=0.45 if critical else 0.0,
        risk_level=risk_level,
    )

    return EvalResult(
        source_path="examples/dummy.jsonl",
        task="dummy task",
        final_status=final_status,
        normalized_steps=steps,
        tree_md="# Trace Tree\n",
        diagnosis=diagnosis,
        metrics=metrics,
    )


class FormatDiagnosisMdTests(unittest.TestCase):
    def test_includes_task_and_status(self):
        result = make_eval_result()
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertIn("dummy task", out)
        self.assertIn("success", out)

    def test_includes_critical_step_block_when_present(self):
        result = make_eval_result(critical=True, final_status="failed")
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertIn("Critical Step", out)
        self.assertIn("Step 2", out)

    def test_omits_critical_block_when_absent(self):
        result = make_eval_result()
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertNotIn("Critical Step", out)


class FormatEvalSummaryMdTests(unittest.TestCase):
    def test_renders_metrics(self):
        result = make_eval_result()
        out = format_eval_summary_md(result)
        self.assertIn("Total steps: 1", out)
        self.assertIn("Risk level: low", out)


class FormatBatchSummaryMdTests(unittest.TestCase):
    def test_empty_batch_renders_none(self):
        from models import BatchSummary
        out = format_batch_summary_md(BatchSummary(), [])
        self.assertIn("none", out)

    def test_pipe_in_action_is_escaped(self):
        result = make_eval_result(critical=True, final_status="failed")
        result.normalized_steps[1].action = "echo a | tee b"
        from evaluator import summarize_batch
        out = format_batch_summary_md(summarize_batch([result]), [result])
        self.assertNotIn("a | tee b |", out)


class WriteEvalResultTests(unittest.TestCase):
    def test_writes_four_files_plus_eval_result(self):
        result = make_eval_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "result"
            write_eval_result(out_dir, result)
            for name in (
                "normalized_steps.json",
                "trace_tree.md",
                "diagnosis.json",
                "diagnosis.md",
                "eval_result.json",
                "eval_summary.md",
            ):
                self.assertTrue((out_dir / name).exists(), f"missing: {name}")

    def test_eval_result_json_round_trips(self):
        result = make_eval_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "r"
            write_eval_result(out_dir, result)
            data = json.loads((out_dir / "eval_result.json").read_text())
            self.assertEqual(data["task"], "dummy task")
            self.assertEqual(data["final_status"], "success")


if __name__ == "__main__":
    unittest.main()
