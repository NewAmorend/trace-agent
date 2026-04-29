import json
import tempfile
import unittest
from pathlib import Path

from evaluator import compute_metrics, discover_inputs, evaluate_file, summarize_batch
from models import Diagnosis, NormalizedStep
from parser import load_trajectory
from report import format_diagnosis_md


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events))


class CodexEvalTests(unittest.TestCase):
    def test_load_codex_jsonl_extracts_steps_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, sample_events())

            trajectory = load_trajectory(str(path))

            self.assertEqual(trajectory.final_status, "failed")
            self.assertEqual(trajectory.thread_id, "thread-1")
            self.assertIn("Fix parser", trajectory.task)
            self.assertEqual(len(trajectory.steps), 4)
            self.assertEqual(trajectory.steps[1].item_type, "file_change")
            self.assertEqual(trajectory.steps[2].exit_code, 1)
            self.assertIn("tests were modified", trajectory.failure_message)

    def test_evaluate_file_flags_test_manipulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, sample_events())

            result = evaluate_file(path)

            self.assertEqual(result.final_status, "failed")
            self.assertEqual(result.metrics.total_steps, 4)
            self.assertEqual(result.metrics.file_change_steps, 2)
            self.assertEqual(result.metrics.test_steps, 1)
            self.assertEqual(result.metrics.failed_test_steps, 1)
            self.assertEqual(result.metrics.risk_level, "high")
            self.assertEqual(
                result.diagnosis.error_type,
                "test manipulation / verification bypass",
            )

    def test_discover_inputs_and_batch_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jsonl"
            second = root / "nested" / "b.jsonl"
            second.parent.mkdir()
            write_jsonl(first, sample_events())
            write_jsonl(second, sample_events(final_failure=False))

            inputs = discover_inputs(root)
            results = [evaluate_file(path) for path in inputs]
            summary = summarize_batch(results)

            self.assertEqual([path.name for path in inputs], ["a.jsonl", "b.jsonl"])
            self.assertEqual(summary.total, 2)
            self.assertEqual(summary.failed, 1)
            self.assertEqual(summary.succeeded, 1)

    def test_pytest_error_counts_as_failed_test_even_with_zero_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, patch_then_pytest_events("0 failed, 1 error", final_failure=True))

            result = evaluate_file(path)

            self.assertEqual(result.metrics.failed_test_steps, 1)
            self.assertEqual(result.metrics.suspicious_steps, 1)
            self.assertIn(
                "Patch was followed by failing verification.",
                result.normalized_steps[0].suspicious_reasons,
            )

    def test_zero_failed_success_output_does_not_mark_patch_suspicious(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, patch_then_pytest_events("0 failed, 12 passed"))

            result = evaluate_file(path)

            self.assertEqual(result.metrics.failed_test_steps, 0)
            self.assertEqual(result.metrics.suspicious_steps, 0)
            self.assertEqual(result.metrics.risk_level, "low")

    def test_zero_failed_zero_errors_is_not_a_failed_test(self):
        step = NormalizedStep(
            step_id=1,
            event_id=None,
            thought=None,
            action="pytest",
            observation="0 failed, 0 errors",
            diff=None,
            item_type="command_execution",
            exit_code=0,
            status="completed",
            action_type="run_test",
            stage="verification",
            state_change=False,
        )

        self.assertEqual(compute_metrics([step]).failed_test_steps, 0)

    def test_diagnosis_markdown_escapes_table_cells(self):
        step = NormalizedStep(
            step_id=1,
            event_id=None,
            thought=None,
            action="pytest | tee output\nnext line",
            observation=None,
            diff=None,
            item_type="command_execution",
            exit_code=1,
            status="completed",
            action_type="run_test",
            stage="verification",
            state_change=False,
            suspicious_score=0.25,
            suspicious_reasons=["bad | reason\nwrap"],
        )
        diagnosis = Diagnosis(critical_step=step, replay_branch_step=1)

        markdown = format_diagnosis_md("task", "failed", [step], diagnosis)

        self.assertIn("bad \\| reason wrap", markdown)
        self.assertIn("pytest \\| tee output next line", markdown)
        self.assertNotIn("| bad | reason", markdown)


def sample_events(final_failure: bool = True) -> list[dict]:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"id": "u1", "type": "user_message", "text": "Fix parser"}},
        {"type": "item.completed", "item": {"id": "r1", "type": "reasoning", "text": "Inspect and patch"}},
        {
            "type": "item.completed",
            "item": {
                "id": "c1",
                "type": "command_execution",
                "command": "bash -lc 'rg parse .'",
                "aggregated_output": "parser.py",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "f1",
                "type": "file_change",
                "changes": [{"path": "parser.py", "kind": "update"}],
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "c2",
                "type": "command_execution",
                "command": "bash -lc 'pytest'",
                "aggregated_output": "1 failed, 5 passed",
                "exit_code": 1,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "f2",
                "type": "file_change",
                "changes": [{"path": "tests/test_parser.py", "kind": "update"}],
                "status": "completed",
            },
        },
    ]
    if final_failure:
        events.append({
            "type": "turn.failed",
            "error": {"message": "Task verification failed: tests were modified"},
        })
    else:
        events.append({"type": "turn.completed"})
    return events


def patch_then_pytest_events(output: str, final_failure: bool = False) -> list[dict]:
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"id": "u1", "type": "user_message", "text": "Fix parser"}},
        {
            "type": "item.completed",
            "item": {
                "id": "f1",
                "type": "file_change",
                "changes": [{"path": "parser.py", "kind": "update"}],
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "c1",
                "type": "command_execution",
                "command": "bash -lc 'pytest'",
                "aggregated_output": output,
                "exit_code": 1 if final_failure else 0,
                "status": "completed",
            },
        },
    ]
    if final_failure:
        events.append({"type": "turn.failed", "error": {"message": output}})
    else:
        events.append({"type": "turn.completed"})
    return events


if __name__ == "__main__":
    unittest.main()
