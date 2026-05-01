import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluator import compute_metrics, discover_inputs, evaluate_file, summarize_batch
from main import build_parser, main as cli_main
from models import Diagnosis, NormalizedStep
from parser import load_trajectory
from report import format_diagnosis_md
from runner import build_codex_exec_command
from swe import build_prompt as build_swe_prompt


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

    def test_cli_eval_subcommand_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "run.jsonl"
            output_dir = root / "out"
            write_jsonl(input_path, sample_events())

            exit_code = cli_main([
                "eval",
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
                "--quiet",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "eval_result.json").exists())
            self.assertTrue((output_dir / "batch_summary.json").exists())

    def test_cli_legacy_eval_flags_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "run.jsonl"
            output_dir = root / "out"
            write_jsonl(input_path, sample_events())

            exit_code = cli_main([
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
                "--quiet",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "eval_summary.md").exists())

    def test_lcb_eval_defaults_to_generated_trajectory_paths(self):
        parser = build_parser()
        args = parser.parse_args(["lcb", "eval", "--quiet"])

        self.assertEqual(args.input, "data/lcb/trajectories")
        self.assertEqual(args.output, "out/lcb_eval")
        self.assertTrue(args.quiet)

    def test_run_command_parses_sandbox_capture_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "Fix failing tests",
            "--output",
            "data/runs/task.jsonl",
            "-C",
            "/repo",
            "--sandbox",
            "workspace-write",
            "--add-dir",
            "/tmp/cache",
            "--eval-output",
            "out/task",
        ])

        self.assertEqual(args.prompt, "Fix failing tests")
        self.assertEqual(args.output, "data/runs/task.jsonl")
        self.assertEqual(args.cwd, "/repo")
        self.assertEqual(args.sandbox, "workspace-write")
        self.assertEqual(args.add_dir, ["/tmp/cache"])
        self.assertEqual(args.eval_output, "out/task")

    def test_codex_exec_command_includes_sandbox_and_workspace(self):
        command = build_codex_exec_command(
            "Fix tests",
            cwd="/repo",
            sandbox="workspace-write",
            model="gpt-5.4",
            add_dirs=["/tmp/cache"],
            full_auto=True,
            skip_git_repo_check=True,
            ephemeral=True,
        )

        self.assertEqual(command[:5], ["codex", "exec", "--json", "-C", "/repo"])
        self.assertIn("--sandbox", command)
        self.assertIn("workspace-write", command)
        self.assertIn("--add-dir", command)
        self.assertIn("/tmp/cache", command)
        self.assertIn("--full-auto", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--ephemeral", command)
        self.assertEqual(command[-1], "Fix tests")

    def test_swe_run_command_parses_long_horizon_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "swe",
            "run",
            "astropy__astropy-12907",
            "--sandbox",
            "workspace-write",
            "--include-test-patch",
            "--apply-tests",
            "--eval-output",
            "out/swe_task",
        ])

        self.assertEqual(args.instance, "astropy__astropy-12907")
        self.assertEqual(args.sandbox, "workspace-write")
        self.assertTrue(args.include_test_patch)
        self.assertTrue(args.apply_tests)
        self.assertEqual(args.eval_output, "out/swe_task")

    def test_swe_eval_defaults_to_generated_trajectory_paths(self):
        parser = build_parser()
        args = parser.parse_args(["swe", "eval", "--quiet"])

        self.assertEqual(args.input, "data/swe/trajectories")
        self.assertEqual(args.output, "out/swe_eval")
        self.assertTrue(args.quiet)

    def test_swe_prompt_contains_repo_issue_and_instructions(self):
        prompt = build_swe_prompt({
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Fix separability matrix.",
            "hints_text": "Look at modeling.",
            "FAIL_TO_PASS": "[\"test_file.py::test_case\"]",
            "test_patch": "diff --git a/test_file.py b/test_file.py",
        }, include_test_patch=True)

        self.assertIn("astropy__astropy-12907", prompt)
        self.assertIn("Fix separability matrix.", prompt)
        self.assertIn("Known fail-to-pass tests", prompt)
        self.assertIn("Reference regression test patch", prompt)
        self.assertIn("Inspect the repository before editing.", prompt)

    def test_top_level_help_is_not_rewritten_to_eval_command(self):
        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as ctx:
                cli_main(["--help"])

        self.assertEqual(ctx.exception.code, 0)

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

    def test_repeated_empty_agent_messages_are_not_suspicious_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, agent_message_only_events())

            result = evaluate_file(path)

            self.assertEqual(result.final_status, "success")
            self.assertEqual(result.metrics.suspicious_steps, 0)
            self.assertEqual(result.metrics.risk_level, "low")

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


def agent_message_only_events() -> list[dict]:
    return [
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {
                "id": "a1",
                "type": "agent_message",
                "text": "I will inspect the task.",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "a2",
                "type": "agent_message",
                "text": "The sandbox blocked the command, so I will answer directly.",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "a3",
                "type": "agent_message",
                "text": "```python\nprint('solution')\n```",
            },
        },
        {"type": "turn.completed"},
    ]


if __name__ == "__main__":
    unittest.main()
