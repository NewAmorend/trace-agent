"""Unit tests for main.py CLI behavior."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import build_parser, run_eval_command
from tests._helpers import write_jsonl


def _codex_events(final_failure: bool = True) -> list[dict]:
    events = [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {"id": "u1", "type": "user_message", "text": "Fix"}},
        {
            "type": "item.completed",
            "item": {
                "id": "c1",
                "type": "command_execution",
                "command": "pytest",
                "aggregated_output": "1 failed" if final_failure else "5 passed",
                "exit_code": 1 if final_failure else 0,
                "status": "completed",
            },
        },
    ]
    if final_failure:
        events.append({"type": "turn.failed", "error": {"message": "fail"}})
    else:
        events.append({"type": "turn.completed"})
    return events


class EvalFormatJsonTests(unittest.TestCase):
    def test_eval_format_flag_defaults_to_text(self):
        parser = build_parser()
        args = parser.parse_args([
            "eval", "--input", "examples/codex_failed_run_001.jsonl",
            "--output", "out/test_format_text",
        ])
        self.assertEqual(args.format, "text")

    def test_eval_format_json_accepted(self):
        parser = build_parser()
        args = parser.parse_args([
            "eval",
            "--input", "examples/codex_failed_run_001.jsonl",
            "--output", "out/test_format_json",
            "--format", "json",
        ])
        self.assertEqual(args.format, "json")

    def test_eval_json_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            parser = build_parser()
            args = parser.parse_args([
                "eval",
                "--input", "examples/codex_failed_run_001.jsonl",
                "--output", str(Path(tmp) / "out"),
                "--format", "json",
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = run_eval_command(args)
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertIn("results", payload)
            self.assertIn("summary", payload)
            self.assertEqual(len(payload["results"]), 1)
            result = payload["results"][0]
            self.assertIn("final_status", result)
            self.assertIn("risk_level", result)
            self.assertIn("confidence", result)


class EvalCIExitCodeTests(unittest.TestCase):
    def test_ci_returns_1_for_failed_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "fail.jsonl"
            write_jsonl(input_path, _codex_events(final_failure=True))
            parser = build_parser()
            args = parser.parse_args([
                "eval", "--input", str(input_path),
                "--output", str(root / "out"), "--ci", "--quiet",
            ])
            code = run_eval_command(args)
            self.assertEqual(code, 1)

    def test_ci_returns_0_for_success_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "ok.jsonl"
            write_jsonl(input_path, _codex_events(final_failure=False))
            parser = build_parser()
            args = parser.parse_args([
                "eval", "--input", str(input_path),
                "--output", str(root / "out"), "--ci", "--quiet",
            ])
            code = run_eval_command(args)
            self.assertEqual(code, 0)


class EvalErrorHandlingTests(unittest.TestCase):
    def test_nonexistent_input_returns_2(self):
        parser = build_parser()
        args = parser.parse_args([
            "eval", "--input", "/nonexistent/path.jsonl",
            "--output", "/tmp/out_test", "--quiet",
        ])
        code = run_eval_command(args)
        self.assertEqual(code, 2)


class EvalBatchModeTests(unittest.TestCase):
    def test_batch_directory_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jsonl").write_text("\n".join(
                json.dumps(e) for e in _codex_events(final_failure=False)
            ))
            (root / "b.jsonl").write_text("\n".join(
                json.dumps(e) for e in _codex_events(final_failure=True)
            ))
            out_dir = root / "out"
            parser = build_parser()
            args = parser.parse_args([
                "eval", "--input", str(root),
                "--output", str(out_dir), "--quiet",
            ])
            code = run_eval_command(args)
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "batch_summary.json").exists())
            summary = json.loads((out_dir / "batch_summary.json").read_text())
            self.assertEqual(summary["summary"]["total"], 2)
            self.assertEqual(summary["summary"]["succeeded"], 1)
            self.assertEqual(summary["summary"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
