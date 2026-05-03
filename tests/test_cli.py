"""Unit tests for main.py CLI behavior."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import build_parser, run_eval_command


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


if __name__ == "__main__":
    unittest.main()
