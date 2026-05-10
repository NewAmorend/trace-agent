import json
import tempfile
import unittest
from pathlib import Path

from adapters.claude_adapter import ClaudeAdapter
from evaluator import evaluate_file
from main import build_parser
from runner import build_claude_exec_command


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events))


def sample_claude_events(final_failure: bool = False) -> list[dict]:
    return [
        {"type": "system", "subtype": "init", "session_id": "sess-1", "tools": ["Bash", "Write"]},
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me inspect the code."},
                    {"type": "text", "text": "I'll look at the source files first."},
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "ls src/"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tu-1",
            "content": [{"type": "text", "text": "parser.py\nanalyzer.py"}],
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg-2",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-2",
                        "name": "Write",
                        "input": {"file_path": "src/parser.py", "content": "# fixed"},
                    },
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tu-2",
            "content": [{"type": "text", "text": "File written successfully"}],
        },
        {
            "type": "result",
            "subtype": "error_during_execution" if final_failure else "success",
            "result": "An error occurred" if final_failure else "Task completed",
            "session_id": "sess-1",
        },
    ]


class ClaudeAdapterDetectTests(unittest.TestCase):
    def test_detect_returns_true_for_claude_stream(self):
        adapter = ClaudeAdapter()
        self.assertTrue(adapter.detect([
            {"type": "system", "subtype": "init", "session_id": "abc"},
        ]))

    def test_detect_returns_false_for_codex_stream(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect([
            {"type": "thread.started", "thread_id": "t1"},
        ]))

    def test_detect_returns_false_for_non_list(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect({}))

    def test_detect_returns_false_for_system_non_init(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect([
            {"type": "system", "subtype": "other"},
        ]))


class ClaudeAdapterTransformTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()
        self.trajectory = self.adapter.transform(sample_claude_events(), "data/task.jsonl")

    def test_thread_id_extracted_from_init(self):
        self.assertEqual(self.trajectory.thread_id, "sess-1")

    def test_final_status_success(self):
        self.assertEqual(self.trajectory.final_status, "success")

    def test_step_count(self):
        self.assertEqual(len(self.trajectory.steps), 3)

    def test_text_block_becomes_agent_message(self):
        step = self.trajectory.steps[0]
        self.assertEqual(step.item_type, "agent_message")
        self.assertIn("source files", step.observation)

    def test_thinking_block_captured_as_thought(self):
        step = self.trajectory.steps[0]
        self.assertIn("inspect", step.thought.lower())

    def test_bash_step_is_command_execution(self):
        step = self.trajectory.steps[1]
        self.assertEqual(step.item_type, "command_execution")
        self.assertEqual(step.action, "ls src/")

    def test_bash_tool_result_merged_as_observation(self):
        step = self.trajectory.steps[1]
        self.assertIn("parser.py", step.observation)

    def test_write_step_is_file_change(self):
        step = self.trajectory.steps[2]
        self.assertEqual(step.item_type, "file_change")
        self.assertIn("src/parser.py", step.action)

    def test_write_tool_result_merged_as_observation(self):
        step = self.trajectory.steps[2]
        self.assertIn("written", step.observation)

    def test_error_result_sets_failed_status(self):
        traj = self.adapter.transform(sample_claude_events(final_failure=True))
        self.assertEqual(traj.final_status, "failed")
        self.assertIsNotNone(traj.failure_message)

    def test_task_inferred_from_source_path(self):
        self.assertEqual(self.trajectory.task, "data task")


class ClaudeEndToEndEvalTests(unittest.TestCase):
    def test_evaluate_claude_trajectory_produces_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claude_run.jsonl"
            write_jsonl(path, sample_claude_events())

            result = evaluate_file(path)

            self.assertEqual(result.final_status, "success")
            self.assertEqual(result.metrics.total_steps, 3)
            self.assertEqual(result.metrics.command_steps, 1)
            self.assertEqual(result.metrics.file_change_steps, 1)


class ClaudeRunnerCommandTests(unittest.TestCase):
    def test_build_claude_exec_command_structure(self):
        cmd = build_claude_exec_command("Fix the bug")
        self.assertEqual(cmd[0], "claude")
        self.assertIn("-p", cmd)
        self.assertIn("Fix the bug", cmd)
        self.assertIn("--output-format", cmd)
        idx = cmd.index("--output-format")
        self.assertEqual(cmd[idx + 1], "stream-json")

    def test_build_claude_exec_command_with_model(self):
        cmd = build_claude_exec_command("Fix the bug", model="claude-opus-4-7")
        self.assertIn("--model", cmd)
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx + 1], "claude-opus-4-7")

    def test_build_claude_exec_command_without_model_has_no_model_flag(self):
        cmd = build_claude_exec_command("Fix the bug")
        self.assertNotIn("--model", cmd)

    def test_build_claude_exec_command_includes_accept_edits_by_default(self):
        """Without --permission-mode acceptEdits, claude blocks Edits in non-
        interactive mode and produces zero-diff trajectories."""
        cmd = build_claude_exec_command("Fix the bug")
        self.assertIn("--permission-mode", cmd)
        idx = cmd.index("--permission-mode")
        self.assertEqual(cmd[idx + 1], "acceptEdits")


class ClaudeCLIFlagTests(unittest.TestCase):
    def test_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["run", "Fix bug", "--output", "out.jsonl"])
        self.assertEqual(args.agent, "codex")

    def test_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["run", "Fix bug", "--output", "out.jsonl", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")

    def test_lcb_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["lcb", "run"])
        self.assertEqual(args.agent, "codex")

    def test_lcb_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["lcb", "run", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")

    def test_swe_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["swe", "run", "instance-1"])
        self.assertEqual(args.agent, "codex")

    def test_swe_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["swe", "run", "instance-1", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")


if __name__ == "__main__":
    unittest.main()
