"""Tests for the DeerFlow adapter and its integration with the analysis pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluator import evaluate_file
from parser import load_trajectory
from tests._helpers import write_jsonl


class DeerFlowAdapterDetectTests(unittest.TestCase):
    def test_detects_valid_deerflow_jsonl(self):
        from adapters.deerflow_adapter import DeerFlowAdapter
        adapter = DeerFlowAdapter()
        data = [
            {"event_type": "run.start", "category": "trace", "content": {"chain": "agent"}},
            {"event_type": "llm.human.input", "category": "message", "content": {"content": "Hello"}},
        ]
        self.assertTrue(adapter.detect(data))

    def test_rejects_empty_list(self):
        from adapters.deerflow_adapter import DeerFlowAdapter
        adapter = DeerFlowAdapter()
        self.assertFalse(adapter.detect([]))

    def test_rejects_codex_format(self):
        from adapters.deerflow_adapter import DeerFlowAdapter
        adapter = DeerFlowAdapter()
        data = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "item.completed", "item": {"id": "u1", "type": "user_message", "text": "Hi"}},
        ]
        self.assertFalse(adapter.detect(data))

    def test_rejects_single_event(self):
        from adapters.deerflow_adapter import DeerFlowAdapter
        adapter = DeerFlowAdapter()
        data = [
            {"event_type": "run.start", "category": "trace", "content": {}},
        ]
        self.assertFalse(adapter.detect(data))

    def test_detects_middleware_events(self):
        from adapters.deerflow_adapter import DeerFlowAdapter
        adapter = DeerFlowAdapter()
        data = [
            {"event_type": "run.start", "category": "trace", "content": {}},
            {"event_type": "middleware:summarize", "category": "middleware", "content": {}},
        ]
        self.assertTrue(adapter.detect(data))


class DeerFlowAdapterTransformTests(unittest.TestCase):
    def test_load_deerflow_jsonl_extracts_task_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_basic_events())

            traj = load_trajectory(str(path))

            self.assertEqual(traj.final_status, "failed")
            self.assertIsNotNone(traj.thread_id)
            self.assertIn("Fix the parser bug", traj.task)
            self.assertTrue(len(traj.steps) > 0)

    def test_llm_ai_response_with_tool_calls_creates_tool_call_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_tool_call_events())

            traj = load_trajectory(str(path))

            tool_call_steps = [s for s in traj.steps if s.item_type == "tool_call"]
            self.assertTrue(len(tool_call_steps) >= 1)
            self.assertIn("bash", tool_call_steps[0].action.lower())

    def test_llm_tool_result_creates_tool_result_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_tool_call_events())

            traj = load_trajectory(str(path))

            result_steps = [s for s in traj.steps if s.item_type == "tool_result"]
            self.assertTrue(len(result_steps) >= 1)

    def test_middleware_events_creates_middleware_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_middleware_events())

            traj = load_trajectory(str(path))

            mw_steps = [s for s in traj.steps if s.item_type == "middleware"]
            self.assertEqual(len(mw_steps), 2)
            self.assertIn("middleware:summarize", mw_steps[0].action)
            self.assertIn("middleware:loop_detect", mw_steps[1].action)

    def test_run_error_marks_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_failure_events())

            traj = load_trajectory(str(path))

            self.assertEqual(traj.final_status, "failed")
            self.assertIn("Agent execution failed", traj.failure_message)

    def test_llm_error_creates_error_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_llm_error_events())

            traj = load_trajectory(str(path))

            error_steps = [s for s in traj.steps if s.item_type == "error"]
            self.assertEqual(len(error_steps), 1)
            self.assertEqual(traj.final_status, "failed")

    def test_agent_message_text_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_agent_message_events())

            traj = load_trajectory(str(path))

            agent_steps = [s for s in traj.steps if s.item_type == "agent_message"]
            self.assertTrue(len(agent_steps) >= 1)
            self.assertIn("I will fix the parser", agent_steps[0].observation)

    def test_successful_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_success_events())

            traj = load_trajectory(str(path))

            self.assertEqual(traj.final_status, "success")


class DeerFlowClassifierTests(unittest.TestCase):
    def test_middleware_step_classified_as_middleware_action(self):
        from classifier import classify_action_type
        self.assertEqual(
            classify_action_type("middleware:summarize", None, "middleware"),
            "middleware_action",
        )

    def test_tool_call_step_classified(self):
        from classifier import classify_action_type
        self.assertEqual(
            classify_action_type("pytest", None, "tool_call"),
            "run_test",
        )

    def test_tool_result_step_classified(self):
        from classifier import classify_action_type
        self.assertEqual(
            classify_action_type("bash", None, "tool_result"),
            "tool_result",
        )

    def test_middleware_stage(self):
        from classifier import classify_stage
        self.assertEqual(
            classify_stage("middleware:summarize", "middleware_action", None, "middleware"),
            "middleware",
        )

    def test_summarization_is_state_changing(self):
        from classifier import is_state_changing
        self.assertTrue(
            is_state_changing("middleware_action", "middleware:summarize", None, "middleware"),
        )

    def test_non_summarization_middleware_is_not_state_changing(self):
        from classifier import is_state_changing
        self.assertFalse(
            is_state_changing("middleware_action", "middleware:title", None, "middleware"),
        )


class DeerFlowAnalysisTests(unittest.TestCase):
    def test_excessive_summarization_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_multiple_summarization_events())

            result = evaluate_file(path)

            mw_steps = [s for s in result.normalized_steps if s.item_type == "middleware"]
            flagged = [s for s in mw_steps if "excessive_summarization" in s.matched_pattern_names]
            self.assertTrue(len(flagged) >= 1)

    def test_loop_detection_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_loop_detection_events())

            result = evaluate_file(path)

            flagged = [s for s in result.normalized_steps if "loop_detection_triggered" in s.matched_pattern_names]
            self.assertTrue(len(flagged) >= 1)

    def test_subagent_failure_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_subagent_failure_events())

            result = evaluate_file(path)

            flagged = [s for s in result.normalized_steps if "subagent_failure" in s.matched_pattern_names]
            self.assertTrue(len(flagged) >= 1)

    def test_tool_error_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_tool_error_events())

            result = evaluate_file(path)

            flagged = [s for s in result.normalized_steps if "tool_error" in s.matched_pattern_names]
            self.assertTrue(len(flagged) >= 1)

    def test_evaluate_deerflow_trajectory_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            write_jsonl(path, _deerflow_basic_events())

            result = evaluate_file(path)

            self.assertEqual(result.final_status, "failed")
            self.assertTrue(len(result.normalized_steps) > 0)
            self.assertIsNotNone(result.diagnosis)
            self.assertIsNotNone(result.metrics)


def _make_event(event_type: str, category: str, content=None, metadata=None, seq: int = 0,
                thread_id: str = "test-thread", run_id: str = "test-run") -> dict:
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "event_type": event_type,
        "category": category,
        "content": content or "",
        "metadata": metadata or {},
        "seq": seq,
        "created_at": "2026-05-12T10:00:00+00:00",
    }


def _deerflow_basic_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {"chain": "lead_agent"}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Fix the parser bug"}, {"caller": "lead_agent"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "I will inspect the parser",
            "tool_calls": [{"name": "bash", "args": {"command": "cat parser.py"}, "id": "tc1"}],
        }, {"caller": "lead_agent", "usage": {"input_tokens": 100, "output_tokens": 50}}, seq=3),
        _make_event("llm.tool.result", "message", {
            "content": "def parse(): pass",
            "name": "bash",
            "tool_call_id": "tc1",
        }, {"caller": "lead_agent"}, seq=4),
        _make_event("llm.ai.response", "message", {
            "content": "Now I'll patch it",
            "tool_calls": [{"name": "str_replace_editor", "args": {"path": "parser.py", "old": "pass", "new": "return True"}, "id": "tc2"}],
        }, {"caller": "lead_agent"}, seq=5),
        _make_event("llm.tool.result", "message", {
            "content": "File updated successfully",
            "name": "str_replace_editor",
            "tool_call_id": "tc2",
        }, {"caller": "lead_agent"}, seq=6),
        _make_event("llm.ai.response", "message", {
            "content": "Let me run the tests",
            "tool_calls": [{"name": "bash", "args": {"command": "pytest"}, "id": "tc3"}],
        }, {"caller": "lead_agent"}, seq=7),
        _make_event("llm.tool.result", "message", {
            "content": "1 failed, 5 passed",
            "name": "bash",
            "tool_call_id": "tc3",
        }, {"caller": "lead_agent"}, seq=8),
        _make_event("run.end", "outputs", {"status": "completed"}, {"status": "success"}, seq=9),
        _make_event("run.error", "error", {"message": "Agent execution failed"}, {"error_type": "RuntimeError"}, seq=10),
    ]


def _deerflow_tool_call_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Run commands"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "",
            "tool_calls": [{"name": "bash", "args": {"command": "ls -la"}, "id": "tc1"}],
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("llm.tool.result", "message", {
            "content": "file1.py\nfile2.py",
            "name": "bash",
            "tool_call_id": "tc1",
        }, {"caller": "lead_agent"}, seq=4),
        _make_event("run.end", "outputs", {}, seq=5),
    ]


def _deerflow_middleware_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Long task"}, seq=2),
        _make_event("middleware:summarize", "middleware", {
            "name": "SummarizationMiddleware",
            "hook": "after_model",
            "action": "summarize_context",
            "changes": {"removed_messages": 10, "summary_length": 200},
        }, {"caller": "middleware:summarize"}, seq=3),
        _make_event("middleware:loop_detect", "middleware", {
            "name": "LoopDetectionMiddleware",
            "hook": "after_model",
            "action": "warn_loop",
            "changes": {"tool_hash": "abc123", "count": 3},
        }, {"caller": "middleware:loop_detect"}, seq=4),
        _make_event("run.end", "outputs", {}, seq=5),
    ]


def _deerflow_failure_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Do something"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "Done",
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("run.error", "error", {"message": "Agent execution failed"}, seq=4),
    ]


def _deerflow_llm_error_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Generate code"}, seq=2),
        _make_event("llm.error", "trace", "Rate limit exceeded", seq=3),
        _make_event("run.end", "outputs", {}, seq=4),
    ]


def _deerflow_agent_message_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Explain the code"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "I will fix the parser by modifying the AST traversal.",
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("run.end", "outputs", {}, seq=4),
    ]


def _deerflow_success_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "List files"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "",
            "tool_calls": [{"name": "bash", "args": {"command": "ls"}, "id": "tc1"}],
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("llm.tool.result", "message", {
            "content": "file1.py",
            "name": "bash",
            "tool_call_id": "tc1",
        }, {"caller": "lead_agent"}, seq=4),
        _make_event("run.end", "outputs", {}, {"status": "success"}, seq=5),
    ]


def _deerflow_multiple_summarization_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Long task"}, seq=2),
        _make_event("middleware:summarize", "middleware", {
            "name": "SummarizationMiddleware",
            "hook": "after_model",
            "action": "summarize_context",
            "changes": {"removed_messages": 10},
        }, {"caller": "middleware:summarize"}, seq=3),
        _make_event("llm.ai.response", "message", {"content": "Continuing..."}, seq=4),
        _make_event("middleware:summarize", "middleware", {
            "name": "SummarizationMiddleware",
            "hook": "after_model",
            "action": "summarize_context",
            "changes": {"removed_messages": 8},
        }, {"caller": "middleware:summarize"}, seq=5),
        _make_event("run.end", "outputs", {}, seq=6),
    ]


def _deerflow_loop_detection_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Fix bug"}, seq=2),
        _make_event("middleware:loop_detect", "middleware", {
            "name": "LoopDetectionMiddleware",
            "hook": "after_model",
            "action": "warn_loop",
            "changes": {"tool_hash": "abc", "count": 4},
        }, {"caller": "middleware:loop_detect"}, seq=3),
        _make_event("run.end", "outputs", {}, seq=4),
    ]


def _deerflow_subagent_failure_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Run sub-agent"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "",
            "tool_calls": [{"name": "task", "args": {"description": "Do stuff"}, "id": "tc1"}],
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("llm.tool.result", "message", {
            "content": "Sub-agent execution failed: timed out after 120s",
            "name": "task",
            "tool_call_id": "tc1",
        }, {"caller": "lead_agent"}, seq=4),
        _make_event("run.end", "outputs", {}, seq=5),
    ]


def _deerflow_tool_error_events() -> list[dict]:
    return [
        _make_event("run.start", "trace", {}, seq=1),
        _make_event("llm.human.input", "message", {"content": "Run tool"}, seq=2),
        _make_event("llm.ai.response", "message", {
            "content": "",
            "tool_calls": [{"name": "bash", "args": {"command": "pytest"}, "id": "tc1"}],
        }, {"caller": "lead_agent"}, seq=3),
        _make_event("llm.tool.result", "message", {
            "content": "Permission denied",
            "name": "bash",
            "tool_call_id": "tc1",
            "status": "error",
        }, {"caller": "lead_agent"}, seq=4),
        _make_event("run.end", "outputs", {}, seq=5),
    ]


if __name__ == "__main__":
    unittest.main()
