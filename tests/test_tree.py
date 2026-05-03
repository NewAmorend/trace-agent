"""Unit tests for tree.py."""

import unittest

from tree import build_trace_tree, render_trace_tree
from tests._helpers import make_normalized_step


class BuildTraceTreeTests(unittest.TestCase):
    def test_empty_steps_returns_empty_list(self):
        self.assertEqual(build_trace_tree([]), [])

    def test_single_explore_step_lives_under_state_zero(self):
        steps = [make_normalized_step(step_id=1, action="ls", state_change=False)]
        nodes = build_trace_tree(steps)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].state_id, 0)
        self.assertEqual(len(nodes[0].steps), 1)

    def test_state_changing_step_creates_new_state(self):
        steps = [
            make_normalized_step(step_id=1, action="ls", state_change=False),
            make_normalized_step(step_id=2, action="apply_patch", state_change=True),
            make_normalized_step(step_id=3, action="pytest", state_change=False),
        ]
        nodes = build_trace_tree(steps)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].state_id, 0)
        self.assertEqual(nodes[1].state_id, 1)
        self.assertEqual(len(nodes[0].steps), 2)
        self.assertEqual(len(nodes[1].steps), 1)
        self.assertEqual(nodes[0].children[0].state_id, 1)


class RenderTraceTreeTests(unittest.TestCase):
    def test_empty_renders_placeholder(self):
        self.assertIn("No steps found", render_trace_tree([]))

    def test_single_state_renders_state_zero_label(self):
        steps = [make_normalized_step(step_id=1, action="ls", action_type="other")]
        out = render_trace_tree(build_trace_tree(steps))
        self.assertIn("State 0", out)
        self.assertIn("Step 1", out)

    def test_state_transition_renders_arrow(self):
        steps = [
            make_normalized_step(step_id=1, action="apply_patch foo", state_change=True),
        ]
        out = render_trace_tree(build_trace_tree(steps))
        self.assertIn("-> State 1", out)

    def test_agent_message_uses_observation_as_display(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="",
                observation="I'll start by reading the file",
                item_type="agent_message",
            )
        ]
        out = render_trace_tree(build_trace_tree(steps))
        self.assertIn("reading the file", out)


if __name__ == "__main__":
    unittest.main()
