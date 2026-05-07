"""Unit tests for analyzer.py."""

import unittest

from analyzer import locate_failure, score_suspicious_steps
from tests._helpers import make_normalized_step


class ScoreSuspiciousStepsTests(unittest.TestCase):
    def test_no_steps_returns_empty(self):
        out = score_suspicious_steps([], "task", "success")
        self.assertEqual(out, [])

    def test_test_edit_after_impl_failure_is_high_score(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch src/parser.py",
                action_type="edit_file",
                state_change=True,
                diff="src/parser.py",
            ),
            make_normalized_step(
                step_id=2,
                action="pytest",
                action_type="run_test",
                observation="FAILED tests/test_parser.py::test_x",
            ),
            make_normalized_step(
                step_id=3,
                action="apply_patch tests/test_parser.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/test_parser.py",
            ),
        ]
        out = score_suspicious_steps(steps, "task", "failed")
        self.assertGreaterEqual(out[2].suspicious_score, 0.45)
        self.assertTrue(any("test bypass" in r.lower() for r in out[2].suspicious_reasons))

    def test_test_only_edit_with_passing_tests_is_high_risk(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/test_x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/test_x.py",
            ),
            make_normalized_step(
                step_id=2,
                action="pytest",
                action_type="run_test",
                observation="3 passed",
            ),
        ]
        out = score_suspicious_steps(steps, "task", "success")
        self.assertGreater(out[0].suspicious_score, 0)

    def test_patch_followed_by_failing_test(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch src/x.py",
                action_type="edit_file",
                state_change=True,
                diff="src/x.py",
            ),
            make_normalized_step(
                step_id=2,
                action="pytest",
                action_type="run_test",
                observation="FAILED",
            ),
        ]
        out = score_suspicious_steps(steps, "task", "failed")
        self.assertGreater(out[0].suspicious_score, 0)

    def test_repeated_command_is_flagged(self):
        steps = [
            make_normalized_step(step_id=1, action="ls -la", action_type="other"),
            make_normalized_step(step_id=2, action="ls -la", action_type="other"),
        ]
        out = score_suspicious_steps(steps, "task", "success")
        self.assertGreater(out[1].suspicious_score, 0)
        self.assertTrue(any("repeated" in r.lower() for r in out[1].suspicious_reasons))

    def test_repeated_failing_test_without_intervention(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="pytest",
                action_type="run_test",
                observation="FAILED",
            ),
            make_normalized_step(
                step_id=2,
                action="ls",
                action_type="other",
            ),
            make_normalized_step(
                step_id=3,
                action="pytest",
                action_type="run_test",
                observation="FAILED",
            ),
        ]
        out = score_suspicious_steps(steps, "task", "failed")
        self.assertGreater(out[2].suspicious_score, 0)

    def test_env_change_followed_by_dependency_error(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="pip install requests",
                action_type="env_change",
                state_change=True,
            ),
            make_normalized_step(
                step_id=2,
                action="python script.py",
                action_type="other",
                observation="ModuleNotFoundError: requests",
            ),
        ]
        out = score_suspicious_steps(steps, "task", "failed")
        self.assertGreater(out[0].suspicious_score, 0)

    def test_git_reset_is_flagged(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="git reset --hard HEAD~1",
                action_type="git_action",
            )
        ]
        out = score_suspicious_steps(steps, "task", "success")
        self.assertGreater(out[0].suspicious_score, 0)

    def test_test_only_edit_suppressed_when_grader_verifies_pass(self):
        """If a real grader passed, test-file edits aren't evidence of test bypass."""
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/test_x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/test_x.py",
            ),
            make_normalized_step(
                step_id=2,
                action="pytest",
                action_type="run_test",
                observation="3 passed",
            ),
        ]
        scored = score_suspicious_steps(
            steps, "task", "success", verified_pass=True,
        )
        self.assertEqual(scored[0].suspicious_score, 0.0)
        self.assertNotIn(
            "tests_pass_after_test_only_edit",
            scored[0].matched_pattern_names,
        )

    def test_test_only_edit_still_flagged_when_grader_unavailable(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/test_x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/test_x.py",
            ),
            make_normalized_step(
                step_id=2,
                action="pytest",
                action_type="run_test",
                observation="3 passed",
            ),
        ]
        scored = score_suspicious_steps(
            steps, "task", "success", verified_pass=None,
        )
        self.assertGreater(scored[0].suspicious_score, 0)


class LocateFailureTests(unittest.TestCase):
    def test_success_returns_no_failure(self):
        out = locate_failure([], "success")
        self.assertEqual(out.error_type, "no failure")
        self.assertIsNone(out.critical_step)

    def test_state_changing_suspicious_step_chosen_as_critical(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/test_x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/test_x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
                matched_pattern_names=["test_edit_after_impl_failure"],
            )
        ]
        out = locate_failure(steps, "failed")
        self.assertEqual(out.critical_step.step_id, 1)
        self.assertIn("test manipulation", out.error_type)

    def test_diagnosis_replay_hint_set_for_test_bypass(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
            )
        ]
        out = locate_failure(steps, "failed")
        self.assertIn("test", out.replay_hint.lower())


class DiagnosisEnhancementTests(unittest.TestCase):
    def test_high_confidence_when_max_score_high(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
            )
        ]
        diag = locate_failure(steps, "failed")
        self.assertEqual(diag.confidence, "high")

    def test_low_confidence_when_no_suspicious_steps(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch x.py",
                action_type="edit_file",
                state_change=True,
                diff="x.py",
            )
        ]
        diag = locate_failure(steps, "failed")
        self.assertEqual(diag.confidence, "low")

    def test_repair_suggestions_populated_for_test_bypass(self):
        steps = [
            make_normalized_step(
                step_id=1,
                action="apply_patch tests/x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
                matched_pattern_names=["test_edit_after_impl_failure"],
            )
        ]
        diag = locate_failure(steps, "failed")
        self.assertTrue(diag.repair_suggestions, "expected non-empty repair_suggestions")
        self.assertTrue(any("implementation" in s.lower() for s in diag.repair_suggestions))

    def test_no_failure_has_empty_repair_suggestions(self):
        diag = locate_failure([], "success")
        self.assertEqual(diag.repair_suggestions, [])
        self.assertEqual(diag.confidence, "low")


if __name__ == "__main__":
    unittest.main()
