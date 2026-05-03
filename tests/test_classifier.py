"""Unit tests for classifier.py."""

import unittest

from classifier import (
    classify_action_type,
    classify_stage,
    is_state_changing,
    normalize_steps,
)
from tests._helpers import make_step


class ClassifyActionTypeTests(unittest.TestCase):
    def test_pytest_is_run_test(self):
        self.assertEqual(classify_action_type("pytest tests/", None), "run_test")

    def test_npm_test_is_run_test(self):
        self.assertEqual(classify_action_type("npm test", None), "run_test")

    def test_grep_is_search(self):
        self.assertEqual(classify_action_type("grep -r foo .", None), "search")

    def test_cat_is_inspect_file(self):
        self.assertEqual(classify_action_type("cat README.md", None), "inspect_file")

    def test_pip_install_is_env_change(self):
        self.assertEqual(classify_action_type("pip install requests", None), "env_change")

    def test_git_status_is_git_action(self):
        self.assertEqual(classify_action_type("git status", None), "git_action")

    def test_diff_present_is_edit_file(self):
        self.assertEqual(classify_action_type("Write src/x.py", "src/x.py"), "edit_file")

    def test_apply_patch_is_edit_file(self):
        self.assertEqual(classify_action_type("apply_patch foo.py", None), "edit_file")

    def test_unknown_command_is_other(self):
        self.assertEqual(classify_action_type("xyzzy --do-thing", None), "other")


class ClassifyStageTests(unittest.TestCase):
    def test_python_version_is_environment_verification(self):
        self.assertEqual(
            classify_stage("python --version", "other", None),
            "environment verification",
        )

    def test_env_change_action_type_maps_to_dependency_installation(self):
        self.assertEqual(
            classify_stage("pip install x", "env_change", None),
            "dependency installation",
        )

    def test_edit_file_action_type_maps_to_patching(self):
        self.assertEqual(
            classify_stage("apply_patch x", "edit_file", None),
            "patching",
        )

    def test_run_test_action_type_maps_to_verification(self):
        self.assertEqual(
            classify_stage("pytest", "run_test", None),
            "verification",
        )

    def test_search_maps_to_inspection_debugging(self):
        self.assertEqual(
            classify_stage("grep x", "search", None),
            "inspection/debugging",
        )

    def test_error_observation_maps_to_inspection_debugging(self):
        self.assertEqual(
            classify_stage("ls", "other", "Traceback ... Exception"),
            "inspection/debugging",
        )


class IsStateChangingTests(unittest.TestCase):
    def test_edit_file_is_state_changing(self):
        self.assertTrue(is_state_changing("edit_file", "Write x", None))

    def test_env_change_is_state_changing(self):
        self.assertTrue(is_state_changing("env_change", "pip install x", None))

    def test_git_checkout_is_state_changing(self):
        self.assertTrue(is_state_changing("git_action", "git checkout main", None))

    def test_git_status_is_not_state_changing(self):
        self.assertFalse(is_state_changing("git_action", "git status", None))

    def test_search_is_not_state_changing(self):
        self.assertFalse(is_state_changing("search", "grep x", None))


class NormalizeStepsTests(unittest.TestCase):
    def test_normalize_preserves_step_count(self):
        steps = [make_step(step_id=1, action="pytest"), make_step(step_id=2, action="cat x")]
        out = normalize_steps(steps)
        self.assertEqual(len(out), 2)

    def test_normalize_assigns_action_types(self):
        steps = [make_step(step_id=1, action="pytest")]
        out = normalize_steps(steps)
        self.assertEqual(out[0].action_type, "run_test")

    def test_normalize_default_score_is_zero(self):
        steps = [make_step(step_id=1, action="pytest")]
        out = normalize_steps(steps)
        self.assertEqual(out[0].suspicious_score, 0.0)
        self.assertEqual(out[0].suspicious_reasons, [])


if __name__ == "__main__":
    unittest.main()
