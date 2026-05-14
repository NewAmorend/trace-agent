"""Unit tests for judge_eval.py."""

import json
import tempfile
import unittest
from pathlib import Path


class LoadLabelTests(unittest.TestCase):
    def _write(self, dir_: Path, name: str, payload: dict) -> Path:
        path = dir_ / name
        path.write_text(json.dumps(payload))
        return path

    def test_load_minimal_label(self):
        from judge_eval import load_label
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = self._write(tmp, "x.labels.json", {
                "trajectory": "examples/codex_failed_run_001.jsonl",
                "labeler": "human",
                "labeled_at": "2026-05-14",
                "final_status": "failed",
                "critical_step_id": 5,
                "steps": [
                    {"step_id": 1, "suspicious": False},
                    {"step_id": 5, "suspicious": True, "category": "test_edit_after_impl_failure"},
                ],
            })
            label = load_label(path)
            self.assertEqual(label.final_status, "failed")
            self.assertEqual(label.critical_step_id, 5)
            self.assertEqual(label.step_labels[5].category, "test_edit_after_impl_failure")
            self.assertFalse(label.step_labels[1].suspicious)
            self.assertTrue(label.step_labels[5].suspicious)

    def test_unknown_category_raises(self):
        from judge_eval import load_label
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = self._write(tmp, "x.labels.json", {
                "trajectory": "examples/codex_failed_run_001.jsonl",
                "labeler": "human",
                "labeled_at": "2026-05-14",
                "final_status": "failed",
                "steps": [
                    {"step_id": 1, "suspicious": True, "category": "no_such_pattern"},
                ],
            })
            with self.assertRaisesRegex(ValueError, "no_such_pattern"):
                load_label(path)

    def test_missing_steps_default_not_suspicious(self):
        from judge_eval import load_label
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = self._write(tmp, "x.labels.json", {
                "trajectory": "examples/codex_failed_run_001.jsonl",
                "labeler": "human",
                "labeled_at": "2026-05-14",
                "final_status": "success",
                "steps": [],
            })
            label = load_label(path)
            self.assertFalse(label.is_suspicious(1))
            self.assertFalse(label.is_suspicious(99))
            self.assertIsNone(label.category_for(1))

    def test_discover_labels_finds_only_label_files(self):
        from judge_eval import discover_labels
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write(tmp, "a.labels.json", {
                "trajectory": "examples/codex_failed_run_001.jsonl",
                "labeler": "human", "labeled_at": "2026-05-14",
                "final_status": "failed", "steps": [],
            })
            (tmp / "README.md").write_text("# not a label")
            labels = discover_labels(tmp)
            self.assertEqual(len(labels), 1)


if __name__ == "__main__":
    unittest.main()
