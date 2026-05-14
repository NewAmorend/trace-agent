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


class MetricMathTests(unittest.TestCase):
    def test_f1_zero_when_both_zero(self):
        from judge_eval import _f1
        self.assertEqual(_f1(0.0, 0.0), 0.0)

    def test_f1_harmonic_mean(self):
        from judge_eval import _f1
        self.assertAlmostEqual(_f1(0.5, 1.0), 2 / 3, places=6)

    def test_category_for_prediction_uses_matched_pattern_names(self):
        from judge_eval import _category_for_prediction
        from tests._helpers import make_normalized_step

        step = make_normalized_step(step_id=1, action="x")
        step.matched_pattern_names = ["test_edit_after_impl_failure", "repeated_command"]
        self.assertEqual(_category_for_prediction(step), "test_edit_after_impl_failure")

    def test_category_for_prediction_none_when_no_matches(self):
        from judge_eval import _category_for_prediction
        from tests._helpers import make_normalized_step

        step = make_normalized_step(step_id=1, action="x")
        self.assertIsNone(_category_for_prediction(step))


class EvaluateJudgesTests(unittest.TestCase):
    def _label_dir_with(self, payload: dict) -> tempfile.TemporaryDirectory:
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "fixture.labels.json"
        path.write_text(json.dumps(payload))
        return tmp

    def test_rule_judge_perfect_labels_yields_high_f1(self):
        """Label *every* rule-flagged step as suspicious; precision and recall should both be 1.0."""
        from analyzer import score_suspicious_steps
        from classifier import normalize_steps
        from judge_eval import evaluate_judges
        from parser import load_trajectory

        trajectory = load_trajectory("examples/codex_failed_run_001.jsonl")
        normalized = normalize_steps(trajectory.steps)
        scored = score_suspicious_steps(normalized, trajectory.task, trajectory.final_status)
        flagged_ids = {s.step_id for s in scored if s.suspicious_score > 0}

        payload = {
            "trajectory": "examples/codex_failed_run_001.jsonl",
            "labeler": "test",
            "labeled_at": "2026-05-14",
            "final_status": trajectory.final_status,
            "critical_step_id": max(scored, key=lambda s: s.suspicious_score).step_id,
            "steps": [{"step_id": sid, "suspicious": True} for sid in sorted(flagged_ids)],
        }
        with self._label_dir_with(payload) as tmp:
            metrics = evaluate_judges(Path(tmp))
            self.assertAlmostEqual(metrics.suspicious_precision, 1.0)
            self.assertAlmostEqual(metrics.suspicious_recall, 1.0)
            self.assertAlmostEqual(metrics.suspicious_f1, 1.0)
            self.assertEqual(metrics.labeled_trajectories, 1)
            self.assertEqual(metrics.skipped_trajectories, 0)

    def test_label_with_no_overlap_yields_zero_f1(self):
        """Label step 1 only (which the rule judge does not flag in this fixture)."""
        from judge_eval import evaluate_judges
        payload = {
            "trajectory": "examples/codex_failed_run_001.jsonl",
            "labeler": "test",
            "labeled_at": "2026-05-14",
            "final_status": "failed",
            "steps": [{"step_id": 1, "suspicious": True}],
        }
        with self._label_dir_with(payload) as tmp:
            metrics = evaluate_judges(Path(tmp))
            self.assertEqual(metrics.suspicious_precision, 0.0)
            self.assertEqual(metrics.suspicious_recall, 0.0)
            self.assertEqual(metrics.suspicious_f1, 0.0)

    def test_missing_trajectory_file_is_skipped(self):
        from judge_eval import evaluate_judges
        payload = {
            "trajectory": "examples/does_not_exist.jsonl",
            "labeler": "test",
            "labeled_at": "2026-05-14",
            "final_status": "failed",
            "steps": [],
        }
        with self._label_dir_with(payload) as tmp:
            metrics = evaluate_judges(Path(tmp))
            self.assertEqual(metrics.labeled_trajectories, 0)
            self.assertEqual(metrics.skipped_trajectories, 1)

    def test_scorer_judge_is_invoked(self):
        from judge_eval import evaluate_judges
        called = {"flag": False}

        def fake_scorer(steps, task, final_status):
            called["flag"] = True
            return steps

        payload = {
            "trajectory": "examples/codex_failed_run_001.jsonl",
            "labeler": "test",
            "labeled_at": "2026-05-14",
            "final_status": "failed",
            "steps": [],
        }
        with self._label_dir_with(payload) as tmp:
            evaluate_judges(Path(tmp), scorer_judge=fake_scorer)
        self.assertTrue(called["flag"])


class FormatMetricsTests(unittest.TestCase):
    def _metrics(self, **overrides):
        from judge_eval import JudgeMetrics
        defaults = dict(
            suspicious_precision=0.8,
            suspicious_recall=0.6,
            suspicious_f1=0.685,
            critical_hit_at_1=0.5,
            category_accuracy=0.75,
            labeled_trajectories=4,
            skipped_trajectories=1,
            per_trajectory=[],
        )
        defaults.update(overrides)
        return JudgeMetrics(**defaults)

    def test_format_md_includes_all_metrics(self):
        from judge_eval import format_metrics_md
        out = format_metrics_md(self._metrics())
        self.assertIn("Precision: 0.800", out)
        self.assertIn("Recall:    0.600", out)
        self.assertIn("F1:        0.685", out)
        self.assertIn("hit@1: 0.500", out)
        self.assertIn("0.750", out)
        self.assertIn("Labeled trajectories: 4", out)
        self.assertIn("Skipped (no trajectory file): 1", out)

    def test_format_md_handles_none_category_accuracy(self):
        from judge_eval import format_metrics_md
        out = format_metrics_md(self._metrics(category_accuracy=None))
        self.assertIn("n/a", out)

    def test_metrics_to_dict_round_trips_to_json(self):
        from judge_eval import metrics_to_dict
        d = metrics_to_dict(self._metrics())
        s = json.dumps(d)
        self.assertEqual(json.loads(s)["suspicious_f1"], 0.685)


if __name__ == "__main__":
    unittest.main()
