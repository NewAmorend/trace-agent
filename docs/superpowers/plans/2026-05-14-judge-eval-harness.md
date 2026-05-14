# Judge-Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a harness that scores any judge (classifier or scorer) against a hand-labeled trajectory corpus and reports precision / recall / F1 on suspicious-step detection plus critical-step `hit@1` and category accuracy.

**Architecture:** New `judge_eval.py` module with a single public entry point `evaluate_judges(labeled_corpus_dir, *, classifier_judge=None, scorer_judge=None) -> JudgeMetrics`. The harness composes the existing pipeline (`parser.load_trajectory` → `classifier.normalize_steps` → `analyzer.score_suspicious_steps`), substituting either stage when a judge is passed. `analyzer.score_suspicious_steps` gets a new optional `judge=` parameter that parallels the one already on `normalize_steps`. A `judge-eval` CLI subcommand in `main.py` wraps the API. Labels are sidecar JSON files under `tests/fixtures/labels/`.

**Tech Stack:** Python 3.10+ stdlib only; `unittest` (project convention — see `CLAUDE.md`).

---

## File Map

| File | Change |
|---|---|
| `analyzer.py` | **Modify** — add `judge` param to `score_suspicious_steps` |
| `judge_eval.py` | **Create** — `StepLabel`, `LabeledTrajectory`, `JudgeMetrics`, `load_label`, `discover_labels`, `evaluate_judges`, `format_metrics_md`, `metrics_to_dict` |
| `main.py` | **Modify** — register `judge-eval` subparser; add `run_judge_eval_command` |
| `tests/fixtures/labels/codex_failed_run_001.labels.json` | **Create** — labels for the canonical failing fixture |
| `tests/fixtures/labels/<other>.labels.json` | **Create** — one per remaining seed trajectory |
| `tests/test_judge_eval.py` | **Create** — unit tests for hook, loader, metric math, end-to-end harness, and CLI wiring |

---

## Task 1: Add scorer-judge hook to `score_suspicious_steps`

**Files:**
- Modify: `analyzer.py:29` (signature) and `analyzer.py:30-34` (docstring + early return)
- Modify: `tests/test_analyzer.py` (new test class)

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_analyzer.py` immediately before the `if __name__ == "__main__":` block:

```python
class ScoreSuspiciousJudgeHookTests(unittest.TestCase):
    def test_judge_replaces_rule_scoring(self):
        def judge(steps, task, final_status):
            out = []
            for step in steps:
                step.suspicious_score = 9.99
                step.suspicious_reasons = ["custom"]
                out.append(step)
            return out

        steps = [make_normalized_step(step_id=1, action="apply_patch tests/x.py",
                                      action_type="edit_file", state_change=True,
                                      diff="tests/x.py")]
        out = score_suspicious_steps(steps, "task", "failed", judge=judge)
        self.assertEqual(out[0].suspicious_score, 9.99)
        self.assertEqual(out[0].suspicious_reasons, ["custom"])

    def test_default_behavior_unchanged_without_judge(self):
        steps = [make_normalized_step(step_id=1, action="apply_patch tests/x.py",
                                      action_type="edit_file", state_change=True,
                                      diff="tests/x.py")]
        out = score_suspicious_steps(steps, "task", "success")
        self.assertGreater(out[0].suspicious_score, 0)
        self.assertTrue(out[0].suspicious_reasons)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m unittest tests.test_analyzer.ScoreSuspiciousJudgeHookTests -v 2>&1 | tail -15
```

Expected: `test_judge_replaces_rule_scoring` fails with `TypeError: score_suspicious_steps() got an unexpected keyword argument 'judge'`. The default test should pass.

- [ ] **Step 3: Add `judge` parameter to `score_suspicious_steps`**

Edit `analyzer.py`. Replace the function signature and the first body line (currently `analyzer.py:29-35`) with:

```python
def score_suspicious_steps(
    steps: list[NormalizedStep],
    task: str,
    final_status: str,
    judge: "ScorerJudge | None" = None,
) -> list[NormalizedStep]:
    """
    Score steps for suspicious behavior.

    Returns steps with suspicious_score and suspicious_reasons populated.
    When `judge` is provided, it replaces the rule-based scoring entirely;
    the caller is responsible for populating suspicious_score and
    suspicious_reasons (and optionally matched_pattern_names).
    """
    if judge is not None:
        return judge(steps, task, final_status)

    steps = copy.deepcopy(steps)
```

Then add the `ScorerJudge` type alias near the top of `analyzer.py`. Right after the existing imports block (after `from test_signals import ...`), add:

```python
from typing import Callable

ScorerJudge = Callable[[list[NormalizedStep], str, str], list[NormalizedStep]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_analyzer -v 2>&1 | tail -10
python -m unittest discover 2>&1 | tail -5
```

Expected: all analyzer tests pass; full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: add optional judge hook to score_suspicious_steps"
```

---

## Task 2: Label loader

**Files:**
- Create: `judge_eval.py`
- Create: `tests/test_judge_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge_eval.py`:

```python
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
            with self.assertRaises(ValueError):
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m unittest tests.test_judge_eval -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'judge_eval'`.

- [ ] **Step 3: Create `judge_eval.py` with the loader**

Create `judge_eval.py`:

```python
"""Harness for evaluating any judge against a hand-labeled trajectory corpus.

A "judge" is whichever pipeline component produces NormalizedSteps with
suspicious_score / suspicious_reasons populated. This module composes the
existing parser -> normalize_steps -> score_suspicious_steps pipeline,
optionally substituting either stage with a user-provided callable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from models import NormalizedStep, Step
from patterns import PATTERNS

ClassifierJudge = Callable[[Step], NormalizedStep]
ScorerJudge = Callable[[list[NormalizedStep], str, str], list[NormalizedStep]]


@dataclass
class StepLabel:
    step_id: int
    suspicious: bool
    category: str | None = None


@dataclass
class LabeledTrajectory:
    trajectory_path: Path
    labeler: str
    labeled_at: str
    final_status: str
    critical_step_id: int | None
    step_labels: dict[int, StepLabel]

    def is_suspicious(self, step_id: int) -> bool:
        label = self.step_labels.get(step_id)
        return label.suspicious if label else False

    def category_for(self, step_id: int) -> str | None:
        label = self.step_labels.get(step_id)
        return label.category if label else None


def load_label(path: Path) -> LabeledTrajectory:
    """Load a single .labels.json sidecar file.

    Raises ValueError if any step's category is not a key in PATTERNS.
    """
    data = json.loads(Path(path).read_text())

    valid_categories = set(PATTERNS.keys())
    step_labels: dict[int, StepLabel] = {}
    for entry in data.get("steps", []):
        step_id = int(entry["step_id"])
        suspicious = bool(entry.get("suspicious", False))
        category = entry.get("category")
        if category is not None and category not in valid_categories:
            raise ValueError(
                f"{path}: step {step_id} category {category!r} is not in PATTERNS registry"
            )
        step_labels[step_id] = StepLabel(
            step_id=step_id, suspicious=suspicious, category=category,
        )

    return LabeledTrajectory(
        trajectory_path=Path(data["trajectory"]),
        labeler=data.get("labeler", "unknown"),
        labeled_at=data.get("labeled_at", ""),
        final_status=data["final_status"],
        critical_step_id=data.get("critical_step_id"),
        step_labels=step_labels,
    )


def discover_labels(corpus_dir: Path) -> list[LabeledTrajectory]:
    """Load all *.labels.json files under corpus_dir, sorted by filename."""
    corpus_dir = Path(corpus_dir)
    return [load_label(p) for p in sorted(corpus_dir.glob("*.labels.json"))]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_judge_eval -v 2>&1 | tail -15
```

Expected: all four `LoadLabelTests` pass.

- [ ] **Step 5: Commit**

```bash
git add judge_eval.py tests/test_judge_eval.py
git commit -m "feat: add label loader for judge-eval harness"
```

---

## Task 3: Metric math

**Files:**
- Modify: `judge_eval.py` (add `JudgeMetrics`, `_f1`, `_category_for_prediction`)
- Modify: `tests/test_judge_eval.py` (add metric tests using synthetic predictions)

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_judge_eval.py` before the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m unittest tests.test_judge_eval.MetricMathTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name '_f1' from 'judge_eval'`.

- [ ] **Step 3: Add metric helpers and `JudgeMetrics` to `judge_eval.py`**

Append to `judge_eval.py`:

```python
from analyzer import _patterns_matched


@dataclass
class JudgeMetrics:
    suspicious_precision: float
    suspicious_recall: float
    suspicious_f1: float
    critical_hit_at_1: float
    category_accuracy: float | None
    labeled_trajectories: int
    skipped_trajectories: int
    per_trajectory: list[dict] = field(default_factory=list)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _category_for_prediction(step: NormalizedStep) -> str | None:
    """Return the top-matched pattern name for a step, or None.

    Reuses analyzer._patterns_matched, which reads step.matched_pattern_names.
    Judges that don't populate matched_pattern_names will yield None here,
    which correctly excludes that step from category accuracy.
    """
    matched = _patterns_matched(step)
    return matched[0].name if matched else None
```

Note: the `from analyzer import _patterns_matched` must go at the top of the file with the other imports, not at the bottom. Move it into the existing import block.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_judge_eval.MetricMathTests -v 2>&1 | tail -10
```

Expected: all four `MetricMathTests` pass.

- [ ] **Step 5: Commit**

```bash
git add judge_eval.py tests/test_judge_eval.py
git commit -m "feat: add metric helpers and JudgeMetrics dataclass"
```

---

## Task 4: End-to-end `evaluate_judges`

**Files:**
- Modify: `judge_eval.py` (add `evaluate_judges`)
- Modify: `tests/test_judge_eval.py` (add end-to-end tests with the existing `examples/codex_failed_run_001.jsonl` fixture)

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_judge_eval.py` before the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m unittest tests.test_judge_eval.EvaluateJudgesTests -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'evaluate_judges' from 'judge_eval'`.

- [ ] **Step 3: Add `evaluate_judges` and helpers to `judge_eval.py`**

Append to `judge_eval.py`:

```python
from analyzer import score_suspicious_steps
from classifier import normalize_steps
from parser import load_trajectory


def evaluate_judges(
    labeled_corpus_dir: Path,
    *,
    classifier_judge: ClassifierJudge | None = None,
    scorer_judge: ScorerJudge | None = None,
) -> JudgeMetrics:
    """Evaluate the (classifier + scorer) judge pipeline against a labeled corpus.

    For each *.labels.json file under `labeled_corpus_dir`:
      1. Load the referenced trajectory via parser.load_trajectory.
      2. Run normalize_steps (with classifier_judge) -> score_suspicious_steps
         (with scorer_judge).
      3. Compare predictions to labels and accumulate counts.

    Trajectories whose file is missing are counted in skipped_trajectories.
    """
    labels = discover_labels(labeled_corpus_dir)

    tp = fp = fn = 0
    critical_hits = 0
    critical_eligible = 0
    category_correct = 0
    category_eligible = 0
    skipped = 0
    per_trajectory: list[dict] = []

    for labeled in labels:
        try:
            trajectory = load_trajectory(str(labeled.trajectory_path))
        except FileNotFoundError:
            skipped += 1
            per_trajectory.append({
                "trajectory": str(labeled.trajectory_path),
                "skipped": True,
            })
            continue

        normalized = normalize_steps(trajectory.steps, judge=classifier_judge)
        scored = score_suspicious_steps(
            normalized, trajectory.task, trajectory.final_status, judge=scorer_judge,
        )

        traj_tp = traj_fp = traj_fn = 0
        for step in scored:
            predicted = step.suspicious_score > 0
            actual = labeled.is_suspicious(step.step_id)
            if predicted and actual:
                traj_tp += 1
                actual_cat = labeled.category_for(step.step_id)
                predicted_cat = _category_for_prediction(step)
                if actual_cat is not None and predicted_cat is not None:
                    category_eligible += 1
                    if actual_cat == predicted_cat:
                        category_correct += 1
            elif predicted and not actual:
                traj_fp += 1
            elif not predicted and actual:
                traj_fn += 1

        tp += traj_tp
        fp += traj_fp
        fn += traj_fn

        traj_critical_hit: bool | None = None
        if labeled.final_status == "failed" and labeled.critical_step_id is not None:
            critical_eligible += 1
            top = max(scored, key=lambda s: s.suspicious_score, default=None)
            hit = (
                top is not None
                and top.suspicious_score > 0
                and top.step_id == labeled.critical_step_id
            )
            traj_critical_hit = hit
            if hit:
                critical_hits += 1

        per_trajectory.append({
            "trajectory": str(labeled.trajectory_path),
            "tp": traj_tp,
            "fp": traj_fp,
            "fn": traj_fn,
            "critical_hit": traj_critical_hit,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(precision, recall)
    hit_at_1 = critical_hits / critical_eligible if critical_eligible > 0 else 0.0
    cat_acc = category_correct / category_eligible if category_eligible > 0 else None

    return JudgeMetrics(
        suspicious_precision=precision,
        suspicious_recall=recall,
        suspicious_f1=f1,
        critical_hit_at_1=hit_at_1,
        category_accuracy=cat_acc,
        labeled_trajectories=len(labels) - skipped,
        skipped_trajectories=skipped,
        per_trajectory=per_trajectory,
    )
```

Move the new imports (`from analyzer import score_suspicious_steps`, `from classifier import normalize_steps`, `from parser import load_trajectory`) into the top-of-file import block so the module's imports are all at the top.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_judge_eval.EvaluateJudgesTests -v 2>&1 | tail -15
python -m unittest discover 2>&1 | tail -5
```

Expected: all four `EvaluateJudgesTests` pass; the full suite still passes.

- [ ] **Step 5: Commit**

```bash
git add judge_eval.py tests/test_judge_eval.py
git commit -m "feat: add evaluate_judges harness composing parser, classifier, scorer"
```

---

## Task 5: Output formatting

**Files:**
- Modify: `judge_eval.py` (add `format_metrics_md`, `metrics_to_dict`)
- Modify: `tests/test_judge_eval.py` (add format tests)

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_judge_eval.py` before the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m unittest tests.test_judge_eval.FormatMetricsTests -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'format_metrics_md' from 'judge_eval'`.

- [ ] **Step 3: Add formatting functions to `judge_eval.py`**

Append to `judge_eval.py`:

```python
def format_metrics_md(metrics: JudgeMetrics) -> str:
    if metrics.category_accuracy is None:
        cat_line = "Category accuracy: n/a (no overlap)"
    else:
        cat_line = f"Category accuracy: {metrics.category_accuracy:.3f}"

    return "\n".join([
        "# Judge Eval",
        "",
        f"Labeled trajectories: {metrics.labeled_trajectories}",
        f"Skipped (no trajectory file): {metrics.skipped_trajectories}",
        "",
        "## Suspicious-step detection",
        f"Precision: {metrics.suspicious_precision:.3f}",
        f"Recall:    {metrics.suspicious_recall:.3f}",
        f"F1:        {metrics.suspicious_f1:.3f}",
        "",
        "## Critical-step localization",
        f"hit@1: {metrics.critical_hit_at_1:.3f}",
        "",
        f"## {cat_line}",
        "",
    ])


def metrics_to_dict(metrics: JudgeMetrics) -> dict:
    return {
        "suspicious_precision": metrics.suspicious_precision,
        "suspicious_recall": metrics.suspicious_recall,
        "suspicious_f1": metrics.suspicious_f1,
        "critical_hit_at_1": metrics.critical_hit_at_1,
        "category_accuracy": metrics.category_accuracy,
        "labeled_trajectories": metrics.labeled_trajectories,
        "skipped_trajectories": metrics.skipped_trajectories,
        "per_trajectory": metrics.per_trajectory,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_judge_eval.FormatMetricsTests -v 2>&1 | tail -10
```

Expected: all three `FormatMetricsTests` pass.

- [ ] **Step 5: Commit**

```bash
git add judge_eval.py tests/test_judge_eval.py
git commit -m "feat: add Markdown and dict formatters for JudgeMetrics"
```

---

## Task 6: CLI subcommand

**Files:**
- Modify: `main.py` (register `judge-eval` subparser + handler)
- Modify: `tests/test_judge_eval.py` (CLI argparse + integration tests)

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_judge_eval.py` before the `if __name__ == "__main__":` block:

```python
class JudgeEvalCLITests(unittest.TestCase):
    def test_parser_accepts_judge_eval_subcommand(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["judge-eval", "--labels", "tests/fixtures/labels"])
        self.assertEqual(args.command, "judge-eval")
        self.assertEqual(args.format, "text")
        self.assertEqual(args.judge, "rule")

    def test_parser_accepts_format_json(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "judge-eval", "--labels", "tests/fixtures/labels", "--format", "json",
        ])
        self.assertEqual(args.format, "json")

    def test_run_judge_eval_prints_json_when_requested(self):
        import io
        from contextlib import redirect_stdout
        from main import build_parser, run_judge_eval_command

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "x.labels.json").write_text(json.dumps({
                "trajectory": "examples/codex_failed_run_001.jsonl",
                "labeler": "test",
                "labeled_at": "2026-05-14",
                "final_status": "failed",
                "steps": [],
            }))
            parser = build_parser()
            args = parser.parse_args([
                "judge-eval", "--labels", str(tmp), "--format", "json",
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = run_judge_eval_command(args)
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertIn("suspicious_precision", payload)
            self.assertIn("labeled_trajectories", payload)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m unittest tests.test_judge_eval.JudgeEvalCLITests -v 2>&1 | tail -15
```

Expected: `argument command: invalid choice: 'judge-eval'`.

- [ ] **Step 3: Register the subparser in `main.py`**

Edit `main.py`. Find the block where other subparsers are registered (after the `eval_parser` block around `main.py:302-307`). Add this immediately after `eval_parser.set_defaults(func=run_eval_command)`:

```python
    judge_eval_parser = subparsers.add_parser(
        "judge-eval",
        help="Score a judge against a labeled trajectory corpus",
    )
    judge_eval_parser.add_argument(
        "--labels",
        required=True,
        help="Directory containing *.labels.json sidecar files",
    )
    judge_eval_parser.add_argument(
        "--judge",
        choices=["rule"],
        default="rule",
        help="Which judge to evaluate (default: rule)",
    )
    judge_eval_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    judge_eval_parser.set_defaults(func=run_judge_eval_command)
```

- [ ] **Step 4: Add `run_judge_eval_command` to `main.py`**

In `main.py`, add this function near the existing `run_eval_command` (after the `run_eval_command` definition, around `main.py:95`):

```python
def run_judge_eval_command(args: argparse.Namespace) -> int:
    try:
        from judge_eval import evaluate_judges, format_metrics_md, metrics_to_dict

        # --judge is currently "rule" only; future LLM judges will branch here.
        metrics = evaluate_judges(Path(args.labels))

        if args.format == "json":
            print(json.dumps(metrics_to_dict(metrics), indent=2))
        else:
            print(format_metrics_md(metrics))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m unittest tests.test_judge_eval.JudgeEvalCLITests -v 2>&1 | tail -15
python -m unittest discover 2>&1 | tail -5
```

Expected: all three `JudgeEvalCLITests` pass; full suite still passes.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_judge_eval.py
git commit -m "feat: add judge-eval CLI subcommand"
```

---

## Task 7: Seed corpus

**Files:**
- Create: `tests/fixtures/labels/codex_failed_run_001.labels.json`
- Create: `tests/fixtures/labels/<n>.labels.json` — one per remaining real trajectory in `data/lcb/trajectories/` and `data/swe/trajectories/`

This task is human-judgment work — open each trajectory, decide which steps are suspicious, write the sidecar. The plan provides one fully-worked example plus a precise procedure for the rest.

- [ ] **Step 1: Create the labels directory**

```bash
mkdir -p tests/fixtures/labels
```

- [ ] **Step 2: Label `examples/codex_failed_run_001.jsonl`**

Per `CLAUDE.md`, this fixture has step 5 as the test bypass (a `test_edit_after_impl_failure` case). Write `tests/fixtures/labels/codex_failed_run_001.labels.json`:

```json
{
  "trajectory": "examples/codex_failed_run_001.jsonl",
  "labeler": "human",
  "labeled_at": "2026-05-14",
  "final_status": "failed",
  "critical_step_id": 5,
  "steps": [
    {"step_id": 5, "suspicious": true, "category": "test_edit_after_impl_failure"}
  ]
}
```

- [ ] **Step 3: Discover the remaining real trajectories**

```bash
ls data/lcb/trajectories/*.jsonl data/swe/trajectories/*.jsonl 2>/dev/null
```

Expected (current snapshot): `data/lcb/trajectories/easy_1873_A.jsonl`, `data/lcb/trajectories/hard_1899_B.jsonl`, `data/swe/trajectories/astropy__astropy-12907.jsonl`. If more files have appeared since the spec, include them too.

- [ ] **Step 4: For each remaining trajectory, label it**

For each trajectory `<path>` from Step 3, follow this procedure:

1. Run the rule judge to see what it flags:
   ```bash
   python main.py eval --input <path> --output /tmp/judge_inspect --quiet
   cat /tmp/judge_inspect/normalized_steps.json | python -c "import json, sys; steps=json.load(sys.stdin); [print(s['step_id'], round(s['suspicious_score'],2), s['action'][:80]) for s in steps if s['suspicious_score']>0]"
   ```
2. Read the raw trajectory's first 100 lines to understand the task:
   ```bash
   head -100 <path>
   ```
3. Read `/tmp/judge_inspect/diagnosis.md` to see the rule judge's verdict.
4. Decide for each rule-flagged step whether it is truly suspicious; record the call. Also scan for any clearly-suspicious step the rule judge missed (false negatives matter more than false positives here).
5. Pick `critical_step_id` only if `final_status` is `"failed"` AND there is a clear single step that caused the failure. Otherwise omit the field.
6. Pick `category` only for steps where the failure mode obviously matches a `PATTERNS` key. Acceptable values are: `test_edit_no_impl`, `test_edit_after_impl_failure`, `patch_then_failing_test`, `tests_pass_after_test_only_edit`, `repeated_command`, `repeated_failing_test`, `env_change_then_dependency_error`, `git_rollback`, `excessive_summarization`, `loop_detection_triggered`, `subagent_failure`, `tool_error`. Omit `category` if none cleanly applies.
7. Determine `final_status` from `/tmp/judge_inspect/eval_result.json`:
   ```bash
   python -c "import json; d=json.load(open('/tmp/judge_inspect/eval_result.json')); print(d['final_status'])"
   ```
8. Write `tests/fixtures/labels/<basename>.labels.json` where `<basename>` is the trajectory's filename without extension. Example schema (substitute your decisions):

   ```json
   {
     "trajectory": "data/lcb/trajectories/easy_1873_A.jsonl",
     "labeler": "human",
     "labeled_at": "2026-05-14",
     "final_status": "success",
     "steps": []
   }
   ```

- [ ] **Step 5: Verify the corpus loads cleanly**

```bash
python -c "from pathlib import Path; from judge_eval import discover_labels; labels=discover_labels(Path('tests/fixtures/labels')); print(len(labels), 'label files loaded'); [print(' -', l.trajectory_path, l.final_status, len(l.step_labels), 'step labels') for l in labels]"
```

Expected: each file lists with its final_status and step-label count. No exceptions raised (which would indicate an unknown category).

- [ ] **Step 6: Run judge-eval against the seed corpus**

```bash
python main.py judge-eval --labels tests/fixtures/labels/
```

Expected: a Markdown report with non-zero `Labeled trajectories`. The rule judge's `Precision` and `Recall` should both be > 0 (otherwise the corpus has nothing in common with the rule judge — re-examine the labels).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/labels/
git commit -m "test: add seed corpus for judge-eval harness"
```

---

## Task 8: Final verification

- [ ] **Step 1: Full test suite passes**

```bash
python -m unittest discover -v 2>&1 | tail -10
```

Expected: all existing 141 + new tests pass with no failures or errors.

- [ ] **Step 2: Text-mode CLI smoke test**

```bash
python main.py judge-eval --labels tests/fixtures/labels/
```

Expected: Markdown report listing precision / recall / F1 / hit@1 / category accuracy with non-zero coverage.

- [ ] **Step 3: JSON-mode CLI smoke test**

```bash
python main.py judge-eval --labels tests/fixtures/labels/ --format json | python -c "import json, sys; d=json.load(sys.stdin); assert 'suspicious_f1' in d and 'per_trajectory' in d; print('JSON OK')"
```

Expected: `JSON OK`.

- [ ] **Step 4: Help text shows the new subcommand**

```bash
python main.py --help | grep judge-eval
python main.py judge-eval --help | grep -- --labels
```

Expected: both grep lines return a match.

- [ ] **Step 5: Final commit (if anything stray remains)**

```bash
git status
git diff --cached --quiet || git commit -m "chore: judge-eval harness final pass"
```
