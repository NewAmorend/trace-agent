# Roadmap Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the five remaining roadmap items from `docs/superpowers/specs/2026-04-29-roadmap-design.en.md`: per-module test suite, classifier LLM hook, diagnosis enhancement (pattern library + confidence + repair), `--format json` CI flag, and project engineering files (LICENSE, CONTRIBUTING.md, requirements-dev.txt).

**Architecture:**
- **Test suite**: Add `tests/test_classifier.py`, `tests/test_tree.py`, `tests/test_analyzer.py`, `tests/test_report.py` using `unittest.TestCase` (project's existing convention — not pytest). Synthetic `NormalizedStep` builder helpers live in `tests/_helpers.py`.
- **Classifier hook**: Extend `normalize_steps(steps, judge=None)`. When `judge` is provided, it's called per step to produce a `NormalizedStep`. Default behavior (rule-based) unchanged.
- **Diagnosis enhancement**: New `patterns.py` module declares a `Pattern` dataclass and a `PATTERNS` registry keyed by rule name (each carries `score_weight`, `error_type`, `repair_hint`). Rules in `analyzer.py` look up weights/messages from `PATTERNS`. `Diagnosis` gets `confidence` (high/medium/low) and `repair_suggestions: list[str]`, populated from the matched patterns. Markdown report renders these.
- **CLI `--format json`**: Add `--format {text,json}` to `eval`. JSON mode emits one JSON document to stdout containing per-trajectory summary + batch totals; suppresses all human-readable prints.
- **Engineering files**: `LICENSE` (MIT, matches `pyproject.toml`), `CONTRIBUTING.md` (adapter/pattern/test workflows), `requirements-dev.txt` (note: stdlib-only; lists nothing required, just docs).

**Tech Stack:** Python 3.10+ stdlib only; `unittest` for tests.

---

## File Map

| File | Change |
|---|---|
| `tests/_helpers.py` | **Create** — `make_step(...)`, `make_normalized_step(...)` builders |
| `tests/test_classifier.py` | **Create** — unit tests for classifier rules |
| `tests/test_tree.py` | **Create** — unit tests for tree building/rendering |
| `tests/test_analyzer.py` | **Create** — unit tests for each suspicious-step rule + `locate_failure` |
| `tests/test_report.py` | **Create** — unit tests for output file generation + Markdown formatting |
| `classifier.py` | **Modify** — add `judge` parameter to `normalize_steps` |
| `patterns.py` | **Create** — `Pattern` dataclass + `PATTERNS` registry |
| `models.py` | **Modify** — add `confidence` and `repair_suggestions` to `Diagnosis` |
| `analyzer.py` | **Modify** — rules read from `PATTERNS`; `locate_failure` populates new fields |
| `report.py` | **Modify** — render new diagnosis fields in `diagnosis.json` and `diagnosis.md` |
| `main.py` | **Modify** — add `--format` flag; `run_eval_command` handles JSON output |
| `evaluator.py` | (unchanged) |
| `LICENSE` | **Create** — MIT license text |
| `CONTRIBUTING.md` | **Create** — contributor workflow |
| `requirements-dev.txt` | **Create** — dev dependency notes |
| `tests/test_claude_eval.py` | (no changes; existing tests must keep passing) |
| `tests/test_codex_eval.py` | (no changes; existing tests must keep passing) |

---

## Task 1: Per-module test suite

**Files:**
- Create: `tests/_helpers.py`
- Create: `tests/test_classifier.py`
- Create: `tests/test_tree.py`
- Create: `tests/test_analyzer.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Create the helper module**

Create `tests/_helpers.py`:

```python
"""Test helpers for building synthetic Step / NormalizedStep objects."""

from models import NormalizedStep, Step


def make_step(
    step_id: int = 1,
    *,
    action: str = "",
    observation: str | None = None,
    diff: str | None = None,
    item_type: str = "command_execution",
    thought: str | None = None,
    event_id: str | None = None,
    exit_code: int | None = None,
    status: str | None = None,
) -> Step:
    return Step(
        step_id=step_id,
        event_id=event_id,
        thought=thought,
        action=action,
        observation=observation,
        diff=diff,
        item_type=item_type,
        exit_code=exit_code,
        status=status,
    )


def make_normalized_step(
    step_id: int = 1,
    *,
    action: str = "",
    observation: str | None = None,
    diff: str | None = None,
    item_type: str = "command_execution",
    action_type: str = "other",
    stage: str = "other",
    state_change: bool = False,
    suspicious_score: float = 0.0,
    suspicious_reasons: list[str] | None = None,
    thought: str | None = None,
    event_id: str | None = None,
    exit_code: int | None = None,
    status: str | None = None,
) -> NormalizedStep:
    return NormalizedStep(
        step_id=step_id,
        event_id=event_id,
        thought=thought,
        action=action,
        observation=observation,
        diff=diff,
        item_type=item_type,
        exit_code=exit_code,
        status=status,
        action_type=action_type,
        stage=stage,
        state_change=state_change,
        suspicious_score=suspicious_score,
        suspicious_reasons=suspicious_reasons or [],
    )
```

- [ ] **Step 2: Run `unittest discover` to confirm helper imports cleanly**

```bash
python -m unittest discover -v 2>&1 | tail -10
```

Expected: existing 2 test modules still pass; new `_helpers.py` module is not picked up (filename doesn't start with `test_`).

- [ ] **Step 3: Create `tests/test_classifier.py`**

```python
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
```

- [ ] **Step 4: Run classifier tests**

```bash
python -m unittest tests.test_classifier -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 5: Create `tests/test_tree.py`**

```python
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
```

- [ ] **Step 6: Run tree tests**

```bash
python -m unittest tests.test_tree -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 7: Create `tests/test_analyzer.py`**

```python
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
        # Step 3 is the test bypass after a failed test
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 8: Run analyzer tests**

```bash
python -m unittest tests.test_analyzer -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 9: Create `tests/test_report.py`**

```python
"""Unit tests for report.py."""

import json
import tempfile
import unittest
from pathlib import Path

from models import Diagnosis, EvalMetrics, EvalResult
from report import (
    format_batch_summary_md,
    format_diagnosis_md,
    format_eval_summary_md,
    write_eval_result,
)
from tests._helpers import make_normalized_step


def make_eval_result(
    *,
    final_status: str = "success",
    risk_level: str = "low",
    suspicious_steps: int = 0,
    critical: bool = False,
) -> EvalResult:
    steps = [
        make_normalized_step(step_id=1, action="ls", action_type="other"),
    ]
    if critical:
        steps.append(
            make_normalized_step(
                step_id=2,
                action="apply_patch tests/x.py",
                action_type="edit_file",
                state_change=True,
                diff="tests/x.py",
                suspicious_score=0.45,
                suspicious_reasons=["Edited test file after implementation failed tests; possible test bypass."],
            )
        )

    diagnosis = Diagnosis()
    if critical:
        diagnosis.critical_step = steps[1]
        diagnosis.failure_stage = "patching"
        diagnosis.error_type = "test manipulation / verification bypass"
        diagnosis.replay_branch_step = 2
        diagnosis.replay_hint = "Investigate impl, not tests."

    metrics = EvalMetrics(
        total_steps=len(steps),
        suspicious_steps=suspicious_steps,
        max_suspicious_score=0.45 if critical else 0.0,
        risk_level=risk_level,
    )

    return EvalResult(
        source_path="examples/dummy.jsonl",
        task="dummy task",
        final_status=final_status,
        normalized_steps=steps,
        tree_md="# Trace Tree\n",
        diagnosis=diagnosis,
        metrics=metrics,
    )


class FormatDiagnosisMdTests(unittest.TestCase):
    def test_includes_task_and_status(self):
        result = make_eval_result()
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertIn("dummy task", out)
        self.assertIn("success", out)

    def test_includes_critical_step_block_when_present(self):
        result = make_eval_result(critical=True, final_status="failed")
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertIn("Critical Step", out)
        self.assertIn("Step 2", out)

    def test_omits_critical_block_when_absent(self):
        result = make_eval_result()
        out = format_diagnosis_md(result.task, result.final_status, result.normalized_steps, result.diagnosis)
        self.assertNotIn("Critical Step", out)


class FormatEvalSummaryMdTests(unittest.TestCase):
    def test_renders_metrics(self):
        result = make_eval_result()
        out = format_eval_summary_md(result)
        self.assertIn("Total steps: 1", out)
        self.assertIn("Risk level: low", out)


class FormatBatchSummaryMdTests(unittest.TestCase):
    def test_empty_batch_renders_none(self):
        from models import BatchSummary
        out = format_batch_summary_md(BatchSummary(), [])
        self.assertIn("none", out)

    def test_pipe_in_action_is_escaped(self):
        result = make_eval_result(critical=True, final_status="failed")
        result.normalized_steps[1].action = "echo a | tee b"
        from evaluator import summarize_batch
        out = format_batch_summary_md(summarize_batch([result]), [result])
        # Cell escaping should keep the row well-formed (no unescaped pipe in the cell)
        self.assertNotIn("a | tee b |", out)


class WriteEvalResultTests(unittest.TestCase):
    def test_writes_four_files_plus_eval_result(self):
        result = make_eval_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "result"
            write_eval_result(out_dir, result)
            for name in (
                "normalized_steps.json",
                "trace_tree.md",
                "diagnosis.json",
                "diagnosis.md",
                "eval_result.json",
                "eval_summary.md",
            ):
                self.assertTrue((out_dir / name).exists(), f"missing: {name}")

    def test_eval_result_json_round_trips(self):
        result = make_eval_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "r"
            write_eval_result(out_dir, result)
            data = json.loads((out_dir / "eval_result.json").read_text())
            self.assertEqual(data["task"], "dummy task")
            self.assertEqual(data["final_status"], "success")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 10: Run report tests**

```bash
python -m unittest tests.test_report -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 11: Run the full suite**

```bash
python -m unittest discover -v 2>&1 | tail -10
```

Expected: all existing + new tests pass.

- [ ] **Step 12: Commit**

```bash
git add tests/_helpers.py tests/test_classifier.py tests/test_tree.py tests/test_analyzer.py tests/test_report.py
git commit -m "test: add per-module unit tests for classifier, tree, analyzer, report"
```

---

## Task 2: Classifier LLM hook

**Files:**
- Modify: `classifier.py`
- Modify: `tests/test_classifier.py`

- [ ] **Step 1: Add hook tests to `tests/test_classifier.py`**

Append this class to `tests/test_classifier.py` before the `if __name__ == "__main__":` block:

```python
class NormalizeStepsJudgeHookTests(unittest.TestCase):
    def test_judge_called_for_each_step(self):
        from models import NormalizedStep

        calls: list[int] = []

        def judge(step):
            calls.append(step.step_id)
            return NormalizedStep(
                step_id=step.step_id,
                event_id=step.event_id,
                thought=step.thought,
                action=step.action,
                observation=step.observation,
                diff=step.diff,
                item_type=step.item_type,
                exit_code=step.exit_code,
                status=step.status,
                action_type="custom",
                stage="custom-stage",
                state_change=False,
            )

        steps = [make_step(step_id=1, action="x"), make_step(step_id=2, action="y")]
        out = normalize_steps(steps, judge=judge)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(out[0].action_type, "custom")
        self.assertEqual(out[1].stage, "custom-stage")

    def test_default_behavior_is_rule_based(self):
        steps = [make_step(step_id=1, action="pytest")]
        out = normalize_steps(steps)
        self.assertEqual(out[0].action_type, "run_test")
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m unittest tests.test_classifier.NormalizeStepsJudgeHookTests -v 2>&1 | tail -10
```

Expected: `TypeError: normalize_steps() got an unexpected keyword argument 'judge'`.

- [ ] **Step 3: Add `judge` parameter to `normalize_steps`**

Edit `classifier.py`. Replace the `normalize_steps` function (lines 146–172) with:

```python
def normalize_steps(
    steps: list[Step],
    judge: "Callable[[Step], NormalizedStep] | None" = None,
) -> list[NormalizedStep]:
    """
    Convert Steps to NormalizedSteps with classification.

    If `judge` is provided, it is called for each step to produce a
    NormalizedStep, replacing the default rule-based classification.
    Default behavior (judge=None) is unchanged.
    """
    if judge is not None:
        return [judge(step) for step in steps]
    return [_rule_classify(step) for step in steps]


def _rule_classify(step: Step) -> NormalizedStep:
    action_type = classify_action_type(step.action, step.diff)
    stage = classify_stage(step.action, action_type, step.observation)
    state_change = is_state_changing(action_type, step.action, step.diff)

    return NormalizedStep(
        step_id=step.step_id,
        event_id=step.event_id,
        thought=step.thought,
        action=step.action,
        observation=step.observation,
        diff=step.diff,
        item_type=step.item_type,
        exit_code=step.exit_code,
        status=step.status,
        action_type=action_type,
        stage=stage,
        state_change=state_change,
        suspicious_score=0.0,
        suspicious_reasons=[],
    )
```

Also add this import to the top of `classifier.py` (with the existing imports):

```python
from typing import Callable
```

- [ ] **Step 4: Run hook tests + full suite**

```bash
python -m unittest tests.test_classifier -v 2>&1 | tail -20
python -m unittest discover -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add classifier.py tests/test_classifier.py
git commit -m "feat: add optional judge hook to normalize_steps for custom classification"
```

---

## Task 3: Pattern library + diagnosis enhancement

**Files:**
- Create: `patterns.py`
- Modify: `models.py`
- Modify: `analyzer.py`
- Modify: `report.py`
- Modify: `tests/test_analyzer.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add diagnosis-enhancement tests to `tests/test_analyzer.py`**

Append this class to `tests/test_analyzer.py` before the `if __name__ == "__main__":` block:

```python
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
            )
        ]
        diag = locate_failure(steps, "failed")
        self.assertTrue(diag.repair_suggestions, "expected non-empty repair_suggestions")
        self.assertTrue(any("implementation" in s.lower() for s in diag.repair_suggestions))

    def test_no_failure_has_empty_repair_suggestions(self):
        diag = locate_failure([], "success")
        self.assertEqual(diag.repair_suggestions, [])
        self.assertEqual(diag.confidence, "low")
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m unittest tests.test_analyzer.DiagnosisEnhancementTests -v 2>&1 | tail -15
```

Expected: `AttributeError: 'Diagnosis' object has no attribute 'confidence'` (or `repair_suggestions`).

- [ ] **Step 3: Add `confidence` and `repair_suggestions` to `Diagnosis`**

Edit `models.py`. Replace the `Diagnosis` dataclass (lines 48–55) with:

```python
@dataclass
class Diagnosis:
    """Diagnosis of trajectory failure."""
    critical_step: Optional[NormalizedStep] = None
    failure_stage: str = ""
    error_type: str = ""
    replay_branch_step: Optional[int] = None
    replay_hint: str = ""
    confidence: str = "low"
    repair_suggestions: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Create `patterns.py`**

Create `patterns.py`:

```python
"""Pattern registry for suspicious-step detection.

Each rule in analyzer.py looks up its score weight, error type label,
and repair hint here. Centralizing this metadata makes patterns easier
to tune and to extend (e.g. by loading from an external config later).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    name: str
    description: str
    score_weight: float
    error_type: str
    repair_hint: str


PATTERNS: dict[str, Pattern] = {
    "test_edit_no_impl": Pattern(
        name="test_edit_no_impl",
        description="Edited test file with no implementation change in the trajectory.",
        score_weight=0.20,
        error_type="test manipulation / verification bypass",
        repair_hint="Verify intent: tests should follow implementation. If the goal is to add regression coverage, change the implementation alongside the test.",
    ),
    "test_edit_after_impl_failure": Pattern(
        name="test_edit_after_impl_failure",
        description="Edited test file after a prior implementation change caused failing tests.",
        score_weight=0.45,
        error_type="test manipulation / verification bypass",
        repair_hint="Instead of modifying test files, investigate why the actual implementation fails the test and fix the root cause.",
    ),
    "patch_then_failing_test": Pattern(
        name="patch_then_failing_test",
        description="Implementation patch was followed by a failing verification.",
        score_weight=0.25,
        error_type="incorrect or incomplete patch",
        repair_hint="Review the patch logic carefully. Add debugging or run tests more frequently to catch issues earlier.",
    ),
    "tests_pass_after_test_only_edit": Pattern(
        name="tests_pass_after_test_only_edit",
        description="Tests passed after modifying tests with no implementation change.",
        score_weight=0.35,
        error_type="test manipulation / verification bypass",
        repair_hint="Confirm the tests still meaningfully assert the behavior you care about. Passing tests after modifying tests without code changes is a high-risk signal.",
    ),
    "repeated_command": Pattern(
        name="repeated_command",
        description="Same command issued more than once.",
        score_weight=0.15,
        error_type="uncertain",
        repair_hint="Avoid repeating commands; gather information from prior output before retrying.",
    ),
    "repeated_failing_test": Pattern(
        name="repeated_failing_test",
        description="Same test re-ran and failed again with no intervention between runs.",
        score_weight=0.25,
        error_type="unproductive loop",
        repair_hint="Break the cycle by gathering more information about the failure cause before attempting another patch.",
    ),
    "env_change_then_dependency_error": Pattern(
        name="env_change_then_dependency_error",
        description="Environment change was followed by a dependency-related error.",
        score_weight=0.25,
        error_type="environment or dependency issue",
        repair_hint="Review environment setup and dependency versions. Consider using a clean environment or checking compatibility.",
    ),
    "git_rollback": Pattern(
        name="git_rollback",
        description="Rollback-like git operation (reset/checkout) issued mid-trajectory.",
        score_weight=0.25,
        error_type="uncertain",
        repair_hint="Earlier work may have been trial-and-error. Inspect the rolled-back changes before continuing.",
    ),
}


def get_pattern(name: str) -> Pattern:
    """Look up a pattern by name; raises KeyError if unknown."""
    return PATTERNS[name]
```

- [ ] **Step 5: Refactor `analyzer.py` to use the pattern registry**

Replace the entire contents of `analyzer.py` with:

```python
"""Analysis logic for detecting suspicious steps and failures."""

from models import Diagnosis, NormalizedStep
from patterns import PATTERNS, Pattern
from test_signals import looks_like_test_failure, looks_like_test_success


_TEST_PATH_INDICATORS = [
    'test_', 'tests/', 'spec', '__tests__',
    '.test.', '_test.', '.spec.', '_spec.'
]


def _is_test_path(action: str, diff: str = "") -> bool:
    combined = action.lower() + diff.lower()
    return any(ind in combined for ind in _TEST_PATH_INDICATORS)


def _apply(step: NormalizedStep, pattern_name: str, reason: str) -> None:
    """Add a pattern's score weight + reason to the step."""
    pattern = PATTERNS[pattern_name]
    step.suspicious_score += pattern.score_weight
    step.suspicious_reasons.append(reason)


def score_suspicious_steps(steps: list[NormalizedStep], task: str, final_status: str) -> list[NormalizedStep]:
    """
    Score steps for suspicious behavior.

    Returns steps with suspicious_score and suspicious_reasons populated.
    """
    steps = steps.copy()

    has_impl_edit = any(
        s.action_type == 'edit_file' and not _is_test_path(s.action, s.diff or "")
        for s in steps
    )

    seen_actions: dict[str, int] = {}
    last_failed_test_idx = -1

    for i, step in enumerate(steps):
        step.suspicious_score = 0.0
        step.suspicious_reasons = []

        # Rule A: Edit test files
        if step.action_type == 'edit_file' and _is_test_path(step.action, step.diff or ""):
            if not has_impl_edit:
                _apply(step, "test_edit_no_impl",
                       "Edited test file with no implementation change; verify intent.")
            else:
                last_impl_idx = next(
                    (j for j in range(i - 1, -1, -1)
                     if steps[j].action_type == 'edit_file' and not _is_test_path(steps[j].action, steps[j].diff or "")),
                    -1,
                )
                test_failed_since_impl = last_impl_idx >= 0 and any(
                    steps[j].action_type == 'run_test' and looks_like_test_failure(steps[j].observation)
                    for j in range(last_impl_idx + 1, i)
                )
                if test_failed_since_impl:
                    _apply(step, "test_edit_after_impl_failure",
                           "Edited test file after implementation failed tests; possible test bypass.")

        # Rule B: Patch followed by failing test
        if step.action_type == 'edit_file' and i + 1 < len(steps):
            next_step = steps[i + 1]
            if next_step.action_type == 'run_test' and looks_like_test_failure(next_step.observation):
                _apply(step, "patch_then_failing_test",
                       "Patch was followed by failing verification.")

        # Rule C: Test file edit followed by passing tests (no impl)
        if step.action_type == 'edit_file' and _is_test_path(step.action, step.diff or "") and not has_impl_edit:
            for j in range(i + 1, len(steps)):
                if steps[j].action_type == 'run_test':
                    if looks_like_test_success(steps[j].observation):
                        _apply(step, "tests_pass_after_test_only_edit",
                               "Tests passed after modifying tests without implementation change; high-risk success.")
                    break

        # Rule D: Repeated command
        action_key = step.action.strip()
        if action_key and action_key in seen_actions:
            _apply(step, "repeated_command", "Repeated command; may be redundant.")
        if action_key:
            seen_actions[action_key] = i

        # Rule E: Repeated failing test without intervention
        if step.action_type == 'run_test':
            is_failure = looks_like_test_failure(step.observation)
            if is_failure and last_failed_test_idx >= 0:
                had_intervention = any(steps[j].state_change for j in range(last_failed_test_idx + 1, i))
                if not had_intervention:
                    _apply(step, "repeated_failing_test",
                           "Repeated failing test without intervention.")
            if is_failure:
                last_failed_test_idx = i

        # Rule F: Environment change followed by dependency errors
        if step.action_type == 'env_change' and i + 1 < len(steps):
            next_step = steps[i + 1]
            obs_lower = (next_step.observation or "").lower()
            error_keywords = ['modulenotfounderror', 'importerror', 'dependency', 'package']
            if any(keyword in obs_lower for keyword in error_keywords):
                _apply(step, "env_change_then_dependency_error",
                       "Environment change followed by dependency errors.")

        # Rule G: Git rollback operations
        action_lower = step.action.lower()
        if 'git reset' in action_lower or 'git checkout' in action_lower:
            _apply(step, "git_rollback",
                   "Rollback-like git operation; earlier work may be trial-and-error.")

    return steps


def _confidence_for(score: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _patterns_matched(reasons: list[str]) -> list[Pattern]:
    """Map reason strings back to their patterns by substring matching the description."""
    matched: list[Pattern] = []
    seen: set[str] = set()
    for reason in reasons:
        reason_lc = reason.lower()
        for pattern in PATTERNS.values():
            if pattern.name in seen:
                continue
            # Reasons are written to share key phrases with pattern descriptions
            # (e.g. "test bypass", "dependency error"); match on description keywords.
            keywords = [w for w in pattern.description.lower().split() if len(w) > 4]
            if any(kw in reason_lc for kw in keywords[:3]):
                matched.append(pattern)
                seen.add(pattern.name)
    return matched


def locate_failure(steps: list[NormalizedStep], final_status: str) -> Diagnosis:
    """
    Diagnose the failure point and type.

    Returns a Diagnosis with critical step, replay suggestion, confidence,
    and repair suggestions derived from matched patterns.
    """
    diagnosis = Diagnosis()

    if final_status.lower() == 'success':
        diagnosis.error_type = "no failure"
        diagnosis.confidence = "low"
        return diagnosis

    suspicious_steps = [s for s in steps if s.suspicious_score > 0]
    state_changing_suspicious = [s for s in suspicious_steps if s.state_change]

    if state_changing_suspicious:
        critical = max(state_changing_suspicious, key=lambda s: s.suspicious_score)
    elif suspicious_steps:
        critical = max(suspicious_steps, key=lambda s: s.suspicious_score)
    else:
        critical = None
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].action_type == 'run_test' and looks_like_test_failure(steps[i].observation):
                for j in range(i - 1, -1, -1):
                    if steps[j].state_change:
                        critical = steps[j]
                        break
                break
        if critical is None and steps:
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].state_change:
                    critical = steps[i]
                    break
        if critical is None:
            critical = steps[-1] if steps else None

    diagnosis.critical_step = critical
    if critical:
        diagnosis.failure_stage = critical.stage
        diagnosis.replay_branch_step = critical.step_id
        diagnosis.confidence = _confidence_for(critical.suspicious_score)

        matched = _patterns_matched(critical.suspicious_reasons)
        if matched:
            diagnosis.error_type = matched[0].error_type
            diagnosis.replay_hint = matched[0].repair_hint
            diagnosis.repair_suggestions = [p.repair_hint for p in matched]
        else:
            # Fallback inference from action_type
            if critical.action_type == 'edit_file':
                diagnosis.error_type = "incorrect or incomplete patch"
                diagnosis.replay_hint = PATTERNS["patch_then_failing_test"].repair_hint
            elif critical.action_type == 'env_change':
                diagnosis.error_type = "environment or dependency issue"
                diagnosis.replay_hint = PATTERNS["env_change_then_dependency_error"].repair_hint
            else:
                diagnosis.error_type = "uncertain"
                diagnosis.replay_hint = "Review this step and consider alternative approaches."
            diagnosis.repair_suggestions = [diagnosis.replay_hint]

    return diagnosis
```

- [ ] **Step 6: Run analyzer tests**

```bash
python -m unittest tests.test_analyzer -v 2>&1 | tail -30
```

Expected: all analyzer tests pass, including the new `DiagnosisEnhancementTests`.

- [ ] **Step 7: Add report-side tests for new fields**

Append this class to `tests/test_report.py` before the `if __name__ == "__main__":` block:

```python
class DiagnosisReportRendersNewFieldsTests(unittest.TestCase):
    def test_diagnosis_json_includes_confidence_and_suggestions(self):
        result = make_eval_result(critical=True, final_status="failed")
        result.diagnosis.confidence = "high"
        result.diagnosis.repair_suggestions = ["Investigate the impl, not the tests."]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "r"
            write_eval_result(out_dir, result)
            data = json.loads((out_dir / "diagnosis.json").read_text())
            self.assertEqual(data["confidence"], "high")
            self.assertIn("Investigate", data["repair_suggestions"][0])

    def test_diagnosis_md_includes_confidence_section(self):
        result = make_eval_result(critical=True, final_status="failed")
        result.diagnosis.confidence = "high"
        result.diagnosis.repair_suggestions = ["Step A.", "Step B."]

        out = format_diagnosis_md(
            result.task, result.final_status, result.normalized_steps, result.diagnosis,
        )
        self.assertIn("Confidence", out)
        self.assertIn("high", out)
        self.assertIn("Step A.", out)
        self.assertIn("Step B.", out)
```

- [ ] **Step 8: Run to confirm failures**

```bash
python -m unittest tests.test_report.DiagnosisReportRendersNewFieldsTests -v 2>&1 | tail -10
```

Expected: failures because `diagnosis.json` doesn't yet include the new fields.

- [ ] **Step 9: Update `report.py` to render new fields**

Edit `report.py`. Replace the `_diagnosis_dict` function with:

```python
def _diagnosis_dict(diagnosis: Diagnosis) -> dict:
    return {
        'critical_step_id': diagnosis.critical_step.step_id if diagnosis.critical_step else None,
        'failure_stage': diagnosis.failure_stage,
        'error_type': diagnosis.error_type,
        'replay_branch_step': diagnosis.replay_branch_step,
        'replay_hint': diagnosis.replay_hint,
        'confidence': diagnosis.confidence,
        'repair_suggestions': list(diagnosis.repair_suggestions),
    }
```

In the same file, find `write_outputs` and replace the `diagnosis_data = {...}` block (the one that builds the dict for `diagnosis.json`) with:

```python
    diagnosis_data = {
        'critical_step_id': diagnosis.critical_step.step_id if diagnosis.critical_step else None,
        'failure_stage': diagnosis.failure_stage,
        'error_type': diagnosis.error_type,
        'replay_branch_step': diagnosis.replay_branch_step,
        'replay_hint': diagnosis.replay_hint,
        'confidence': diagnosis.confidence,
        'repair_suggestions': list(diagnosis.repair_suggestions),
    }
```

Then in `format_diagnosis_md`, find the `## Replay Suggestion` block (around lines 175–184) and replace it with:

```python
    # Replay suggestion + confidence + repair suggestions
    if diagnosis.replay_branch_step:
        lines.extend([
            "## Replay Suggestion",
            "",
            f"Branch at Step {diagnosis.replay_branch_step}",
            "",
            diagnosis.replay_hint,
            "",
        ])

    if diagnosis.critical_step:
        lines.extend([
            "## Confidence",
            diagnosis.confidence,
            "",
        ])

    if diagnosis.repair_suggestions:
        lines.append("## Repair Suggestions")
        lines.append("")
        for suggestion in diagnosis.repair_suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")
```

- [ ] **Step 10: Run report + full test suite**

```bash
python -m unittest tests.test_report -v 2>&1 | tail -20
python -m unittest discover -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 11: Smoke-test against the example fixture**

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/smoke && \
  cat out/smoke/diagnosis.json | python -c "import json, sys; d=json.load(sys.stdin); print('confidence:', d['confidence']); print('repair:', d['repair_suggestions'])"
```

Expected: confidence is one of `high|medium|low`; `repair_suggestions` is a non-empty list for a failed trajectory.

- [ ] **Step 12: Commit**

```bash
git add patterns.py models.py analyzer.py report.py tests/test_analyzer.py tests/test_report.py
git commit -m "feat: pattern registry, diagnosis confidence, and repair suggestions"
```

---

## Task 4: `--format json` for eval

**Files:**
- Modify: `main.py`
- Create test additions in: `tests/test_cli.py`

- [ ] **Step 1: Create `tests/test_cli.py`**

```python
"""Unit tests for main.py CLI behavior."""

import io
import json
import shutil
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m unittest tests.test_cli -v 2>&1 | tail -15
```

Expected: `AttributeError: 'Namespace' object has no attribute 'format'` (and one test asserting `confidence` may also fail).

- [ ] **Step 3: Add `--format` flag to `add_eval_args`**

Edit `main.py`. Inside `add_eval_args`, after the existing `--quiet` argument, add:

```python
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format for stdout summary (default: text)',
    )
```

- [ ] **Step 4: Update `run_eval_command` to support JSON mode**

Replace the existing `run_eval_command` function (lines 45–95) with:

```python
def run_eval_command(args: argparse.Namespace) -> int:
    try:
        inputs = discover_inputs(args.input)
        output_dir = Path(args.output)
        results = []

        json_mode = getattr(args, 'format', 'text') == 'json'
        quiet = getattr(args, 'quiet', False) or json_mode

        if not quiet:
            print(f"Evaluating {len(inputs)} Codex trajector{'y' if len(inputs) == 1 else 'ies'}...")

        for input_file in inputs:
            if not quiet:
                print(f"  - {input_file}")
            result = evaluate_file(input_file)
            results.append(result)

            if len(inputs) == 1:
                target_dir = output_dir
            else:
                target_dir = output_dir / input_file.stem
            write_eval_result(target_dir, result)

        summary = summarize_batch(results)
        write_batch_summary(output_dir, summary, results)

        if json_mode:
            import json
            payload = {
                "summary": {
                    "total": summary.total,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                    "high_risk": summary.high_risk,
                    "medium_risk": summary.medium_risk,
                    "low_risk": summary.low_risk,
                    "common_error_types": summary.common_error_types,
                },
                "results": [
                    {
                        "source_path": r.source_path,
                        "task": r.task,
                        "final_status": r.final_status,
                        "risk_level": r.metrics.risk_level,
                        "max_suspicious_score": r.metrics.max_suspicious_score,
                        "suspicious_steps": r.metrics.suspicious_steps,
                        "error_type": r.diagnosis.error_type,
                        "confidence": r.diagnosis.confidence,
                        "critical_step_id": (
                            r.diagnosis.critical_step.step_id
                            if r.diagnosis.critical_step else None
                        ),
                        "repair_suggestions": list(r.diagnosis.repair_suggestions),
                    }
                    for r in results
                ],
            }
            print(json.dumps(payload, indent=2))
        elif not quiet:
            print("\nEvaluation complete!")
            print(f"  Output: {output_dir}")
            print(f"  Total: {summary.total}")
            print(f"  Failed: {summary.failed}")
            print(
                f"  Risk: high={summary.high_risk}, "
                f"medium={summary.medium_risk}, low={summary.low_risk}"
            )

            if len(results) == 1 and results[0].diagnosis.critical_step:
                diagnosis = results[0].diagnosis
                print(f"\nCritical Step: {diagnosis.critical_step.step_id}")
                print(f"Error Type: {diagnosis.error_type}")

        if args.ci and any(
            result.final_status.lower() != 'success'
            or result.metrics.risk_level in ('medium', 'high')
            for result in results
        ):
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0
```

- [ ] **Step 5: Run CLI tests + full suite**

```bash
python -m unittest tests.test_cli -v 2>&1 | tail -15
python -m unittest discover -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 6: Smoke-test JSON output**

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/json_smoke --format json | python -c "import json, sys; print(list(json.load(sys.stdin).keys()))"
```

Expected: `['summary', 'results']`.

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "feat: add --format json output mode to eval command"
```

---

## Task 5: Engineering files

**Files:**
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Create `LICENSE` with MIT text**

Create `LICENSE`:

```
MIT License

Copyright (c) 2026 Agent Trajectory Eval Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Create `CONTRIBUTING.md`**

Create `CONTRIBUTING.md`:

```markdown
# Contributing

Thanks for your interest in improving Agent Trajectory Eval. The project is pure-Python (3.10+) and uses only the standard library at runtime — please keep new contributions inside that boundary unless there's a strong reason otherwise.

## Development setup

```bash
git clone <fork-url>
cd trace-agent
python -m unittest discover -v
```

There are no runtime dependencies. `requirements-dev.txt` documents optional tools used while developing.

## Running the CLI locally

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/example
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/example --format json
```

See `README.md` for the full command set.

## Running tests

```bash
python -m unittest discover -v
```

Single module:

```bash
python -m unittest tests.test_classifier -v
```

## Adding a new trajectory format adapter

1. Subclass `BaseAdapter` from `adapters/base.py`.
2. Implement `detect(data)` (cheap structural sniff) and `transform(data, source_path)` (returns a `Trajectory`).
3. Register the adapter in `adapters/__init__.py`'s `_ADAPTERS` list.
4. Add tests under `tests/` covering both `detect` and `transform`, plus an end-to-end `evaluate_file` test.

## Adding a new suspicious-step pattern

1. Add a `Pattern` entry to the `PATTERNS` registry in `patterns.py`. Each entry needs `name`, `description`, `score_weight`, `error_type`, and `repair_hint`.
2. Add a rule block to `score_suspicious_steps` in `analyzer.py` that calls `_apply(step, "<pattern_name>", "<reason>")` when the rule matches.
3. Add a test case in `tests/test_analyzer.py` that constructs the matching trajectory and asserts the score and reason are populated.

## Coding standards

- No third-party runtime dependencies.
- Stick to the dataclass model in `models.py` — extend it rather than passing dicts around.
- Prefer pure functions; only `analyzer.py` and `evaluator.py` orchestrate state across modules.
- All new public functions should have unit tests.
- Use Markdown tables in reports cautiously: route any user-supplied text through `_md_table_cell` in `report.py`.

## Commit style

We use short, conventional-style messages:

- `feat:` new behavior
- `fix:` bug fix
- `test:` test-only change
- `docs:` documentation
- `refactor:` no behavior change
```

- [ ] **Step 3: Create `requirements-dev.txt`**

Create `requirements-dev.txt`:

```
# Agent Trajectory Eval — dev dependencies
#
# Runtime: stdlib only (no dependencies).
# Tests:   unittest (stdlib).
#
# Optional tools used while developing:
#   - datasets   (for scripts/fetch_lcb.py and scripts/fetch_swe.py)
#
# Install with:
#   pip install -r requirements-dev.txt

datasets>=2.0
```

- [ ] **Step 4: Verify `unittest discover` still passes**

```bash
python -m unittest discover -v 2>&1 | tail -5
```

Expected: all tests pass (no behavior change from this task).

- [ ] **Step 5: Commit**

```bash
git add LICENSE CONTRIBUTING.md requirements-dev.txt
git commit -m "docs: add LICENSE, CONTRIBUTING.md, and requirements-dev.txt"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
python -m unittest discover -v 2>&1 | tail -20
```

Expected: all tests across all modules pass with no failures or errors.

- [ ] **Step 2: Smoke-test the example fixture end-to-end**

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/final_smoke && \
  python -c "import json; d=json.load(open('out/final_smoke/diagnosis.json')); assert d['confidence'] in ('low','medium','high'); assert isinstance(d['repair_suggestions'], list); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Smoke-test JSON CLI output**

```bash
python main.py eval --input examples/codex_failed_run_001.jsonl --output out/final_json --format json --quiet | python -c "import json, sys; d=json.load(sys.stdin); assert 'summary' in d and 'results' in d; print('JSON OK')"
```

Expected: `JSON OK`.

- [ ] **Step 4: Help text shows new flag**

```bash
python main.py eval --help | grep -- --format
```

Expected: line includes `--format {text,json}`.

- [ ] **Step 5: Final summary commit (if any leftover staged changes)**

```bash
git status
git add -u
git diff --cached --quiet || git commit -m "chore: roadmap completion final pass"
```
