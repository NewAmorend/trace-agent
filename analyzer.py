"""Analysis logic for detecting suspicious steps and failures."""

import copy

from models import Diagnosis, NormalizedStep
from patterns import PATTERNS, Pattern, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
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
    step.matched_pattern_names.append(pattern_name)


def score_suspicious_steps(steps: list[NormalizedStep], task: str, final_status: str) -> list[NormalizedStep]:
    """
    Score steps for suspicious behavior.

    Returns steps with suspicious_score and suspicious_reasons populated.
    """
    steps = copy.deepcopy(steps)

    has_impl_edit = any(
        s.action_type == 'edit_file' and not _is_test_path(s.action, s.diff or "")
        for s in steps
    )

    seen_actions: dict[str, int] = {}
    last_failed_test_idx = -1

    for i, step in enumerate(steps):
        step.suspicious_score = 0.0
        step.suspicious_reasons = []
        step.matched_pattern_names = []

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

        # Rule D: Repeated command (excluding test runs — re-running tests is expected)
        action_key = step.action.strip()
        if action_key and action_key in seen_actions and step.action_type != 'run_test':
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
    if score >= RISK_THRESHOLD_HIGH:
        return "high"
    if score >= RISK_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


def _patterns_matched(step: NormalizedStep) -> list[Pattern]:
    """Return the patterns that fired on this step, in order, deduped."""
    seen: set[str] = set()
    result: list[Pattern] = []
    for name in step.matched_pattern_names:
        if name in seen:
            continue
        seen.add(name)
        result.append(PATTERNS[name])
    return result


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

        matched = _patterns_matched(critical)
        if matched:
            diagnosis.error_type = matched[0].error_type
            diagnosis.replay_hint = matched[0].repair_hint
            diagnosis.repair_suggestions = [p.repair_hint for p in matched]
        else:
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
