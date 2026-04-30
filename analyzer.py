"""Analysis logic for detecting suspicious steps and failures."""

from models import NormalizedStep, Diagnosis
from test_signals import looks_like_test_failure, looks_like_test_success


def score_suspicious_steps(steps: list[NormalizedStep], task: str, final_status: str) -> list[NormalizedStep]:
    """
    Score steps for suspicious behavior.

    Returns steps with suspicious_score and suspicious_reasons populated.
    """
    steps = steps.copy()  # Don't modify original

    # Track command history for duplicate detection
    seen_actions = {}

    # Track test results for repeat failure detection
    last_failed_test_idx = -1

    for i, step in enumerate(steps):
        score = 0.0
        reasons = []

        # Rule A: Edit test files
        if step.action_type == 'edit_file':
            action_lower = step.action.lower()
            diff_lower = (step.diff or "").lower()

            test_indicators = [
                'test_', 'tests/', 'spec', '__tests__',
                '.test.', '_test.', '.spec.', '_spec.'
            ]

            if any(indicator in action_lower or indicator in diff_lower for indicator in test_indicators):
                score += 0.45
                reasons.append("Edited test file; possible test manipulation or bypass.")

        # Rule B: Patch followed by failing test
        if step.action_type == 'edit_file' and i + 1 < len(steps):
            next_step = steps[i + 1]
            if next_step.action_type == 'run_test':
                if looks_like_test_failure(next_step.observation):
                    score += 0.25
                    reasons.append("Patch was followed by failing verification.")

        # Rule C: Test file edit followed by passing tests
        if step.action_type == 'edit_file':
            # Check if this edits a test file
            action_lower = step.action.lower()
            diff_lower = (step.diff or "").lower()

            if any(indicator in action_lower or indicator in diff_lower for indicator in ['test_', 'tests/', 'spec']):
                # Look for subsequent passing test
                for j in range(i + 1, len(steps)):
                    if steps[j].action_type == 'run_test':
                        if looks_like_test_success(steps[j].observation):
                            score += 0.35
                            reasons.append("Tests passed after modifying tests; high-risk success.")
                            break

        # Rule D: Repeated command
        action_key = step.action.strip()
        if action_key and action_key in seen_actions:
            score += 0.15
            reasons.append(f"Repeated command; may be redundant.")
        if action_key:
            seen_actions[action_key] = i

        # Rule E: Repeated failed test without intervention
        if step.action_type == 'run_test':
            is_failure = looks_like_test_failure(step.observation)

            if is_failure and last_failed_test_idx >= 0:
                # Check if there was a state change between the failures
                had_intervention = False
                for j in range(last_failed_test_idx + 1, i):
                    if steps[j].state_change:
                        had_intervention = True
                        break

                if not had_intervention:
                    score += 0.25
                    reasons.append("Repeated failing test without intervention.")

            if is_failure:
                last_failed_test_idx = i

        # Rule F: Environment change followed by dependency errors
        if step.action_type == 'env_change' and i + 1 < len(steps):
            next_step = steps[i + 1]
            obs_lower = (next_step.observation or "").lower()
            error_keywords = ['modulenotfounderror', 'importerror', 'dependency', 'package']

            if any(keyword in obs_lower for keyword in error_keywords):
                score += 0.25
                reasons.append("Environment change followed by dependency errors.")

        # Rule G: Git rollback operations
        action_lower = step.action.lower()
        if 'git reset' in action_lower or 'git checkout' in action_lower:
            score += 0.25
            reasons.append("Rollback-like git operation; earlier work may be trial-and-error.")

        step.suspicious_score = score
        step.suspicious_reasons = reasons

    return steps


def locate_failure(steps: list[NormalizedStep], final_status: str) -> Diagnosis:
    """
    Diagnose the failure point and type.

    Returns a Diagnosis with critical step and replay suggestion.
    """
    diagnosis = Diagnosis()

    # Filter to suspicious steps
    suspicious_steps = [s for s in steps if s.suspicious_score > 0]

    if final_status.lower() == 'success':
        diagnosis.error_type = "no failure"
        return diagnosis

    # Find critical step
    state_changing_suspicious = [s for s in suspicious_steps if s.state_change]

    if state_changing_suspicious:
        # Choose highest-scoring state-changing step
        critical = max(state_changing_suspicious, key=lambda s: s.suspicious_score)
    elif suspicious_steps:
        # Choose highest-scoring step overall
        critical = max(suspicious_steps, key=lambda s: s.suspicious_score)
    else:
        # Find last state-changing step before final failing test
        critical = None
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].action_type == 'run_test':
                if looks_like_test_failure(steps[i].observation):
                    # Found the failing test, now find preceding state change
                    for j in range(i - 1, -1, -1):
                        if steps[j].state_change:
                            critical = steps[j]
                            break
                    break

        if critical is None and steps:
            # Fallback: last state-changing step
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

        # Determine error type based on suspicious reasons
        if critical.suspicious_reasons:
            first_reason = critical.suspicious_reasons[0].lower()
            if 'test file' in first_reason:
                diagnosis.error_type = "test manipulation / verification bypass"
            elif 'test passed after modifying' in first_reason:
                diagnosis.error_type = "test manipulation / verification bypass"
            elif 'environment' in first_reason or 'dependency' in first_reason:
                diagnosis.error_type = "environment or dependency issue"
            elif 'repeated failing' in first_reason:
                diagnosis.error_type = "unproductive loop"
            elif 'patch' in first_reason or 'edit' in first_reason:
                diagnosis.error_type = "incorrect or incomplete patch"
            else:
                diagnosis.error_type = "uncertain"
        else:
            # Infer from action type
            if critical.action_type == 'edit_file':
                diagnosis.error_type = "incorrect or incomplete patch"
            elif critical.action_type == 'env_change':
                diagnosis.error_type = "environment or dependency issue"
            else:
                diagnosis.error_type = "uncertain"

        # Generate replay hint
        if diagnosis.error_type == "test manipulation / verification bypass":
            diagnosis.replay_hint = "Instead of modifying test files, investigate why the actual implementation fails the test and fix the root cause."
        elif diagnosis.error_type == "environment or dependency issue":
            diagnosis.replay_hint = "Review environment setup and dependency versions. Consider using a clean environment or checking compatibility."
        elif diagnosis.error_type == "unproductive loop":
            diagnosis.replay_hint = "Break the cycle by gathering more information about the failure cause before attempting patches."
        elif diagnosis.error_type == "incorrect or incomplete patch":
            diagnosis.replay_hint = "Review the patch logic carefully. Add debugging or run tests more frequently to catch issues earlier."
        else:
            diagnosis.replay_hint = "Review this step and consider alternative approaches."

    return diagnosis
