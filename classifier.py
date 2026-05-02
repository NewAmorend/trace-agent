"""Classification logic for steps."""

import re
from models import Step, NormalizedStep


def classify_action_type(action: str, diff: str | None) -> str:
    """Classify action type based on command and diff."""
    action_lower = action.lower()

    # Check for test commands
    test_patterns = [
        r'\bpytest\b',
        r'\bnpm\s+test\b',
        r'\byarn\s+test\b',
        r'\bcargo\s+test\b',
        r'\bgo\s+test\b',
        r'\bmvn\s+test\b',
        r'\bgradle\s+test\b',
        r'\bunittest\b',
        r'\bpython\s+-m\s+pytest\b',
    ]
    if any(re.search(pattern, action_lower) for pattern in test_patterns):
        return 'run_test'

    # Check for search commands
    search_patterns = [
        r'\brg\b',
        r'\bgrep\b',
        r'\bfind\b',
        r'\bweb_search\b',
        r'\bglob\s*\{',       # Claude Glob tool
        r'\bwebsearch\s*\{',  # Claude WebSearch tool
    ]
    if any(re.search(pattern, action_lower) for pattern in search_patterns):
        return 'search'

    # Check for inspection commands
    inspect_patterns = [
        r'\bcat\b',
        r'\bsed\b',
        r'\bnl\b',
        r'\bhead\b',
        r'\btail\b',
        r'\blesser\b',
        r'\bless\b',
        r'\bread\s*\{',      # Claude Read tool
        r'\bwebfetch\s*\{',  # Claude WebFetch tool
    ]
    if any(re.search(pattern, action_lower) for pattern in inspect_patterns):
        return 'inspect_file'

    # Check for environment change commands
    env_patterns = [
        r'\bpip\s+install\b',
        r'\bnpm\s+install\b',
        r'\byarn\s+add\b',
        r'\bapt\s+install\b',
        r'\bconda\s+install\b',
        r'\bbrew\s+install\b',
    ]
    if any(re.search(pattern, action_lower) for pattern in env_patterns):
        return 'env_change'

    # Check for git actions
    git_patterns = [
        r'\bgit\s+status\b',
        r'\bgit\s+diff\b',
        r'\bgit\s+checkout\b',
        r'\bgit\s+reset\b',
        r'\bgit\s+apply\b',
    ]
    if any(re.search(pattern, action_lower) for pattern in git_patterns):
        return 'git_action'

    # Check for edit operations
    if diff and diff.strip():
        return 'edit_file'

    edit_patterns = [
        r'\bapply_patch\b',
        r'\bwrite\s+file\b',
        r'\bappend\s+to\b',
    ]
    if any(re.search(pattern, action_lower) for pattern in edit_patterns):
        return 'edit_file'

    return 'other'


def classify_stage(action: str, action_type: str, observation: str | None) -> str:
    """Classify the stage of a step."""
    action_lower = action.lower()
    obs_lower = (observation or "").lower()

    # Environment verification
    env_verify_patterns = [
        r'\bpython\s+--version\b',
        r'\bnode\s+--version\b',
        r'\bpip\s+--version\b',
        r'\bnpm\s+--version\b',
        r'\bcargo\s+--version\b',
        r'\bgo\s+version\b',
    ]
    if any(re.search(pattern, action_lower) for pattern in env_verify_patterns):
        return 'environment verification'

    # Direct mapping from action types
    if action_type == 'env_change':
        return 'dependency installation'
    if action_type == 'edit_file':
        return 'patching'
    if action_type == 'run_test':
        return 'verification'
    if action_type in ['search', 'inspect_file']:
        return 'inspection/debugging'

    # Check observation for stage clues
    if 'error' in obs_lower or 'exception' in obs_lower:
        return 'inspection/debugging'

    return 'other'


def is_state_changing(action_type: str, action: str, diff: str | None) -> bool:
    """Determine if a step changes the system state."""
    action_lower = action.lower()

    if action_type == 'edit_file':
        return True
    if action_type == 'env_change':
        return True

    # Specific git operations
    if action_type == 'git_action':
        if 'git checkout' in action_lower:
            return True
        if 'git reset' in action_lower:
            return True
        if 'git apply' in action_lower:
            return True

    return False


def normalize_steps(steps: list[Step]) -> list[NormalizedStep]:
    """Convert Steps to NormalizedSteps with classification."""
    normalized = []
    for step in steps:
        action_type = classify_action_type(step.action, step.diff)
        stage = classify_stage(step.action, action_type, step.observation)
        state_change = is_state_changing(action_type, step.action, step.diff)

        norm_step = NormalizedStep(
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
            suspicious_reasons=[]
        )
        normalized.append(norm_step)

    return normalized
