"""Classification logic for steps."""

import re
from typing import Callable
from models import Step, NormalizedStep

_TEST_PATTERNS = [
    re.compile(r'\bpytest\b'),
    re.compile(r'\bnpm\s+test\b'),
    re.compile(r'\byarn\s+test\b'),
    re.compile(r'\bcargo\s+test\b'),
    re.compile(r'\bgo\s+test\b'),
    re.compile(r'\bmvn\s+test\b'),
    re.compile(r'\bgradle\s+test\b'),
    re.compile(r'\bunittest\b'),
    re.compile(r'\bpython\s+-m\s+pytest\b'),
]

_SEARCH_PATTERNS = [
    re.compile(r'\brg\b'),
    re.compile(r'\bgrep\b'),
    re.compile(r'\bfind\b'),
    re.compile(r'\bweb_search\b'),
    re.compile(r'\bglob\s*\{'),
    re.compile(r'\bwebsearch\s*\{'),
]

_INSPECT_PATTERNS = [
    re.compile(r'\bcat\b'),
    re.compile(r'\bsed\b'),
    re.compile(r'\bnl\b'),
    re.compile(r'\bhead\b'),
    re.compile(r'\btail\b'),
    re.compile(r'\blesser\b'),
    re.compile(r'\bless\b'),
    re.compile(r'\bread\s*\{'),
    re.compile(r'\bwebfetch\s*\{'),
]

_ENV_PATTERNS = [
    re.compile(r'\bpip\s+install\b'),
    re.compile(r'\bnpm\s+install\b'),
    re.compile(r'\byarn\s+add\b'),
    re.compile(r'\bapt\s+install\b'),
    re.compile(r'\bconda\s+install\b'),
    re.compile(r'\bbrew\s+install\b'),
]

_GIT_PATTERNS = [
    re.compile(r'\bgit\s+status\b'),
    re.compile(r'\bgit\s+diff\b'),
    re.compile(r'\bgit\s+checkout\b'),
    re.compile(r'\bgit\s+reset\b'),
    re.compile(r'\bgit\s+apply\b'),
]

_EDIT_PATTERNS = [
    re.compile(r'\bapply_patch\b'),
    re.compile(r'\bwrite\s+file\b'),
    re.compile(r'\bappend\s+to\b'),
]

_ENV_VERIFY_PATTERNS = [
    re.compile(r'\bpython\s+--version\b'),
    re.compile(r'\bnode\s+--version\b'),
    re.compile(r'\bpip\s+--version\b'),
    re.compile(r'\bnpm\s+--version\b'),
    re.compile(r'\bcargo\s+--version\b'),
    re.compile(r'\bgo\s+version\b'),
]


def classify_action_type(action: str, diff: str | None, item_type: str = "") -> str:
    """Classify action type based on command and diff."""
    if item_type == "middleware":
        return "middleware_action"
    if item_type == "tool_call":
        return _classify_tool_call(action)
    if item_type == "tool_result":
        return "tool_result"
    if item_type == "error":
        return "error"

    action_lower = action.lower()

    if any(p.search(action_lower) for p in _TEST_PATTERNS):
        return 'run_test'

    if any(p.search(action_lower) for p in _SEARCH_PATTERNS):
        return 'search'

    if any(p.search(action_lower) for p in _INSPECT_PATTERNS):
        return 'inspect_file'

    if any(p.search(action_lower) for p in _ENV_PATTERNS):
        return 'env_change'

    if any(p.search(action_lower) for p in _GIT_PATTERNS):
        return 'git_action'

    if diff and diff.strip():
        return 'edit_file'

    if any(p.search(action_lower) for p in _EDIT_PATTERNS):
        return 'edit_file'

    return 'other'


_TOOL_CALL_MAP = {
    "bash": "run_test" if True else "other",
    "shell": "other",
}

_TEST_TOOL_NAMES = frozenset({"pytest", "run_tests", "test_runner", "npm_test"})
_EDIT_TOOL_NAMES = frozenset({"apply_patch", "write_file", "edit_file", "str_replace_editor", "create_file", "insert_content", "replace_in_file"})
_SEARCH_TOOL_NAMES = frozenset({"search", "grep", "find", "glob", "web_search", "ripgrep"})
_INSPECT_TOOL_NAMES = frozenset({"cat", "read_file", "view_file", "read", "head", "tail"})
_ENV_TOOL_NAMES = frozenset({"pip_install", "npm_install", "install_package"})


def _classify_tool_call(action: str) -> str:
    tool_name = action.split()[0].lower() if action else ""
    if tool_name in _TEST_TOOL_NAMES:
        return "run_test"
    if tool_name in _EDIT_TOOL_NAMES:
        return "edit_file"
    if tool_name in _SEARCH_TOOL_NAMES:
        return "search"
    if tool_name in _INSPECT_TOOL_NAMES:
        return "inspect_file"
    if tool_name in _ENV_TOOL_NAMES:
        return "env_change"
    if "test" in tool_name or "pytest" in tool_name:
        return "run_test"
    if "patch" in tool_name or "edit" in tool_name or "write" in tool_name:
        return "edit_file"
    if "search" in tool_name or "grep" in tool_name or "find" in tool_name:
        return "search"
    if "bash" in tool_name or "shell" in tool_name or "execute" in tool_name:
        return "other"
    return "tool_call"


def classify_stage(action: str, action_type: str, observation: str | None, item_type: str = "") -> str:
    """Classify the stage of a step."""
    action_lower = action.lower()
    obs_lower = (observation or "").lower()

    if item_type == "middleware":
        return "middleware"

    if any(p.search(action_lower) for p in _ENV_VERIFY_PATTERNS):
        return 'environment verification'

    if action_type == 'env_change':
        return 'dependency installation'
    if action_type == 'edit_file':
        return 'patching'
    if action_type == 'run_test':
        return 'verification'
    if action_type in ['search', 'inspect_file', 'tool_call']:
        return 'inspection/debugging'
    if action_type == 'middleware_action':
        return 'middleware'
    if action_type == 'tool_result':
        return 'inspection/debugging'
    if action_type == 'error':
        return 'error'

    if 'error' in obs_lower or 'exception' in obs_lower:
        return 'inspection/debugging'

    return 'other'


def is_state_changing(action_type: str, action: str, diff: str | None, item_type: str = "") -> bool:
    """Determine if a step changes the system state."""
    if item_type == "middleware":
        action_lower = action.lower()
        if "summarize" in action_lower or "compaction" in action_lower:
            return True
        return False

    action_lower = action.lower()

    if action_type == 'edit_file':
        return True
    if action_type == 'env_change':
        return True

    if action_type == 'git_action':
        if 'git checkout' in action_lower:
            return True
        if 'git reset' in action_lower:
            return True
        if 'git apply' in action_lower:
            return True

    return False


def normalize_steps(
    steps: list[Step],
    judge: Callable[[Step], NormalizedStep] | None = None,
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
    action_type = classify_action_type(step.action, step.diff, step.item_type)
    stage = classify_stage(step.action, action_type, step.observation, step.item_type)
    state_change = is_state_changing(action_type, step.action, step.diff, step.item_type)

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
        matched_pattern_names=[],
        extra=step.extra,
    )
