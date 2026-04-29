"""Shared predicates for interpreting test command output."""

import re


_POSITIVE_FAILURE_COUNT = re.compile(
    r"\b[1-9]\d*\s+(?:failed|failures?|errors?|errored)\b"
)
_ZERO_FAILURE_COUNT = re.compile(r"\b0\s+(?:failed|failures?|errors?)\b")
_NEGATED_FAILURE = re.compile(r"\b(?:no failures|no errors)\b")
_GENERIC_FAILURE = re.compile(r"\b(?:failed|failure|error|exception|traceback)\b")
_SUCCESS_ONLY = re.compile(r"\b(?:all passed|passed|no failures|no errors)\b")


def looks_like_test_failure(observation: str | None) -> bool:
    """Return True when test output indicates an actual failure or error."""
    text = (observation or "").lower()
    if not text:
        return False

    if _POSITIVE_FAILURE_COUNT.search(text):
        return True

    if not _GENERIC_FAILURE.search(text):
        return False

    has_zero_count = bool(_ZERO_FAILURE_COUNT.search(text))
    has_negated_failure = bool(_NEGATED_FAILURE.search(text))
    cleaned = _ZERO_FAILURE_COUNT.sub("", text)
    cleaned = _NEGATED_FAILURE.sub("", cleaned)
    if _GENERIC_FAILURE.search(cleaned):
        return True

    if has_zero_count or has_negated_failure:
        return False

    return not _SUCCESS_ONLY.search(text)


def looks_like_test_success(observation: str | None) -> bool:
    """Return True when test output looks successful and has no failures."""
    text = (observation or "").lower()
    if looks_like_test_failure(text):
        return False
    return bool(_SUCCESS_ONLY.search(text))
