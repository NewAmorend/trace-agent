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
    matched_pattern_names: list[str] | None = None,
    thought: str | None = None,
    event_id: str | None = None,
    exit_code: int | None = None,
    status: str | None = None,
) -> NormalizedStep:
    step = NormalizedStep(
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
    if matched_pattern_names is not None:
        step._matched_pattern_names = list(matched_pattern_names)
    return step
