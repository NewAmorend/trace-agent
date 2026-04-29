"""Data models for trajectory analysis."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    """Raw step from trajectory JSON."""
    step_id: int
    thought: Optional[str] = None
    action: str = ""
    observation: Optional[str] = None
    diff: Optional[str] = None


@dataclass
class NormalizedStep:
    """Step with additional classification fields."""
    step_id: int
    thought: Optional[str]
    action: str
    observation: Optional[str]
    diff: Optional[str]
    action_type: str
    stage: str
    state_change: bool
    suspicious_score: float = 0.0
    suspicious_reasons: list[str] = field(default_factory=list)


@dataclass
class TraceNode:
    """Node in trace tree."""
    state_id: int
    steps: list[NormalizedStep] = field(default_factory=list)
    children: list['TraceNode'] = field(default_factory=list)


@dataclass
class Diagnosis:
    """Diagnosis of trajectory failure."""
    critical_step: Optional[NormalizedStep] = None
    failure_stage: str = ""
    error_type: str = ""
    replay_branch_step: Optional[int] = None
    replay_hint: str = ""
