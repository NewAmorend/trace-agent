"""Data models for trajectory analysis."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    """Raw step from trajectory JSON."""
    step_id: int
    event_id: Optional[str] = None
    thought: Optional[str] = None
    action: str = ""
    observation: Optional[str] = None
    diff: Optional[str] = None
    item_type: str = ""
    exit_code: Optional[int] = None
    status: Optional[str] = None


@dataclass
class NormalizedStep:
    """Step with additional classification fields."""
    step_id: int
    event_id: Optional[str]
    thought: Optional[str]
    action: str
    observation: Optional[str]
    diff: Optional[str]
    item_type: str
    exit_code: Optional[int]
    status: Optional[str]
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


@dataclass
class Trajectory:
    """A normalized trajectory loaded from a Codex session or exec stream."""
    source_path: str
    task: str
    final_status: str
    steps: list[Step]
    thread_id: Optional[str] = None
    failure_message: Optional[str] = None


@dataclass
class EvalMetrics:
    """Aggregate metrics derived from a trajectory."""
    total_steps: int = 0
    command_steps: int = 0
    file_change_steps: int = 0
    test_steps: int = 0
    failed_test_steps: int = 0
    state_changes: int = 0
    suspicious_steps: int = 0
    max_suspicious_score: float = 0.0
    risk_level: str = "low"


@dataclass
class EvalResult:
    """Full evaluation result for one trajectory."""
    source_path: str
    task: str
    final_status: str
    normalized_steps: list[NormalizedStep]
    tree_md: str
    diagnosis: Diagnosis
    metrics: EvalMetrics
    failure_message: Optional[str] = None


@dataclass
class BatchSummary:
    """Summary for a directory of evaluated trajectories."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    high_risk: int = 0
    medium_risk: int = 0
    low_risk: int = 0
    common_error_types: dict[str, int] = field(default_factory=dict)
