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

from analyzer import ScorerJudge, _patterns_matched
from models import NormalizedStep, Step
from patterns import PATTERNS

ClassifierJudge = Callable[[Step], NormalizedStep]


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
