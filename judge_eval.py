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

from analyzer import ScorerJudge, _patterns_matched, score_suspicious_steps
from classifier import normalize_steps
from parser import load_trajectory
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


def evaluate_judges(
    labeled_corpus_dir: Path,
    *,
    classifier_judge: ClassifierJudge | None = None,
    scorer_judge: ScorerJudge | None = None,
) -> JudgeMetrics:
    """Evaluate the (classifier + scorer) judge pipeline against a labeled corpus.

    For each *.labels.json file under `labeled_corpus_dir`:
      1. Load the referenced trajectory via parser.load_trajectory.
      2. Run normalize_steps (with classifier_judge) -> score_suspicious_steps
         (with scorer_judge).
      3. Compare predictions to labels and accumulate counts.

    Trajectories whose file is missing are counted in skipped_trajectories.
    """
    labels = discover_labels(labeled_corpus_dir)

    tp = fp = fn = 0
    critical_hits = 0
    critical_eligible = 0
    category_correct = 0
    category_eligible = 0
    skipped = 0
    per_trajectory: list[dict] = []

    for labeled in labels:
        try:
            trajectory = load_trajectory(str(labeled.trajectory_path))
        except FileNotFoundError:
            skipped += 1
            per_trajectory.append({
                "trajectory": str(labeled.trajectory_path),
                "skipped": True,
            })
            continue

        normalized = normalize_steps(trajectory.steps, judge=classifier_judge)
        scored = score_suspicious_steps(
            normalized, trajectory.task, trajectory.final_status, judge=scorer_judge,
        )

        traj_tp = traj_fp = traj_fn = 0
        for step in scored:
            predicted = step.suspicious_score > 0
            actual = labeled.is_suspicious(step.step_id)
            if predicted and actual:
                traj_tp += 1
                actual_cat = labeled.category_for(step.step_id)
                predicted_cat = _category_for_prediction(step)
                if actual_cat is not None and predicted_cat is not None:
                    category_eligible += 1
                    if actual_cat == predicted_cat:
                        category_correct += 1
            elif predicted and not actual:
                traj_fp += 1
            elif not predicted and actual:
                traj_fn += 1

        tp += traj_tp
        fp += traj_fp
        fn += traj_fn

        traj_critical_hit: bool | None = None
        if labeled.final_status == "failed" and labeled.critical_step_id is not None:
            critical_eligible += 1
            top = max(scored, key=lambda s: s.suspicious_score, default=None)
            hit = (
                top is not None
                and top.suspicious_score > 0
                and top.step_id == labeled.critical_step_id
            )
            traj_critical_hit = hit
            if hit:
                critical_hits += 1

        per_trajectory.append({
            "trajectory": str(labeled.trajectory_path),
            "tp": traj_tp,
            "fp": traj_fp,
            "fn": traj_fn,
            "critical_hit": traj_critical_hit,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(precision, recall)
    hit_at_1 = critical_hits / critical_eligible if critical_eligible > 0 else 0.0
    cat_acc = category_correct / category_eligible if category_eligible > 0 else None

    return JudgeMetrics(
        suspicious_precision=precision,
        suspicious_recall=recall,
        suspicious_f1=f1,
        critical_hit_at_1=hit_at_1,
        category_accuracy=cat_acc,
        labeled_trajectories=len(labels) - skipped,
        skipped_trajectories=skipped,
        per_trajectory=per_trajectory,
    )


def format_metrics_md(metrics: JudgeMetrics) -> str:
    if metrics.category_accuracy is None:
        cat_line = "Category accuracy: n/a (no overlap)"
    else:
        cat_line = f"Category accuracy: {metrics.category_accuracy:.3f}"

    return "\n".join([
        "# Judge Eval",
        "",
        f"Labeled trajectories: {metrics.labeled_trajectories}",
        f"Skipped (no trajectory file): {metrics.skipped_trajectories}",
        "",
        "## Suspicious-step detection",
        f"Precision: {metrics.suspicious_precision:.3f}",
        f"Recall:    {metrics.suspicious_recall:.3f}",
        f"F1:        {metrics.suspicious_f1:.3f}",
        "",
        "## Critical-step localization",
        f"hit@1: {metrics.critical_hit_at_1:.3f}",
        "",
        f"## {cat_line}",
        "",
    ])


def metrics_to_dict(metrics: JudgeMetrics) -> dict:
    return {
        "suspicious_precision": metrics.suspicious_precision,
        "suspicious_recall": metrics.suspicious_recall,
        "suspicious_f1": metrics.suspicious_f1,
        "critical_hit_at_1": metrics.critical_hit_at_1,
        "category_accuracy": metrics.category_accuracy,
        "labeled_trajectories": metrics.labeled_trajectories,
        "skipped_trajectories": metrics.skipped_trajectories,
        "per_trajectory": metrics.per_trajectory,
    }
