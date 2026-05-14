# Judge-Eval Harness — Design

**Date:** 2026-05-14
**Status:** Approved

## Problem

trace-agent's classifier and analyzer are rule-based. There is a `judge=` hook on `normalize_steps` (`classifier.py:205`) but no way to know whether replacing it with an LLM judge would actually be better. We need measurement infrastructure before we add any LLM-backed judges.

## Goal

A small, reusable harness that scores any judge against a hand-labeled corpus and reports precision / recall / F1 on suspicious-step detection plus critical-step localization. Ships with the existing rule judge as the only included judge; users BYO LLM judges.

## Non-goals

- Shipping a reference LLM judge in this roadmap. Once we have baseline numbers we can decide what to build.
- Cost / latency tracking. Add when an LLM judge actually exists.
- CI gating on metric thresholds. Add once we know the rule judge's baseline.
- Provider-specific code anywhere in core. (Continues the principle from `2026-04-29-roadmap-design.en.md`.)

## Judge contract — two split hooks

```python
ClassifierJudge = Callable[[Step], NormalizedStep]                                      # already exists
ScorerJudge     = Callable[[list[NormalizedStep], str, str], list[NormalizedStep]]      # new — parallels score_suspicious_steps
```

A judge swaps either or both stages of the pipeline. Defaults are the existing rule implementations. The harness composes:

```
normalize_steps(steps, judge=classifier_judge)
  → score_suspicious_steps(normalized, task, status, judge=scorer_judge)
```

## Harness API

```python
@dataclass
class JudgeMetrics:
    suspicious_precision: float
    suspicious_recall:    float
    suspicious_f1:        float
    critical_hit_at_1:    float          # fraction of failed trajectories where top-scored step == labeled critical_step_id
    category_accuracy:    float | None   # over (labeled & predicted) suspicious steps; None if no overlap
    labeled_trajectories: int
    skipped_trajectories: int            # corpus entries with no label file
    per_trajectory:       list[dict]     # one row per trajectory for debugging

def evaluate_judges(
    labeled_corpus_dir: Path,
    *,
    classifier_judge: ClassifierJudge | None = None,
    scorer_judge:     ScorerJudge     | None = None,
) -> JudgeMetrics: ...
```

The corpus dir contains `*.labels.json` sidecar files; each file points at a trajectory by relative path. The harness loads each trajectory via the existing `parser.load_trajectory()`, runs the pipeline (composing the two judges), then compares to labels.

## Label format

Sidecar JSON, one file per labeled trajectory, stored under `tests/fixtures/labels/<trajectory_basename>.labels.json`:

```json
{
  "trajectory": "examples/codex_failed_run_001.jsonl",
  "labeler": "human",
  "labeled_at": "2026-05-14",
  "final_status": "failed",
  "critical_step_id": 5,
  "steps": [
    {"step_id": 1, "suspicious": false},
    {"step_id": 5, "suspicious": true, "category": "test_edit_after_impl_failure"}
  ]
}
```

Rules:
- Steps not listed default to `suspicious: false` — labelers only write the interesting rows.
- `category`, when present, MUST be a key in the `PATTERNS` registry from `patterns.py`. The label loader validates this and rejects unknown categories.
- `trajectory` is resolved relative to the repo root.

## Metrics

- **Suspicious-detection**: a step is "predicted suspicious" iff `suspicious_score > 0`. Compute precision, recall, F1 across all labeled steps in the corpus (micro-averaged).
- **Critical-step `hit@1`**: among trajectories whose label has `final_status == "failed"` and a `critical_step_id`, count how often the judge's top-scored step's `step_id` matches.
- **Category accuracy**: among steps where the human labeled `suspicious: true` AND the judge predicted suspicious AND both have a category, compute exact-match accuracy. Mapping from a judge's `suspicious_reasons` to a `PATTERNS` key reuses `analyzer._patterns_matched` (which already does this for `Diagnosis`).
- **Coverage**: `labeled_trajectories` and `skipped_trajectories` make it obvious when the corpus is too thin.

## CLI surface

```
trace-agent judge-eval --labels tests/fixtures/labels/ [--judge rule] [--format text|json]
```

- `--judge rule` is the default and currently the only choice. The flag exists so adding `--judge llm:foo` later is a non-breaking change.
- `--format` mirrors the existing flag on `eval` (`main.py:45`).
- Exit 0 on success, 2 on error. No CI gating in v1.

## Seed corpus

5 hand-labeled trajectories — cheap to produce, enough to detect regressions:

1. `examples/codex_failed_run_001.jsonl` — the existing test-manipulation fixture (failed).
2. One pass + one fail from `data/lcb/trajectories/`.
3. One pass + one fail from `data/swe/`.

## File map

| File | Change |
|---|---|
| `judge_eval.py` | **Create** — `JudgeMetrics`, `evaluate_judges`, label loader, default rule pipeline composer |
| `analyzer.py` | **Modify** — add `judge: ScorerJudge \| None = None` param to `score_suspicious_steps`; default behavior unchanged |
| `main.py` | **Modify** — register `judge-eval` subcommand; reuse `--format` text/json wiring |
| `tests/fixtures/labels/*.labels.json` | **Create** — 5 hand-labeled trajectories |
| `tests/test_judge_eval.py` | **Create** — label loader (incl. unknown-category rejection), metric math (synthetic predictions vs labels), end-to-end harness on a tiny fixture, scorer-judge hook contract |

## Verification

- `python -m unittest discover` — all existing 141 + new tests pass.
- `python main.py judge-eval --labels tests/fixtures/labels/` — emits a Markdown report; rule judge baseline numbers are non-zero for both precision and recall.
- `python main.py judge-eval --labels tests/fixtures/labels/ --format json | jq .` — valid JSON with all `JudgeMetrics` fields.
