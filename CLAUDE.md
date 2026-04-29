# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Trajectory Eval is a Codex-first Python CLI tool for evaluating coding agent trajectories. It parses Codex JSONL session or `codex exec --json` streams and produces normalized steps, trace trees, risk metrics, suspicious step detection, failure diagnosis reports, and batch summaries. Pure Python standard library — no runtime dependencies.

## Commands

```bash
# Run evaluation (Codex JSONL format)
python main.py --input examples/codex_failed_run_001.jsonl --output out/codex_failed_run_001

# Batch evaluation
python main.py --input data/lcb/trajectories --output out/lcb_eval

# CI mode: returns 1 for failed or medium/high-risk trajectories
python main.py --input examples/codex_failed_run_001.jsonl --output out/codex_failed_run_001 --ci

# Verify outputs
cat out/failed_run_001/trace_tree.md
cat out/failed_run_001/diagnosis.md

# Run tests
python -m unittest discover
```

### LiveCodeBench Testing

Use LCB problems to generate real Codex trajectories for testing:

```bash
# 1. Fetch curated problems (requires: pip install datasets)
python scripts/fetch_lcb.py

# 2. Run Codex CLI against problems (requires: codex CLI installed)
python scripts/run_lcb.py                        # all problems
python scripts/run_lcb.py --difficulty easy       # easy only
python scripts/run_lcb.py --difficulty hard --limit 1  # just one hard problem

# 3. Analyze generated trajectories
python main.py --input data/lcb/trajectories/<file>.jsonl --output out/lcb_test
```

Validation is done with `python -m unittest discover` and the Codex fixture:
- Trace tree shows state transitions: State 0 -> State 1 -> State 2
- Step 5 is flagged as suspicious (test file manipulation)
- Diagnosis identifies the correct critical step and error type

## Architecture

Entry point is `main.py`. Core modules are flat in the project root. Adapter modules live in `adapters/`.

**Data flow**: `main.py` calls modules in sequence:

```
parser.py → evaluator.py → classifier.py → tree.py → analyzer.py → report.py
```

All modules operate on dataclasses from `models.py`: `Trajectory` → `Step` → `NormalizedStep` (enriched with classifications) → `TraceNode` (tree structure) → `Diagnosis` and `EvalResult`.

### Module Responsibilities

- **models.py**: Dataclasses — `Step`, `NormalizedStep`, `TraceNode`, `Diagnosis`
- **parser.py**: `load_trajectory()` — Codex JSONL loading, adapter-based format detection, delegates to registered adapters
- **evaluator.py**: `evaluate_file()` + `summarize_batch()` — single and directory evaluation orchestration
- **adapters/**: Format adapter layer
  - **base.py**: `BaseAdapter` ABC with `detect()` and `transform()` methods
  - **codex_adapter.py**: `CodexAdapter` — OpenAI Codex CLI JSONL session format
  - **__init__.py**: Registry — `get_adapter()`, `register_adapter()`
- **classifier.py**: `normalize_steps()` — classifies each step's action type, stage, and whether it's state-changing
- **tree.py**: `build_trace_tree()` + `render_trace_tree()` — groups steps into states; state-changing steps create new child states
- **analyzer.py**: `score_suspicious_steps()` + `locate_failure()` — rule-based suspicious pattern detection and failure diagnosis
- **report.py**: `write_outputs()` — generates 4 output files (`normalized_steps.json`, `trace_tree.md`, `diagnosis.json`, `diagnosis.md`)

### Trace Tree Logic

- State 0 is the root. Exploration steps (no state change) stay under the current state.
- State-changing steps (`edit_file`, `env_change`, git checkout/reset/apply) create new child states.
- Output renders as indented steps with `-> State N` transitions.

## Input Formats

**Codex JSONL**: One JSON object per line from `codex exec --json` or `~/.codex/sessions/rollout-*.jsonl`. Events follow the `ThreadEvent` schema with `item.completed` events providing `command_execution`, `file_change`, `reasoning`, `agent_message`, and other item types. Task is inferred from content; final_status is inferred from `turn.failed` events.

The old `InternalAdapter` file remains in the repository for reference, but it is not registered by default. Treat Codex JSONL as the supported input format.

## Extension Points

- New action types → `classify_action_type()` in `classifier.py`
- New trajectory formats → implement `BaseAdapter`, call `register_adapter()` in `adapters/__init__.py`
- New suspicious patterns → `score_suspicious_steps()` in `analyzer.py`
- New output formats → `report.py`
- Tree rendering changes → `render_trace_tree()` in `tree.py`
