# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Codex Trajectory Analyzer is a minimal Python CLI tool for analyzing coding agent trajectories. It parses agent execution traces (JSON and JSONL) and produces normalized steps, trace trees, suspicious step detection, and failure diagnosis reports. Supports multiple trajectory formats via an adapter pattern. Pure Python standard library — no external dependencies.

## Commands

```bash
# Run analysis (internal JSON format)
python main.py --input examples/failed_run_001.json --output out/failed_run_001

# Run analysis (Codex JSONL format)
python main.py --input examples/codex_failed_run_001.jsonl --output out/codex_failed_run_001

# Verify outputs
cat out/failed_run_001/trace_tree.md
cat out/failed_run_001/diagnosis.md
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

There is no test suite. Validation is done by running against `examples/failed_run_001.json` and checking:
- Trace tree shows state transitions: State 0 -> State 1 -> State 2
- Step 5 is flagged as suspicious (test file manipulation)
- Diagnosis identifies the correct critical step and error type

## Architecture

Entry point is `main.py`. Core modules are flat in the project root. Adapter modules live in `adapters/`.

**Data flow**: `main.py` calls modules in sequence:

```
parser.py → classifier.py → tree.py → analyzer.py → report.py
```

All modules operate on dataclasses from `models.py`: `Step` → `NormalizedStep` (enriched with classifications) → `TraceNode` (tree structure) → `Diagnosis` (failure analysis).

### Module Responsibilities

- **models.py**: Dataclasses — `Step`, `NormalizedStep`, `TraceNode`, `Diagnosis`
- **parser.py**: `load_trajectory()` — JSON/JSONL loading, adapter-based format detection, delegates to registered adapters
- **adapters/**: Format adapter layer
  - **base.py**: `BaseAdapter` ABC with `detect()` and `transform()` methods
  - **internal.py**: `InternalAdapter` — original `{task, final_status, steps}` JSON format
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

**Internal JSON**: `{task, final_status, steps}` array. Each step requires `step_id` and `action`; `thought`, `observation`, and `diff` are optional.

**Codex JSONL**: One JSON object per line from `codex exec --json` or `~/.codex/sessions/rollout-*.jsonl`. Events follow the `ThreadEvent` schema with `item.completed` events providing `command_execution`, `file_change`, `reasoning`, `agent_message`, and other item types. Task is inferred from content; final_status is inferred from `turn.failed` events.

## Extension Points

- New action types → `classify_action_type()` in `classifier.py`
- New trajectory formats → implement `BaseAdapter`, call `register_adapter()` in `adapters/__init__.py`
- New suspicious patterns → `score_suspicious_steps()` in `analyzer.py`
- New output formats → `report.py`
- Tree rendering changes → `render_trace_tree()` in `tree.py`
