# Agent Trajectory Eval

A Codex-first evaluation toolkit for coding agent trajectories. It reads Codex
CLI JSONL session or `codex exec --json` streams and produces:

1. **Normalized Steps**: Codex events converted into consistent step records
2. **Trace Tree**: State transitions created by file and environment changes
3. **Risk Metrics**: Counts for commands, tests, file changes, failures, and suspicious behavior
4. **Failure Diagnosis**: Rule-based critical-step detection with replay hints
5. **Batch Reports**: Directory-level summaries for CI or regression evaluation

The project intentionally focuses on Codex JSONL today. The adapter interface is
still present, but only the Codex adapter is registered by default.

## Quick Start

```bash
trace-agent eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval
```

For a directory of trajectories:

```bash
trace-agent eval --input data/lcb/trajectories --output out/lcb_eval
```

CI-style exit codes:

```bash
trace-agent eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval --ci
```

- `0`: tool ran and no evaluated trajectory failed or reached medium/high risk
- `1`: evaluation ran, but at least one trajectory failed or reached medium/high risk
- `2`: tool error, invalid input, or unsupported format

The older flat form is still supported for compatibility:

```bash
trace-eval --input examples/codex_failed_run_001.jsonl --output out/codex_eval
```

## Commands

```bash
# Run Codex in a sandbox, capture JSONL, then evaluate it
trace-agent run \
  --output data/runs/task_001.jsonl \
  --eval-output out/task_001 \
  -C /path/to/repo \
  --sandbox workspace-write \
  "Fix failing tests. Inspect first, edit code, run tests, and stop when tests pass."

# Fetch a small LiveCodeBench sample into data/lcb/problems
trace-agent lcb fetch

# Run Codex on one easy problem and save JSONL trajectories
trace-agent lcb run --difficulty easy --limit 1

# Evaluate generated LiveCodeBench trajectories
trace-agent lcb eval

# Fetch SWE-bench Lite tasks
trace-agent swe fetch --limit 5

# Prepare a real repo at the task base commit
trace-agent swe prepare astropy__astropy-12907

# Run Codex on a SWE-bench Lite task in a writable sandbox
trace-agent swe run astropy__astropy-12907 \
  --sandbox workspace-write \
  --timeout 1200 \
  --eval-output out/swe_astropy_12907

# Evaluate generated SWE trajectories
trace-agent swe eval
```

When running from a checkout without installing the package, prefix commands with
`uv run`, for example `uv run trace-agent eval --input ... --output ...`.

## Outputs

Single-trajectory output directories contain:

- `normalized_steps.json`
- `trace_tree.md`
- `diagnosis.json`
- `diagnosis.md`
- `eval_result.json`
- `eval_summary.md`

Batch runs also write:

- `batch_summary.json`
- `batch_summary.md`

## Input Format

The supported input is Codex JSONL: one JSON event per line. The evaluator
handles `thread.started`, `turn.completed`, `turn.failed`, and `item.completed`
events for Codex item types such as `reasoning`, `command_execution`,
`file_change`, `agent_message`, `mcp_tool_call`, `error`, and `web_search`.

The older internal JSON example remains in the repository as historical sample
data, but the default evaluator is now Codex-only.

## Key Concepts

### Explore vs State Change

- **Explore**: Steps that gather information without modifying the system (search, inspection)
- **State Change**: Steps that modify the system (file edits, environment changes)

State-changing steps create new "states" in the trace tree. The agent explores within a state, then transitions to a new state after making changes.

### Stages

Actions are classified into stages:
- **environment verification**: Checking tool versions and environment setup
- **dependency installation**: Installing packages or dependencies
- **inspection/debugging**: Searching and inspecting files
- **patching**: Making code changes
- **verification**: Running tests
- **other**: Actions that don't fit other categories

### Action Types

- `inspect_file`: Reading file contents (cat, sed, head, tail)
- `search`: Searching code (rg, grep, find)
- `run_test`: Running tests (pytest, cargo test, npm test)
- `edit_file`: Modifying files (apply_patch, write file)
- `env_change`: Installing dependencies (pip install, npm install)
- `git_action`: Git operations
- `other`: Unclassified actions

## Suspicious Step Detection

The tool detects potentially problematic patterns:

- **Test file manipulation**: Editing test files to make tests pass
- **Patches followed by failing tests**: Changes that don't fix issues
- **Repeated commands**: Redundant actions
- **Repeated test failures**: Failing the same test without intervention
- **Environment issues**: Dependency problems after environment changes
- **Git rollbacks**: Trial-and-error behavior

Each suspicious step gets a score (0.0 to 1.0+) and explanatory reasons.

## Output Files

### normalized_steps.json

Complete step data with all classifications:

```json
[
  {
    "step_id": 1,
    "thought": "I need to inspect the parser",
    "action": "rg \"parse\" .",
    "observation": "parser.py contains parse_config",
    "diff": null,
    "action_type": "search",
    "stage": "inspection/debugging",
    "state_change": false,
    "suspicious_score": 0.0,
    "suspicious_reasons": []
  }
]
```

### trace_tree.md

Visual representation showing state transitions:

```markdown
# Trace Tree

State 0
  - Step 1 [inspection/debugging | search | explore] rg "parse" .
  - Step 2 [inspection/debugging | inspect_file | explore] sed -n '1,160p' parser.py
  - Step 3 [patching | edit_file | state_change] apply_patch parser.py
    -> State 1
```

### diagnosis.md

Human-readable analysis report including:
- Task description and final status
- Critical failure step
- Table of all suspicious steps with scores and reasons
- Replay suggestion with hints for alternative approaches

## Limitations

This is a **minimal viable product (MVP)** - a rule-based analyzer, not a full CodeTracer implementation. It uses simple pattern matching and heuristic rules rather than machine learning or sophisticated semantic analysis.

## Architecture

The codebase is organized into clear modules:

- `models.py`: Data structures (Trajectory, Step, NormalizedStep, TraceNode, Diagnosis, EvalResult)
- `parser.py`: Codex JSONL loading and validation
- `evaluator.py`: Single-file evaluation, directory discovery, and batch summaries
- `adapters/codex_adapter.py`: Codex event stream conversion
- `classifier.py`: Action type, stage, and state change classification
- `tree.py`: Trace tree building and rendering
- `analyzer.py`: Suspicious step scoring and failure diagnosis
- `report.py`: Output file generation
- `main.py`: CLI interface

## Technical Details

- **Language**: Python 3.x
- **Dependencies**: Python standard library only
- **Design**: Pure Python standard library
- **Design**: Clear separation of concerns
- **Extensibility**: Easy to add new classification rules
- **Type hints**: Added for better code clarity
- **Tests**: `python -m unittest discover`

## Extending the Analyzer

To extend the analyzer:

1. Add new action types in `classifier.py`
2. Add new suspicious rules in `analyzer.py`
3. Enhance output formats in `report.py`
4. Modify tree rendering in `tree.py`
