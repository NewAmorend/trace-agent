# Codex Trajectory Analyzer

A minimal Python command-line tool for analyzing coding agent trajectories. This tool parses agent execution traces and produces:

1. **Normalized Steps**: Enriched step data with action types and stages
2. **Trace Tree**: Visual representation of agent exploration and state changes
3. **Suspicious Step Detection**: Scores steps for potentially problematic patterns
4. **Failure Diagnosis**: Identifies critical failure points and provides replay suggestions

## Input Format

The tool expects a JSON file with this structure:

```json
{
  "task": "Fix the parser bug",
  "final_status": "failed",
  "steps": [
    {
      "step_id": 1,
      "thought": "I need to inspect the parser",
      "action": "rg \"parse\" .",
      "observation": "parser.py contains parse_config",
      "diff": null
    }
  ]
}
```

### Required Fields
- `task`: Description of what the agent is trying to accomplish
- `final_status`: "success" or "failed"
- `steps`: Array of step objects

### Step Fields
- `step_id` (required): Integer identifier
- `thought` (optional): Agent's reasoning
- `action` (required): Command the agent executed
- `observation` (optional): Result of the action
- `diff` (optional): File diff if applicable

## How to Run

```bash
python main.py --input examples/failed_run_001.json --output out/failed_run_001
```

This will generate four files in the output directory:

- `normalized_steps.json` - Full step data with classifications
- `trace_tree.md` - Visual tree of agent execution
- `diagnosis.json` - Machine-readable diagnosis
- `diagnosis.md` - Human-readable diagnosis report

## Example Command

```bash
python main.py --input examples/failed_run_001.json --output out/analysis
```

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

- `models.py`: Data structures (Step, NormalizedStep, TraceNode, Diagnosis)
- `parser.py`: JSON loading and validation
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

## Extending the Analyzer

To extend the analyzer:

1. Add new action types in `classifier.py`
2. Add new suspicious rules in `analyzer.py`
3. Enhance output formats in `report.py`
4. Modify tree rendering in `tree.py`
