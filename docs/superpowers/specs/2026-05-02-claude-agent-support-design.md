# Claude Agent Support Design

**Date:** 2026-05-02
**Status:** Approved

## Problem

trace-agent can only run Codex CLI as the agent harness. Users want to run Claude Code CLI as an alternative agent, capture its trajectory, and evaluate it through the same pipeline.

## Goal

Add `--agent {codex,claude}` to the `run`, `lcb run`, and `swe run` commands. Both agents produce a `.jsonl` trajectory file that flows through the existing eval pipeline unchanged.

## Non-Goals

- API-based agent harness (OpenAI SDK / Anthropic SDK agentic loop) — deferred
- Sandbox/permission control for Claude Code CLI — Claude manages its own permissions
- Changes to the eval pipeline (`classifier`, `analyzer`, `tree`, `report`)

## Architecture

```
CLI (main.py)
  --agent {codex, claude}       ← new flag on run / lcb run / swe run

runner.py
  run_codex_trace()             ← unchanged
  run_claude_trace()            ← new

adapters/
  codex_adapter.py              ← unchanged
  claude_adapter.py             ← new
  __init__.py                   ← register ClaudeAdapter

eval pipeline                   ← completely unchanged
  parser → classifier → tree → analyzer → report
```

## Components

### runner.py — `run_claude_trace()`

Mirrors `run_codex_trace()` in structure. Calls:

```
claude -p "<prompt>" --output-format stream-json --verbose -C <cwd>
```

- `--output-format stream-json` emits one JSON object per line to stdout
- `--verbose` includes tool results in the stream (required for trajectory capture)
- `-C <cwd>` sets the working directory (same as Codex)
- `--model <model>` passed through when provided
- `--sandbox`, `--full-auto`, `--skip-git-repo-check`, `--ephemeral` are Codex-only; silently ignored for Claude runs

Returns the same `CodexRunResult` dataclass. Writes output to the same `.jsonl` path. Timeout and stderr handling identical to Codex path.

### adapters/claude_adapter.py — `ClaudeAdapter`

**Detection:** `detect()` returns `True` if any of the first 10 lines contains `{"type":"system","subtype":"init"}` — unique to Claude Code stream-json format.

**Claude Code stream-json event types:**

| Event | Fields used |
|---|---|
| `system` / `init` | `session_id` (stored as `thread_id`) |
| `assistant` | `message.content[]` — text blocks and tool_use blocks |
| `tool_result` | `tool_use_id`, `content[].text` — merged into preceding step |
| `result` | `subtype` (`success`/`error`), `result` text |

**Step mapping:**

| Claude event | `item_type` | `action` | `observation` |
|---|---|---|---|
| `assistant` text block | `agent_message` | — | text content |
| `assistant` tool_use `Bash` | `command_execution` | command string | filled by matching `tool_result` |
| `assistant` tool_use `Write`/`Edit`/`str_replace_based_edit_tool` | `file_change` | file path | filled by matching `tool_result` |
| `assistant` tool_use (other) | `command_execution` | `tool_name args` | filled by matching `tool_result` |
| `result` subtype=`error` | sets `has_failure=True`, `failure_message` | — | — |

Tool results are matched to their preceding step by `tool_use_id` and merged in. `reasoning` (thinking blocks) are captured as `pending_thought` and attached to the next step.

**Final status:** `failed` if any `result` event has `subtype=error`, else `success`.

### adapters/\_\_init\_\_.py

Register `ClaudeAdapter` alongside `CodexAdapter`. Detection order: CodexAdapter first (existing behavior), ClaudeAdapter second.

### main.py — CLI changes

Add `--agent {codex,claude}` argument to:
- `run` subparser (default: `codex`)
- `lcb run` subparser (default: `codex`)
- `swe run` subparser (default: `codex`)

Dispatch logic:
```python
if args.agent == "claude":
    result = run_claude_trace(prompt, output=..., cwd=..., model=..., ...)
else:
    result = run_codex_trace(prompt, output=..., cwd=..., sandbox=..., model=..., ...)
```

`eval`, `lcb eval`, `swe eval` are unchanged — adapter auto-detection handles format.

## Data Flow

```
trace-agent lcb run --agent claude --difficulty easy
  → build_prompt(problem)
  → run_claude_trace(prompt, output="data/lcb/trajectories/easy_X.jsonl")
      → subprocess: claude -p "..." --output-format stream-json --verbose -C <workspace>
      → writes stdout to .jsonl
  → (optional) evaluate_file(.jsonl)
      → load_trajectory() → ClaudeAdapter.detect() → ClaudeAdapter.transform()
      → normalize_steps() → score_suspicious_steps() → build_trace_tree()
      → locate_failure() → write_outputs()
```

## Backward Compatibility

- All existing commands default to `--agent codex` — no behavior change
- Existing Codex `.jsonl` files evaluate identically — `CodexAdapter` still detected first
- `CodexRunResult` dataclass reused unchanged

## Testing

- Unit test `ClaudeAdapter.detect()` — rejects Codex lines, accepts Claude init line
- Unit test `ClaudeAdapter.transform()` — fixture with sample Claude stream-json events covering: text block, Bash tool_use + tool_result, file_change tool_use + tool_result, error result
- Existing `test_codex_eval.py` must pass unchanged
