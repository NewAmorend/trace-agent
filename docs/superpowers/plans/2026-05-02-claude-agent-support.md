# Claude Agent Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Claude Code CLI as an alternative agent harness alongside Codex for `run`, `lcb run`, and `swe run` commands, with trajectory eval working unchanged.

**Architecture:** A new `ClaudeAdapter` parses Claude Code's `stream-json` format into the same `Trajectory`/`Step` model. A new `run_claude_trace()` in `runner.py` runs `claude -p "..." --output-format stream-json` as a subprocess. The `--agent {codex,claude}` flag on three subcommands dispatches to the right runner; the eval pipeline is untouched.

**Tech Stack:** Python 3.10+, stdlib only, `claude` CLI binary (external dep for runtime, not tests), `unittest`.

---

## File Map

| File | Change |
|---|---|
| `adapters/claude_adapter.py` | **Create** — ClaudeAdapter detect + transform |
| `adapters/__init__.py` | **Modify** — append ClaudeAdapter to registry |
| `runner.py` | **Modify** — add `build_claude_exec_command()` + `run_claude_trace()` |
| `lcb.py` | **Modify** — add `agent` param to `run_problem()` + `run_lcb()`; refactor to use runner |
| `swe.py` | **Modify** — add `agent` param to `run_task()` |
| `main.py` | **Modify** — add `--agent` flag to `run`, `lcb run`, `swe run` subparsers |
| `tests/test_claude_eval.py` | **Create** — all Claude-specific tests |

---

## Task 1: ClaudeAdapter

**Files:**
- Create: `adapters/claude_adapter.py`
- Create: `tests/test_claude_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claude_eval.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from adapters.claude_adapter import ClaudeAdapter
from evaluator import evaluate_file


def write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events))


def sample_claude_events(final_failure: bool = False) -> list[dict]:
    return [
        {"type": "system", "subtype": "init", "session_id": "sess-1", "tools": ["Bash", "Write"]},
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me inspect the code."},
                    {"type": "text", "text": "I'll look at the source files first."},
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {"command": "ls src/"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tu-1",
            "content": [{"type": "text", "text": "parser.py\nanalyzer.py"}],
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg-2",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-2",
                        "name": "Write",
                        "input": {"file_path": "src/parser.py", "content": "# fixed"},
                    },
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tu-2",
            "content": [{"type": "text", "text": "File written successfully"}],
        },
        {
            "type": "result",
            "subtype": "error_during_execution" if final_failure else "success",
            "result": "An error occurred" if final_failure else "Task completed",
            "session_id": "sess-1",
        },
    ]


class ClaudeAdapterDetectTests(unittest.TestCase):
    def test_detect_returns_true_for_claude_stream(self):
        adapter = ClaudeAdapter()
        self.assertTrue(adapter.detect([
            {"type": "system", "subtype": "init", "session_id": "abc"},
        ]))

    def test_detect_returns_false_for_codex_stream(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect([
            {"type": "thread.started", "thread_id": "t1"},
        ]))

    def test_detect_returns_false_for_non_list(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect({}))

    def test_detect_returns_false_for_system_non_init(self):
        adapter = ClaudeAdapter()
        self.assertFalse(adapter.detect([
            {"type": "system", "subtype": "other"},
        ]))


class ClaudeAdapterTransformTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()
        self.trajectory = self.adapter.transform(sample_claude_events(), "data/task.jsonl")

    def test_thread_id_extracted_from_init(self):
        self.assertEqual(self.trajectory.thread_id, "sess-1")

    def test_final_status_success(self):
        self.assertEqual(self.trajectory.final_status, "success")

    def test_step_count(self):
        # text block (agent_message) + Bash (command_execution) + Write (file_change) = 3
        self.assertEqual(len(self.trajectory.steps), 3)

    def test_text_block_becomes_agent_message(self):
        step = self.trajectory.steps[0]
        self.assertEqual(step.item_type, "agent_message")
        self.assertIn("source files", step.observation)

    def test_thinking_block_captured_as_thought(self):
        step = self.trajectory.steps[0]
        self.assertIn("inspect", step.thought.lower())

    def test_bash_step_is_command_execution(self):
        step = self.trajectory.steps[1]
        self.assertEqual(step.item_type, "command_execution")
        self.assertEqual(step.action, "ls src/")

    def test_bash_tool_result_merged_as_observation(self):
        step = self.trajectory.steps[1]
        self.assertIn("parser.py", step.observation)

    def test_write_step_is_file_change(self):
        step = self.trajectory.steps[2]
        self.assertEqual(step.item_type, "file_change")
        self.assertIn("src/parser.py", step.action)

    def test_write_tool_result_merged_as_observation(self):
        step = self.trajectory.steps[2]
        self.assertIn("written", step.observation)

    def test_error_result_sets_failed_status(self):
        traj = self.adapter.transform(sample_claude_events(final_failure=True))
        self.assertEqual(traj.final_status, "failed")
        self.assertIsNotNone(traj.failure_message)

    def test_task_inferred_from_source_path(self):
        self.assertEqual(self.trajectory.task, "data task")


class ClaudeEndToEndEvalTests(unittest.TestCase):
    def test_evaluate_claude_trajectory_produces_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claude_run.jsonl"
            write_jsonl(path, sample_claude_events())

            result = evaluate_file(path)

            self.assertEqual(result.final_status, "success")
            self.assertEqual(result.metrics.total_steps, 3)
            self.assertEqual(result.metrics.command_steps, 1)
            self.assertEqual(result.metrics.file_change_steps, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_claude_eval.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'adapters.claude_adapter'`

- [ ] **Step 3: Implement ClaudeAdapter**

Create `adapters/claude_adapter.py`:

```python
"""Adapter for Claude Code CLI --output-format stream-json."""

import json
from adapters.base import BaseAdapter
from models import Step, Trajectory


class ClaudeAdapter(BaseAdapter):
    """Handles Claude Code CLI stream-json output (claude -p ... --output-format stream-json)."""

    def detect(self, data: object) -> bool:
        if not isinstance(data, list):
            return False
        for line in data[:10]:
            if (
                isinstance(line, dict)
                and line.get('type') == 'system'
                and line.get('subtype') == 'init'
            ):
                return True
        return False

    def transform(self, data: list[dict], source_path: str = "") -> Trajectory:
        steps: list[Step] = []
        step_id = 0
        pending_thought: str | None = None
        thread_id: str | None = None
        has_failure = False
        failure_message: str | None = None
        tool_use_step_idx: dict[str, int] = {}

        for line in data:
            event_type = line.get('type', '')

            if event_type == 'system' and line.get('subtype') == 'init':
                thread_id = line.get('session_id') or thread_id
                continue

            if event_type == 'assistant':
                message = line.get('message', {})
                content_blocks = message.get('content', [])
                msg_id = message.get('id')

                for block in content_blocks:
                    block_type = block.get('type', '')

                    if block_type == 'thinking':
                        pending_thought = block.get('thinking', '') or pending_thought
                        continue

                    if block_type == 'text':
                        text = block.get('text', '').strip()
                        if text:
                            step_id += 1
                            steps.append(Step(
                                step_id=step_id,
                                event_id=msg_id,
                                thought=pending_thought,
                                action='',
                                observation=text,
                                diff=None,
                                item_type='agent_message',
                            ))
                            pending_thought = None
                        continue

                    if block_type == 'tool_use':
                        tool_name = block.get('name', '')
                        tool_id = block.get('id', '')
                        tool_input = block.get('input', {})

                        step_id += 1
                        if tool_name == 'Bash':
                            action = tool_input.get('command', '')
                            item_type = 'command_execution'
                            diff = None
                        elif tool_name in ('Write', 'Edit', 'MultiEdit', 'str_replace_based_edit_tool'):
                            path = (
                                tool_input.get('file_path')
                                or tool_input.get('path')
                                or '?'
                            )
                            action = f"apply_patch {path}"
                            item_type = 'file_change'
                            diff = path
                        else:
                            action = tool_name
                            if tool_input:
                                action += f" {json.dumps(tool_input)}"
                            item_type = 'command_execution'
                            diff = None

                        steps.append(Step(
                            step_id=step_id,
                            event_id=tool_id,
                            thought=pending_thought,
                            action=action,
                            observation=None,
                            diff=diff,
                            item_type=item_type,
                        ))
                        tool_use_step_idx[tool_id] = len(steps) - 1
                        pending_thought = None
                continue

            if event_type == 'tool_result':
                tool_use_id = line.get('tool_use_id', '')
                obs = _extract_tool_result_text(line.get('content', []))
                if tool_use_id in tool_use_step_idx:
                    steps[tool_use_step_idx[tool_use_id]].observation = obs
                continue

            if event_type == 'result':
                if line.get('subtype') != 'success':
                    has_failure = True
                    failure_message = line.get('result') or line.get('subtype')
                continue

        return Trajectory(
            source_path=source_path,
            task=_infer_task_from_source(source_path),
            final_status='failed' if has_failure else 'success',
            steps=steps,
            thread_id=thread_id,
            failure_message=failure_message,
        )


def _extract_tool_result_text(content: list) -> str | None:
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get('type') == 'text':
            parts.append(item.get('text', ''))
        elif isinstance(item, str):
            parts.append(item)
    return '\n'.join(parts) if parts else None


def _infer_task_from_source(source_path: str) -> str:
    if not source_path:
        return 'Unknown task'
    name = source_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    return name.replace('_', ' ').replace('-', ' ')
```

- [ ] **Step 4: Run tests — expect ClaudeEndToEndEvalTests to still fail (adapter not registered)**

```bash
python -m pytest tests/test_claude_eval.py -v 2>&1 | tail -20
```

Expected: detect + transform tests pass; `ClaudeEndToEndEvalTests` fails with `ValueError: Unrecognized trajectory format`.

- [ ] **Step 5: Commit**

```bash
git add adapters/claude_adapter.py tests/test_claude_eval.py
git commit -m "feat: add ClaudeAdapter for stream-json trajectory format"
```

---

## Task 2: Register ClaudeAdapter

**Files:**
- Modify: `adapters/__init__.py`

- [ ] **Step 1: Register the adapter**

Edit `adapters/__init__.py` — add ClaudeAdapter after CodexAdapter:

```python
"""Adapter registry for trajectory format detection and conversion."""

from adapters.base import BaseAdapter
from adapters.claude_adapter import ClaudeAdapter
from adapters.codex_adapter import CodexAdapter

_ADAPTERS: list[BaseAdapter] = [
    CodexAdapter(),
    ClaudeAdapter(),
]


def get_adapter(data: object) -> BaseAdapter | None:
    """Find and return the first adapter that can handle the given data."""
    for adapter in _ADAPTERS:
        if adapter.detect(data):
            return adapter
    return None


def register_adapter(adapter: BaseAdapter) -> None:
    """Register a custom adapter. Added at the front of the detection list."""
    _ADAPTERS.insert(0, adapter)
```

- [ ] **Step 2: Run all tests**

```bash
python -m unittest discover -v 2>&1 | tail -30
```

Expected: all tests pass, including `ClaudeEndToEndEvalTests`.

- [ ] **Step 3: Commit**

```bash
git add adapters/__init__.py
git commit -m "feat: register ClaudeAdapter in adapter registry"
```

---

## Task 3: Claude Runner

**Files:**
- Modify: `runner.py`
- Modify: `tests/test_claude_eval.py`

- [ ] **Step 1: Add runner tests to `tests/test_claude_eval.py`**

First, add this import to the **top** of `tests/test_claude_eval.py` (with the other imports):

```python
from runner import build_claude_exec_command
```

Then append this class to `tests/test_claude_eval.py` (before `if __name__ == "__main__":`):

```python
class ClaudeRunnerCommandTests(unittest.TestCase):
    def test_build_claude_exec_command_structure(self):
        cmd = build_claude_exec_command("Fix the bug")
        self.assertEqual(cmd[0], "claude")
        self.assertIn("-p", cmd)
        self.assertIn("Fix the bug", cmd)
        self.assertIn("--output-format", cmd)
        idx = cmd.index("--output-format")
        self.assertEqual(cmd[idx + 1], "stream-json")

    def test_build_claude_exec_command_with_model(self):
        cmd = build_claude_exec_command("Fix the bug", model="claude-opus-4-7")
        self.assertIn("--model", cmd)
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx + 1], "claude-opus-4-7")

    def test_build_claude_exec_command_without_model_has_no_model_flag(self):
        cmd = build_claude_exec_command("Fix the bug")
        self.assertNotIn("--model", cmd)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_claude_eval.py::ClaudeRunnerCommandTests -v
```

Expected: `ImportError: cannot import name 'build_claude_exec_command' from 'runner'`

- [ ] **Step 3: Add `build_claude_exec_command` and `run_claude_trace` to `runner.py`**

Append to `runner.py` after the existing `run_codex_trace()` function (before `_as_text`):

```python
def build_claude_exec_command(
    prompt: str,
    *,
    model: str | None = None,
) -> list[str]:
    """Build the claude command used by the Claude runner."""
    command = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"]
    if model:
        command.extend(["--model", model])
    return command


def run_claude_trace(
    prompt: str,
    *,
    output: str | Path,
    cwd: str | Path = ".",
    model: str | None = None,
    timeout: int | None = None,
) -> CodexRunResult:
    """Run Claude Code CLI and write its JSONL event stream to output."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_claude_exec_command(prompt, model=model)

    print(f"Running Claude Code in {Path(cwd).resolve()}...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except FileNotFoundError:
        print("ERROR: 'claude' CLI not found. Is it installed?", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            output_path.write_text(_as_text(exc.stdout))
        raise

    output_path.write_text(result.stdout)
    event_count = len([line for line in result.stdout.splitlines() if line.strip()])

    if result.stderr:
        stderr_path = output_path.with_suffix(output_path.suffix + ".stderr")
        stderr_path.write_text(result.stderr)
        print(f"Claude stderr saved to {stderr_path}")

    print(f"Saved {event_count} events to {output_path}")
    if result.returncode != 0:
        print(f"WARNING: claude exited with {result.returncode}")

    return CodexRunResult(
        trajectory_path=output_path,
        returncode=result.returncode,
        event_count=event_count,
        stderr=result.stderr,
    )
```

- [ ] **Step 4: Run all tests**

```bash
python -m unittest discover -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add runner.py tests/test_claude_eval.py
git commit -m "feat: add build_claude_exec_command and run_claude_trace to runner"
```

---

## Task 4: lcb.py and swe.py — agent parameter

**Files:**
- Modify: `lcb.py`
- Modify: `swe.py`

- [ ] **Step 1: Refactor `run_problem()` in `lcb.py`**

First, add these imports to the **top** of `lcb.py` (with the other imports):

```python
from runner import run_claude_trace, run_codex_trace
```

Replace the existing `run_problem()` function (lines 104–154) with:

```python
def run_problem(
    problem_file: str,
    problems_dir: str | Path = DEFAULT_PROBLEMS_DIR,
    trajectories_dir: str | Path = DEFAULT_TRAJECTORIES_DIR,
    timeout: int = 300,
    agent: str = "codex",
) -> Path | None:
    """Run an agent against one problem file and save the JSONL trajectory."""

    problem_path = Path(problems_dir) / problem_file
    with problem_path.open("r") as f:
        problem = json.load(f)

    qid = problem["question_id"]
    difficulty = problem["difficulty"]
    prompt = build_prompt(problem)

    output_dir = Path(trajectories_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{difficulty}_{qid}.jsonl"

    print(f"  Running {problem_file} [{agent}]...")
    print(f"    Prompt: {problem.get('question_title', '???')[:50]}")

    try:
        if agent == "claude":
            result = run_claude_trace(prompt, output=output_path, timeout=timeout)
        else:
            result = run_codex_trace(prompt, output=output_path, cwd=".", timeout=timeout)
    except FileNotFoundError:
        print(f"    ERROR: '{agent}' CLI not found. Is it installed?", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: {agent} took > {timeout}s, skipping")
        return None

    if result.event_count == 0:
        print(f"    WARNING: no output from {agent} (exit code {result.returncode})")
        return None

    return result.trajectory_path
```

Also remove the now-unused `import subprocess` block check — `subprocess` is still needed for the `TimeoutExpired` catch, so keep it. Verify `import subprocess` is still at the top of `lcb.py`.

- [ ] **Step 2: Add `agent` param to `run_lcb()` in `lcb.py`**

Change the signature of `run_lcb()` and pass `agent` through to `run_problem()`:

```python
def run_lcb(
    problems_dir: str | Path = DEFAULT_PROBLEMS_DIR,
    trajectories_dir: str | Path = DEFAULT_TRAJECTORIES_DIR,
    difficulty: str = "all",
    limit: int | None = None,
    timeout: int = 300,
    agent: str = "codex",
) -> dict[str, list[str]]:
    """Run an agent against problems listed in the LiveCodeBench manifest."""
    manifest_path = Path(problems_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run `trace-agent lcb fetch` first.")

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    if difficulty != "all":
        manifest = [p for p in manifest if p["difficulty"] == difficulty]

    if limit:
        by_diff: dict[str, list[dict]] = {}
        for problem in manifest:
            by_diff.setdefault(problem["difficulty"], []).append(problem)
        manifest = []
        for problems in by_diff.values():
            manifest.extend(problems[:limit])

    if not manifest:
        print("No problems to run.")
        return {"success": [], "skipped": []}

    print(f"Running {len(manifest)} problem(s) with {agent}...\n")

    results = {"success": [], "skipped": []}
    for entry in manifest:
        path = run_problem(
            entry["filename"],
            problems_dir=problems_dir,
            trajectories_dir=trajectories_dir,
            timeout=timeout,
            agent=agent,
        )
        if path:
            results["success"].append(entry["filename"])
        else:
            results["skipped"].append(entry["filename"])

    print("\n--- Summary ---")
    print(f"  Success: {len(results['success'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Trajectories saved to: {Path(trajectories_dir).resolve()}")
    return results
```

- [ ] **Step 3: Add `agent` param to `run_task()` in `swe.py`**

First, update the existing top-level import in `swe.py` from:

```python
from runner import CodexRunResult, run_codex_trace
```

to:

```python
from runner import CodexRunResult, run_claude_trace, run_codex_trace
```

Then replace the existing `run_task()` function:

```python
def run_task(
    instance_id: str,
    *,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    worktrees_dir: str | Path = DEFAULT_WORKTREES_DIR,
    trajectories_dir: str | Path = DEFAULT_TRAJECTORIES_DIR,
    sandbox: str = "workspace-write",
    model: str | None = None,
    profile: str | None = None,
    full_auto: bool = False,
    include_test_patch: bool = False,
    apply_tests: bool = False,
    timeout: int | None = None,
    eval_output: str | Path | None = None,
    agent: str = "codex",
) -> tuple[CodexRunResult, Path]:
    """Prepare a SWE workspace, run an agent, and capture the trajectory."""

    task = load_task(instance_id, tasks_dir)
    workspace = prepare_workspace(task, worktrees_dir)
    if apply_tests:
        apply_test_patch(workspace, task)

    prompt = build_prompt(task, include_test_patch=include_test_patch)
    trajectory_path = Path(trajectories_dir) / f"{instance_id}.jsonl"

    if agent == "claude":
        result = run_claude_trace(
            prompt,
            output=trajectory_path,
            cwd=workspace,
            model=model,
            timeout=timeout,
        )
    else:
        result = run_codex_trace(
            prompt,
            output=trajectory_path,
            cwd=workspace,
            sandbox=sandbox,
            model=model,
            profile=profile,
            full_auto=full_auto,
            timeout=timeout,
        )
    return result, workspace
```

- [ ] **Step 4: Run all tests**

```bash
python -m unittest discover -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add lcb.py swe.py
git commit -m "feat: add agent param to lcb.run_lcb and swe.run_task"
```

---

## Task 5: main.py — --agent flag and CLI tests

**Files:**
- Modify: `main.py`
- Modify: `tests/test_claude_eval.py`

- [ ] **Step 1: Add CLI tests to `tests/test_claude_eval.py`**

First, add this import to the **top** of `tests/test_claude_eval.py` (with the other imports):

```python
from main import build_parser
```

Then append this class to `tests/test_claude_eval.py` (before `if __name__ == "__main__":`):

```python
class ClaudeCLIFlagTests(unittest.TestCase):
    def test_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["run", "Fix bug", "--output", "out.jsonl"])
        self.assertEqual(args.agent, "codex")

    def test_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["run", "Fix bug", "--output", "out.jsonl", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")

    def test_lcb_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["lcb", "run"])
        self.assertEqual(args.agent, "codex")

    def test_lcb_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["lcb", "run", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")

    def test_swe_run_defaults_to_codex(self):
        parser = build_parser()
        args = parser.parse_args(["swe", "run", "instance-1"])
        self.assertEqual(args.agent, "codex")

    def test_swe_run_accepts_claude_agent(self):
        parser = build_parser()
        args = parser.parse_args(["swe", "run", "instance-1", "--agent", "claude"])
        self.assertEqual(args.agent, "claude")
```

- [ ] **Step 2: Run to confirm failures**

```bash
python -m pytest tests/test_claude_eval.py::ClaudeCLIFlagTests -v
```

Expected: `AttributeError: Namespace object has no attribute 'agent'`

- [ ] **Step 3: Add `--agent` to `run` subparser in `main.py`**

In `build_parser()`, find the `run_trace_parser` block (around line 231). Add after `run_trace_parser.add_argument("--quiet", ...)`:

```python
    run_trace_parser.add_argument(
        "--agent",
        choices=["codex", "claude"],
        default="codex",
        help="Agent harness to use (default: codex)",
    )
```

- [ ] **Step 4: Add `--agent` to `lcb run` subparser in `main.py`**

Find the `run_parser` block (the `lcb run` subparser, around line 279). Add after `run_parser.add_argument("--timeout", ...)`:

```python
    run_parser.add_argument(
        "--agent",
        choices=["codex", "claude"],
        default="codex",
        help="Agent harness to use (default: codex)",
    )
```

- [ ] **Step 5: Add `--agent` to `swe run` subparser in `main.py`**

Find the `swe_run_parser` block (around line 332). Add after `swe_run_parser.add_argument("--quiet", ...)`:

```python
    swe_run_parser.add_argument(
        "--agent",
        choices=["codex", "claude"],
        default="codex",
        help="Agent harness to use (default: codex)",
    )
```

- [ ] **Step 6: Wire `args.agent` through command handlers in `main.py`**

In `run_codex_command()`, change the `run_codex_trace(...)` call to dispatch on agent:

```python
def run_codex_command(args: argparse.Namespace) -> int:
    try:
        if args.agent == "claude":
            from runner import run_claude_trace
            result = run_claude_trace(
                args.prompt,
                output=args.output,
                cwd=args.cwd,
                model=args.model,
                timeout=args.timeout,
            )
        else:
            result = run_codex_trace(
                args.prompt,
                output=args.output,
                cwd=args.cwd,
                sandbox=args.sandbox,
                model=args.model,
                profile=args.profile,
                add_dirs=args.add_dir,
                full_auto=args.full_auto,
                skip_git_repo_check=args.skip_git_repo_check,
                ephemeral=args.ephemeral,
                timeout=args.timeout,
            )
        if args.eval_output:
            eval_args = argparse.Namespace(
                input=str(result.trajectory_path),
                output=args.eval_output,
                ci=False,
                quiet=args.quiet,
            )
            eval_code = run_eval_command(eval_args)
            if eval_code:
                return eval_code
        return 0 if result.returncode == 0 else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
```

In `run_lcb_run_command()`, pass `agent=args.agent` to `run_lcb()`:

```python
def run_lcb_run_command(args: argparse.Namespace) -> int:
    try:
        run_lcb(
            problems_dir=args.problems_dir,
            trajectories_dir=args.trajectories_dir,
            difficulty=args.difficulty,
            limit=args.limit,
            timeout=args.timeout,
            agent=args.agent,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0
```

In `run_swe_run_command()`, pass `agent=args.agent` to `run_task()`:

```python
def run_swe_run_command(args: argparse.Namespace) -> int:
    try:
        result, workspace = run_task(
            args.instance,
            tasks_dir=args.tasks_dir,
            worktrees_dir=args.worktrees_dir,
            trajectories_dir=args.trajectories_dir,
            sandbox=args.sandbox,
            model=args.model,
            profile=args.profile,
            full_auto=args.full_auto,
            include_test_patch=args.include_test_patch,
            apply_tests=args.apply_tests,
            timeout=args.timeout,
            agent=args.agent,
        )
        print(f"Workspace: {workspace}")
        if args.eval_output:
            eval_args = argparse.Namespace(
                input=str(result.trajectory_path),
                output=args.eval_output,
                ci=False,
                quiet=args.quiet,
            )
            eval_code = run_eval_command(eval_args)
            if eval_code:
                return eval_code
        return 0 if result.returncode == 0 else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
```

- [ ] **Step 7: Run all tests**

```bash
python -m unittest discover -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_claude_eval.py
git commit -m "feat: add --agent flag to run, lcb run, swe run subcommands"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run full test suite**

```bash
python -m unittest discover -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Smoke-test the parser with help output**

```bash
python main.py run --help | grep agent
python main.py lcb run --help | grep agent
python main.py swe run --help | grep agent
```

Expected: each shows `--agent {codex,claude}` in the help text.

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "feat: Claude Code CLI agent support complete"
```
