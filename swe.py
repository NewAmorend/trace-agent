"""SWE-bench Lite helpers for long-horizon Codex trajectory capture."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from runner import CodexRunResult, run_claude_trace, run_codex_trace, run_deerflow_trace


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "test"
DEFAULT_TASKS_DIR = Path("data/swe/tasks")
DEFAULT_WORKTREES_DIR = Path("data/swe/worktrees")
DEFAULT_TRAJECTORIES_DIR = Path("data/swe/trajectories")


def fetch_tasks(
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    *,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    """Fetch SWE-bench Lite task rows via HuggingFace's JSON rows API."""
    target_dir = Path(tasks_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    query = urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": limit,
    })
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    print(f"Loading SWE-bench Lite tasks from {url}...")

    with urlopen(url, timeout=30) as response:
        payload = json.load(response)

    tasks = []
    for item in payload.get("rows", []):
        task = item["row"]
        task["row_idx"] = item.get("row_idx")
        filename = f"{task['instance_id']}.json"
        (target_dir / filename).write_text(
            json.dumps(task, indent=2, ensure_ascii=False)
        )
        tasks.append({
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "filename": filename,
            "row_idx": task.get("row_idx"),
        })
        print(f"  Saved: {task['instance_id']} ({task['repo']})")

    (target_dir / "manifest.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False)
    )
    print(f"\nDone. {len(tasks)} tasks saved to {target_dir.resolve()}")
    return tasks


def load_task(instance_id: str, tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> dict:
    """Load one fetched SWE task by instance id."""
    path = Path(tasks_dir) / f"{instance_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `trace-agent swe fetch` first.")
    with path.open("r", encoding='utf-8') as f:
        return json.load(f)


def prepare_workspace(
    task: dict,
    worktrees_dir: str | Path = DEFAULT_WORKTREES_DIR,
) -> Path:
    """Clone the task repo and check out the base commit if needed."""
    worktree = Path(worktrees_dir) / task["instance_id"]
    if worktree.exists():
        print(f"Workspace already exists: {worktree}")
        return worktree

    worktree.parent.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{task['repo']}.git"
    print(f"Cloning {repo_url} into {worktree}...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(worktree)], check=True, timeout=300)
    subprocess.run(
        ["git", "-C", str(worktree), "checkout", task["base_commit"]],
        check=True,
        timeout=60,
    )
    return worktree


def build_prompt(task: dict, *, include_test_patch: bool = False) -> str:
    """Build a long-horizon Codex prompt for a SWE-bench task."""
    parts = [
        f"SWE-bench Lite task: {task['instance_id']}",
        f"Repository: {task['repo']}",
        f"Base commit: {task['base_commit']}",
        "",
        "Problem statement:",
        task.get("problem_statement", "").strip(),
    ]

    hints = (task.get("hints_text") or "").strip()
    if hints:
        parts.extend(["", "Hints:", hints])

    fail_to_pass = (task.get("FAIL_TO_PASS") or "").strip()
    if fail_to_pass:
        parts.extend(["", "Known fail-to-pass tests:", fail_to_pass])

    if include_test_patch and task.get("test_patch"):
        parts.extend(["", "Reference regression test patch:", task["test_patch"]])

    parts.extend([
        "",
        "Instructions:",
        "- Inspect the repository before editing.",
        "- Fix the root cause in the implementation.",
        "- Add or update tests only when they are needed to verify the behavior.",
        "- Run the most relevant tests you can in this sandbox.",
        "- Stop when the fix is implemented and verification is complete.",
    ])
    return "\n".join(parts)


def apply_test_patch(workspace: str | Path, task: dict) -> Path | None:
    """Apply the task's public regression test patch to the prepared workspace."""
    patch = task.get("test_patch")
    if not patch:
        return None

    workspace_path = Path(workspace)
    patch_path = workspace_path / ".swe_test.patch"
    patch_path.write_text(patch)
    subprocess.run(["git", "-C", str(workspace_path), "apply", str(patch_path)], check=True)
    print(f"Applied test patch: {patch_path}")
    return patch_path


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
    elif agent == "deerflow":
        result = run_deerflow_trace(
            prompt,
            output=trajectory_path,
            cwd=workspace,
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
