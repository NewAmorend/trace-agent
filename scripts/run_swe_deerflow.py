#!/usr/bin/env python3
"""Batch run SWE-bench Lite tasks through DeerFlow and evaluate with trace-agent."""

import json
import subprocess
import sys
import time
from pathlib import Path

TRACE_AGENT = Path(__file__).resolve().parent.parent
DEERFLOW_BACKEND = Path("/home/agentuser/deer-flow/backend")
TASKS_DIR = TRACE_AGENT / "data" / "swe" / "tasks"
TRAJECTORIES_DIR = TRACE_AGENT / "data" / "swe" / "trajectories"
WORKTREES_DIR = TRACE_AGENT / "data" / "swe" / "worktrees"
OUTPUT_DIR = TRACE_AGENT / "out" / "swe_deerflow"


def load_manifest():
    manifest_path = TASKS_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}. Run: python main.py swe fetch --limit 10")
        sys.exit(1)
    return json.loads(manifest_path.read_text())


def build_prompt(task: dict) -> str:
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
    parts.extend([
        "",
        "Instructions:",
        "- Inspect the repository before editing.",
        "- Fix the root cause in the implementation.",
        "- Run the most relevant tests you can.",
        "- Stop when the fix is implemented and verification is complete.",
    ])
    return "\n".join(parts)


def prepare_workspace(task: dict) -> Path:
    worktree = WORKTREES_DIR / task["instance_id"]
    if worktree.exists():
        return worktree
    worktree.parent.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{task['repo']}.git"
    print(f"  Cloning {repo_url} ...")
    subprocess.run(["git", "clone", repo_url, str(worktree)], check=True, timeout=600)
    subprocess.run(["git", "-C", str(worktree), "checkout", task["base_commit"]], check=True, timeout=60)
    return worktree


def run_deerflow(prompt: str, workspace: Path, output: Path) -> bool:
    venv_python = DEERFLOW_BACKEND / ".venv" / "bin" / "python"
    script = DEERFLOW_BACKEND / "scripts" / "run_agent_direct.py"

    env_file = DEERFLOW_BACKEND.parent / ".env"
    env = dict(__import__("os").environ)
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()

    cmd = [
        str(venv_python), str(script),
        "--prompt", prompt,
        "--output", str(output.resolve()),
        "--cwd", str(workspace.resolve()),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(DEERFLOW_BACKEND), env=env,
        )
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
        if result.returncode != 0 and result.stderr:
            print(f"    STDERR: {result.stderr[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (600s)")
        return False


def main():
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    tasks = manifest[:10]
    print(f"Running {len(tasks)} SWE-bench Lite tasks through DeerFlow (kimi-k2.6)\n")

    results = []
    for i, entry in enumerate(tasks):
        instance_id = entry["instance_id"]
        task_path = TASKS_DIR / f"{instance_id}.json"
        trajectory_path = TRAJECTORIES_DIR / f"{instance_id}.jsonl"

        print(f"[{i+1}/{len(tasks)}] {instance_id}")

        task = json.loads(task_path.read_text())
        prompt = build_prompt(task)

        try:
            workspace = prepare_workspace(task)
        except Exception as exc:
            print(f"  SKIP: workspace prep failed: {exc}")
            continue

        t0 = time.time()
        ok = run_deerflow(prompt, workspace, trajectory_path)
        elapsed = time.time() - t0
        print(f"  {'OK' if ok else 'FAIL'} ({elapsed:.1f}s)")

        results.append({"instance_id": instance_id, "success": ok, "elapsed": elapsed, "trajectory": str(trajectory_path)})

    print(f"\n{'='*60}")
    print(f"  Completed: {sum(r['success'] for r in results)}/{len(results)}")
    print(f"{'='*60}\n")

    print("Evaluating trajectories with trace-agent...")
    eval_cmd = [sys.executable, str(TRACE_AGENT / "main.py"), "eval",
                "--input", str(TRAJECTORIES_DIR), "--output", str(OUTPUT_DIR), "--quiet"]
    subprocess.run(eval_cmd, cwd=str(TRACE_AGENT))

    print(f"\nResults: {OUTPUT_DIR}/eval_result.json")


if __name__ == "__main__":
    main()
