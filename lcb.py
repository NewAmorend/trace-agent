"""LiveCodeBench helpers for fetching problems and running Codex."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from runner import run_claude_trace, run_codex_trace


DEFAULT_PROBLEMS_DIR = Path("data/lcb/problems")
DEFAULT_TRAJECTORIES_DIR = Path("data/lcb/trajectories")
DEFAULT_REPO = "livecodebench/code_generation_lite"
DEFAULT_SPLIT_FILE = "test.jsonl"
DEFAULT_PROBLEMS_PER_DIFFICULTY = 3


def fetch_problems(
    problems_dir: str | Path = DEFAULT_PROBLEMS_DIR,
    problems_per_difficulty: int = DEFAULT_PROBLEMS_PER_DIFFICULTY,
    repo: str = DEFAULT_REPO,
    split_file: str = DEFAULT_SPLIT_FILE,
) -> list[dict]:
    """Fetch a small LiveCodeBench sample and write project-compatible files."""
    target_dir = Path(problems_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{split_file}"
    print(f"Loading LiveCodeBench dataset from {url}...")

    limits = {
        "easy": problems_per_difficulty,
        "medium": problems_per_difficulty,
        "hard": problems_per_difficulty,
    }
    counts = {difficulty: 0 for difficulty in limits}
    manifest = []

    with urlopen(url) as response:
        for raw_line in response:
            row = json.loads(raw_line.decode("utf-8"))
            difficulty = row.get("difficulty", "unknown")
            if difficulty not in limits or counts[difficulty] >= limits[difficulty]:
                continue

            qid = str(row.get("question_id", "unknown"))
            filename = f"{difficulty}_{qid}.json"
            problem_data = _problem_from_row(row, difficulty)

            (target_dir / filename).write_text(
                json.dumps(problem_data, indent=2, ensure_ascii=False)
            )
            manifest.append({
                "id": qid,
                "title": problem_data["question_title"],
                "difficulty": difficulty,
                "platform": problem_data["platform"],
                "filename": filename,
            })
            counts[difficulty] += 1
            print(f"  Saved: {filename} - {problem_data['question_title'][:60]}")

            if all(counts[difficulty] >= limit for difficulty, limit in limits.items()):
                break

    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"\nDone. {len(manifest)} problems saved to {target_dir.resolve()}")
    return manifest


def build_prompt(problem: dict) -> str:
    """Build a Codex prompt for one LiveCodeBench problem."""
    title = problem.get("question_title", "")
    content = problem.get("question_content", "")
    starter = problem.get("starter_code", "")
    tests = problem.get("public_test_cases", [])

    parts = [f"## Problem: {title}\n", content]

    if starter:
        parts.append(f"\n### Starter Code\n```python\n{starter}\n```")

    if tests:
        parts.append("\n### Example Test Cases")
        for tc in tests[:5]:
            parts.append(
                "Input:\n"
                f"```\n{tc.get('input', '')}\n```\n"
                "Expected Output:\n"
                f"```\n{tc.get('output', '')}\n```"
            )

    parts.append(
        "\nWrite a complete Python solution. Include the solution function and tests. "
        "Make sure the solution passes all test cases."
    )

    return "\n".join(parts)


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


def _problem_from_row(row: dict, difficulty: str) -> dict:
    tests = row.get("public_test_cases", [])
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            tests = []

    return {
        "question_id": str(row.get("question_id", "unknown")),
        "question_title": row.get("question_title", ""),
        "question_content": row.get("question_content", ""),
        "platform": row.get("platform", ""),
        "difficulty": difficulty,
        "starter_code": row.get("starter_code", ""),
        "public_test_cases": tests,
    }
