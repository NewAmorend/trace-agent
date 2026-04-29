#!/usr/bin/env python3
"""Run Codex CLI against LiveCodeBench problems and capture trajectories.

Usage:
    python scripts/run_lcb.py                          # run all problems
    python scripts/run_lcb.py --difficulty easy        # easy only
    python scripts/run_lcb.py --difficulty hard --limit 1

Requires: Codex CLI installed and authenticated.
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'lcb', 'problems')
TRAJECTORIES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'lcb', 'trajectories')


def build_prompt(problem: dict) -> str:
    title = problem.get("question_title", "")
    content = problem.get("question_content", "")
    starter = problem.get("starter_code", "")
    tests = problem.get("public_test_cases", [])

    parts = [f"## Problem: {title}\n", content]

    if starter:
        parts.append(f"\n### Starter Code\n```python\n{starter}\n```")

    if tests:
        parts.append("\n### Example Test Cases")
        for i, tc in enumerate(tests[:5], 1):
            parts.append(f"Input:\n```\n{tc.get('input', '')}\n```\nExpected Output:\n```\n{tc.get('output', '')}\n```")

    parts.append(
        "\nWrite a complete Python solution. Include the solution function and tests. "
        "Make sure the solution passes all test cases."
    )

    return "\n".join(parts)


def run_problem(problem_file: str) -> str | None:
    filepath = os.path.join(PROBLEMS_DIR, problem_file)
    with open(filepath, 'r') as f:
        problem = json.load(f)

    qid = problem["question_id"]
    difficulty = problem["difficulty"]
    prompt = build_prompt(problem)

    output_path = os.path.join(TRAJECTORIES_DIR, f"{difficulty}_{qid}.jsonl")

    print(f"  Running {problem_file}...")
    print(f"    Prompt: {problem.get('question_title', '???')[:50]}")

    try:
        result = subprocess.run(
            ["codex", "exec", "--json", prompt],
            capture_output=True,
            text=True,
            timeout=300,
        )

        output = result.stdout
        if not output.strip():
            print(f"    WARNING: no output from codex (exit code {result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[:200]}")
            return None

        os.makedirs(TRAJECTORIES_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(output)

        lines = output.strip().splitlines()
        print(f"    Saved {len(lines)} events to {os.path.basename(output_path)}")
        return output_path

    except FileNotFoundError:
        print("    ERROR: 'codex' CLI not found. Is it installed?", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: Codex took > 300s, skipping")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Codex CLI against LCB problems")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "all"],
        default="all", help="Filter by difficulty (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max problems to run per difficulty"
    )
    args = parser.parse_args()

    manifest_path = os.path.join(PROBLEMS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Run scripts/fetch_lcb.py first.", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if args.difficulty != "all":
        manifest = [p for p in manifest if p["difficulty"] == args.difficulty]

    if args.limit:
        by_diff = {}
        for p in manifest:
            d = p["difficulty"]
            by_diff.setdefault(d, []).append(p)
        manifest = []
        for d, problems in by_diff.items():
            manifest.extend(problems[:args.limit])

    if not manifest:
        print("No problems to run.")
        return

    print(f"Running {len(manifest)} problem(s) with Codex CLI...\n")

    results = {"success": [], "failed": [], "skipped": []}
    for entry in manifest:
        path = run_problem(entry["filename"])
        if path:
            results["success"].append(entry["filename"])
        else:
            results["skipped"].append(entry["filename"])

    print(f"\n--- Summary ---")
    print(f"  Success: {len(results['success'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Trajectories saved to: {os.path.abspath(TRAJECTORIES_DIR)}")


if __name__ == "__main__":
    main()
