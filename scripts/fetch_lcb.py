#!/usr/bin/env python3
"""Fetch curated LiveCodeBench problems from HuggingFace.

Usage:
    pip install datasets
    python scripts/fetch_lcb.py
"""

import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'lcb', 'problems')
PROBLEMS_PER_DIFFICULTY = 3


def fetch():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading LiveCodeBench dataset...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        trust_remote_code=True,
    )

    by_difficulty = {}
    for row in ds:
        diff = row.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(row)

    manifest = []

    for difficulty in ("easy", "medium", "hard"):
        problems = by_difficulty.get(difficulty, [])
        if not problems:
            print(f"  Warning: no {difficulty} problems found")
            continue

        selected = problems[:PROBLEMS_PER_DIFFICULTY]
        for problem in selected:
            qid = problem.get("question_id", "unknown")
            filename = f"{difficulty}_{qid}.json"
            filepath = os.path.join(DATA_DIR, filename)

            problem_data = {
                "question_id": qid,
                "question_title": problem.get("question_title", ""),
                "question_content": problem.get("question_content", ""),
                "platform": problem.get("platform", ""),
                "difficulty": difficulty,
                "starter_code": problem.get("starter_code", ""),
                "public_test_cases": problem.get("public_test_cases", []),
            }

            with open(filepath, 'w') as f:
                json.dump(problem_data, f, indent=2, ensure_ascii=False)

            manifest.append({
                "id": qid,
                "title": problem_data["question_title"],
                "difficulty": difficulty,
                "platform": problem_data["platform"],
                "filename": filename,
            })
            print(f"  Saved: {filename} — {problem_data['question_title'][:60]}")

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(manifest)} problems saved to {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    fetch()
