#!/usr/bin/env python3
"""Fetch curated LiveCodeBench problems."""

import argparse

from lcb import fetch_problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch curated LiveCodeBench problems")
    parser.add_argument("--problems-dir", default="data/lcb/problems")
    parser.add_argument("--per-difficulty", type=int, default=3)
    parser.add_argument("--repo", default="livecodebench/code_generation_lite")
    parser.add_argument("--split-file", default="test.jsonl")
    args = parser.parse_args()

    fetch_problems(
        problems_dir=args.problems_dir,
        problems_per_difficulty=args.per_difficulty,
        repo=args.repo,
        split_file=args.split_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
