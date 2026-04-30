#!/usr/bin/env python3
"""Run Codex CLI against LiveCodeBench problems and capture trajectories."""

import argparse

from lcb import run_lcb


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Codex CLI against LCB problems")
    parser.add_argument("--problems-dir", default="data/lcb/problems")
    parser.add_argument("--trajectories-dir", default="data/lcb/trajectories")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Filter by difficulty (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max problems per difficulty")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    run_lcb(
        problems_dir=args.problems_dir,
        trajectories_dir=args.trajectories_dir,
        difficulty=args.difficulty,
        limit=args.limit,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
