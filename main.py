#!/usr/bin/env python3
"""Main CLI for Codex Trajectory Analyzer."""

import argparse
import sys

from parser import load_trajectory
from classifier import normalize_steps
from tree import build_trace_tree, render_trace_tree
from analyzer import score_suspicious_steps, locate_failure
from report import write_outputs


def main():
    parser = argparse.ArgumentParser(
        description='Analyze coding agent trajectories'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to trajectory JSON file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    try:
        # Load trajectory
        print(f"Loading trajectory from {args.input}...")
        task, final_status, steps = load_trajectory(args.input)
        print(f"  Task: {task}")
        print(f"  Final Status: {final_status}")
        print(f"  Steps: {len(steps)}")

        # Normalize steps
        print("Normalizing steps...")
        normalized = normalize_steps(steps)

        # Build trace tree
        print("Building trace tree...")
        tree_nodes = build_trace_tree(normalized)
        tree_md = render_trace_tree(tree_nodes)

        # Score suspicious steps
        print("Scoring suspicious steps...")
        scored = score_suspicious_steps(normalized, task, final_status)

        # Locate failure
        print("Analyzing failure...")
        diagnosis = locate_failure(scored, final_status)

        # Write outputs
        print(f"Writing outputs to {args.output}...")
        write_outputs(args.output, task, final_status, scored, tree_md, diagnosis)

        print("\nAnalysis complete!")
        print(f"  - {args.output}/normalized_steps.json")
        print(f"  - {args.output}/trace_tree.md")
        print(f"  - {args.output}/diagnosis.json")
        print(f"  - {args.output}/diagnosis.md")

        if diagnosis.critical_step:
            print(f"\nCritical Step: {diagnosis.critical_step.step_id}")
            print(f"Error Type: {diagnosis.error_type}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
