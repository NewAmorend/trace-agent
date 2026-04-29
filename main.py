#!/usr/bin/env python3
"""Main CLI for Agent Trajectory Eval."""

import argparse
import sys
from pathlib import Path

from evaluator import discover_inputs, evaluate_file, summarize_batch
from report import write_batch_summary, write_eval_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Evaluate Codex agent trajectories'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to a Codex JSONL file, or a directory containing trajectory files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='Return exit code 1 when any evaluated trajectory is medium/high risk or failed'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print errors and final result'
    )

    args = parser.parse_args()

    try:
        inputs = discover_inputs(args.input)
        output_dir = Path(args.output)
        results = []

        if not args.quiet:
            print(f"Evaluating {len(inputs)} Codex trajector{'y' if len(inputs) == 1 else 'ies'}...")

        for input_file in inputs:
            if not args.quiet:
                print(f"  - {input_file}")
            result = evaluate_file(input_file)
            results.append(result)

            if len(inputs) == 1:
                target_dir = output_dir
            else:
                target_dir = output_dir / input_file.stem
            write_eval_result(target_dir, result)

        summary = summarize_batch(results)
        write_batch_summary(output_dir, summary, results)

        if not args.quiet:
            print("\nEvaluation complete!")
            print(f"  Output: {output_dir}")
            print(f"  Total: {summary.total}")
            print(f"  Failed: {summary.failed}")
            print(
                f"  Risk: high={summary.high_risk}, "
                f"medium={summary.medium_risk}, low={summary.low_risk}"
            )

            if len(results) == 1 and results[0].diagnosis.critical_step:
                diagnosis = results[0].diagnosis
                print(f"\nCritical Step: {diagnosis.critical_step.step_id}")
                print(f"Error Type: {diagnosis.error_type}")

        if args.ci and any(
            result.final_status.lower() != 'success'
            or result.metrics.risk_level in ('medium', 'high')
            for result in results
        ):
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
