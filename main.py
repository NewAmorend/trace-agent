#!/usr/bin/env python3
"""Main CLI for Agent Trajectory Eval."""

import argparse
import sys
from pathlib import Path

from evaluator import discover_inputs, evaluate_file, summarize_batch
from lcb import fetch_problems, run_lcb
from report import write_batch_summary, write_eval_result
from runner import run_codex_trace
from swe import fetch_tasks, load_task, prepare_workspace, run_task


def add_eval_args(
    parser: argparse.ArgumentParser,
    *,
    input_default: str | None = None,
    output_default: str | None = None,
) -> None:
    parser.add_argument(
        '--input',
        default=input_default,
        required=input_default is None,
        help='Path to a Codex JSONL file, or a directory containing trajectory files'
    )
    parser.add_argument(
        '--output',
        default=output_default,
        required=output_default is None,
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


def run_eval_command(args: argparse.Namespace) -> int:
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


def run_lcb_fetch_command(args: argparse.Namespace) -> int:
    try:
        fetch_problems(
            problems_dir=args.problems_dir,
            problems_per_difficulty=args.per_difficulty,
            repo=args.repo,
            split_file=args.split_file,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def run_lcb_run_command(args: argparse.Namespace) -> int:
    try:
        run_lcb(
            problems_dir=args.problems_dir,
            trajectories_dir=args.trajectories_dir,
            difficulty=args.difficulty,
            limit=args.limit,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def run_codex_command(args: argparse.Namespace) -> int:
    try:
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


def run_swe_fetch_command(args: argparse.Namespace) -> int:
    try:
        fetch_tasks(
            tasks_dir=args.tasks_dir,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            limit=args.limit,
            offset=args.offset,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def run_swe_prepare_command(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.instance, args.tasks_dir)
        workspace = prepare_workspace(task, args.worktrees_dir)
        print(f"Workspace: {workspace}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-agent",
        description="Evaluate Codex agent trajectories and run sandbox benchmark traces",
    )
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a Codex JSONL trajectory file or directory",
    )
    add_eval_args(eval_parser)
    eval_parser.set_defaults(func=run_eval_command)

    run_trace_parser = subparsers.add_parser(
        "run",
        help="Run Codex in a sandbox and capture a JSONL trajectory",
    )
    run_trace_parser.add_argument("prompt", help="Task prompt to pass to codex exec")
    run_trace_parser.add_argument("--output", required=True, help="JSONL trajectory output path")
    run_trace_parser.add_argument("-C", "--cwd", default=".", help="Workspace root for Codex")
    run_trace_parser.add_argument(
        "--sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
    )
    run_trace_parser.add_argument("--model", default=None)
    run_trace_parser.add_argument("--profile", default=None)
    run_trace_parser.add_argument(
        "--add-dir",
        action="append",
        default=[],
        help="Additional writable/readable directory for Codex; can be repeated",
    )
    run_trace_parser.add_argument("--full-auto", action="store_true")
    run_trace_parser.add_argument("--skip-git-repo-check", action="store_true")
    run_trace_parser.add_argument("--ephemeral", action="store_true")
    run_trace_parser.add_argument("--timeout", type=int, default=None)
    run_trace_parser.add_argument(
        "--eval-output",
        default=None,
        help="Optional report directory; evaluate the captured trajectory after the run",
    )
    run_trace_parser.add_argument("--quiet", action="store_true")
    run_trace_parser.set_defaults(func=run_codex_command)

    lcb_parser = subparsers.add_parser(
        "lcb",
        help="Fetch LiveCodeBench problems, run Codex, and evaluate trajectories",
    )
    lcb_subparsers = lcb_parser.add_subparsers(dest="lcb_command")

    fetch_parser = lcb_subparsers.add_parser(
        "fetch",
        help="Fetch a small LiveCodeBench sample into data/lcb/problems",
    )
    fetch_parser.add_argument("--problems-dir", default="data/lcb/problems")
    fetch_parser.add_argument("--per-difficulty", type=int, default=3)
    fetch_parser.add_argument("--repo", default="livecodebench/code_generation_lite")
    fetch_parser.add_argument("--split-file", default="test.jsonl")
    fetch_parser.set_defaults(func=run_lcb_fetch_command)

    run_parser = lcb_subparsers.add_parser(
        "run",
        help="Run Codex against fetched LiveCodeBench problems",
    )
    run_parser.add_argument("--problems-dir", default="data/lcb/problems")
    run_parser.add_argument("--trajectories-dir", default="data/lcb/trajectories")
    run_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
    )
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--timeout", type=int, default=300)
    run_parser.set_defaults(func=run_lcb_run_command)

    lcb_eval_parser = lcb_subparsers.add_parser(
        "eval",
        help="Evaluate generated LiveCodeBench trajectories",
    )
    add_eval_args(
        lcb_eval_parser,
        input_default="data/lcb/trajectories",
        output_default="out/lcb_eval",
    )
    lcb_eval_parser.set_defaults(func=run_eval_command)

    swe_parser = subparsers.add_parser(
        "swe",
        help="Fetch and run SWE-bench Lite long-horizon tasks",
    )
    swe_subparsers = swe_parser.add_subparsers(dest="swe_command")

    swe_fetch_parser = swe_subparsers.add_parser(
        "fetch",
        help="Fetch SWE-bench Lite task metadata",
    )
    swe_fetch_parser.add_argument("--tasks-dir", default="data/swe/tasks")
    swe_fetch_parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    swe_fetch_parser.add_argument("--config", default="default")
    swe_fetch_parser.add_argument("--split", default="test")
    swe_fetch_parser.add_argument("--limit", type=int, default=10)
    swe_fetch_parser.add_argument("--offset", type=int, default=0)
    swe_fetch_parser.set_defaults(func=run_swe_fetch_command)

    swe_prepare_parser = swe_subparsers.add_parser(
        "prepare",
        help="Clone a SWE task repo and check out its base commit",
    )
    swe_prepare_parser.add_argument("instance", help="SWE instance id")
    swe_prepare_parser.add_argument("--tasks-dir", default="data/swe/tasks")
    swe_prepare_parser.add_argument("--worktrees-dir", default="data/swe/worktrees")
    swe_prepare_parser.set_defaults(func=run_swe_prepare_command)

    swe_run_parser = swe_subparsers.add_parser(
        "run",
        help="Prepare and run Codex on one SWE-bench Lite task",
    )
    swe_run_parser.add_argument("instance", help="SWE instance id")
    swe_run_parser.add_argument("--tasks-dir", default="data/swe/tasks")
    swe_run_parser.add_argument("--worktrees-dir", default="data/swe/worktrees")
    swe_run_parser.add_argument("--trajectories-dir", default="data/swe/trajectories")
    swe_run_parser.add_argument(
        "--sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
    )
    swe_run_parser.add_argument("--model", default=None)
    swe_run_parser.add_argument("--profile", default=None)
    swe_run_parser.add_argument("--full-auto", action="store_true")
    swe_run_parser.add_argument(
        "--include-test-patch",
        action="store_true",
        help="Include the public regression test patch in the prompt",
    )
    swe_run_parser.add_argument(
        "--apply-tests",
        action="store_true",
        help="Apply the task test patch before running Codex",
    )
    swe_run_parser.add_argument("--timeout", type=int, default=None)
    swe_run_parser.add_argument("--eval-output", default=None)
    swe_run_parser.add_argument("--quiet", action="store_true")
    swe_run_parser.set_defaults(func=run_swe_run_command)

    swe_eval_parser = swe_subparsers.add_parser(
        "eval",
        help="Evaluate generated SWE trajectories",
    )
    add_eval_args(
        swe_eval_parser,
        input_default="data/swe/trajectories",
        output_default="out/swe_eval",
    )
    swe_eval_parser.set_defaults(func=run_eval_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility for `trace-eval --input ... --output ...`.
    if args_list and args_list[0].startswith("-") and args_list[0] not in ("-h", "--help"):
        args_list.insert(0, "eval")

    parser = build_parser()
    args = parser.parse_args(args_list)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
