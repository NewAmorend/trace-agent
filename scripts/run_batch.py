#!/usr/bin/env python3
"""Run a batch of SWE-bench Lite tasks with resume support.

Maintains a manifest JSON next to the trajectories directory. Each row
carries its own status (``pending``, ``done``, ``agent_error``,
``crashed``, ``interrupted``) plus the verified-grade outcome from the
sidecar. Re-running the same command picks up where the prior run left
off — useful when an API rate-limit or session timeout kills the batch
mid-flight.

Usage:
    python scripts/run_batch.py \\
        --instances-file data/swe/runs_v1/instances.txt \\
        --runs-dir data/swe/runs_v1 \\
        --agent claude \\
        --timeout 360
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Allow running as `python scripts/run_batch.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swe import run_task  # noqa: E402


MANIFEST_NAME = "_batch.json"
DONE_STATUSES = {"done"}
RETRY_DEFAULT_STATUSES = {"pending", "crashed", "interrupted"}
RETRY_OPTIONAL_STATUSES = {"agent_error"}


def load_or_init_manifest(manifest_path: Path, instance_ids: list[str]) -> list[dict]:
    """Read an existing manifest and merge any new instance ids; init if absent.

    Existing rows keep their state. New ids are appended in pending state.
    Order is preserved for stable resume behavior.
    """
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        existing = {row["instance_id"]: row for row in manifest}
        for instance_id in instance_ids:
            if instance_id not in existing:
                manifest.append({"instance_id": instance_id, "status": "pending"})
        return manifest
    return [{"instance_id": instance_id, "status": "pending"} for instance_id in instance_ids]


def save_manifest(manifest_path: Path, manifest: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def should_run(row: dict, *, retry_agent_error: bool, force_ids: set[str]) -> bool:
    """Decide whether a row needs to run on this invocation."""
    if row["instance_id"] in force_ids:
        return True
    status = row.get("status", "pending")
    if status in DONE_STATUSES:
        return False
    if status in RETRY_DEFAULT_STATUSES:
        return True
    if status in RETRY_OPTIONAL_STATUSES:
        return retry_agent_error
    # Unknown status -> retry to be safe.
    return True


def classify_post_run(
    row: dict,
    *,
    returncode: int,
    event_count: int,
    sidecar: dict | None,
) -> str:
    """Mark a row as ``done`` or ``agent_error`` after a successful run_task call.

    A row is ``done`` if the agent process exited cleanly OR produced a
    substantive trajectory (>=10 events). Anything else suggests a
    rate-limit, auth, or other early-exit error worth flagging for
    optional retry.
    """
    if returncode == 0:
        return "done"
    if event_count >= 10:
        return "done"
    return "agent_error"


def read_sidecar(trajectory_path: Path) -> dict | None:
    sidecar = trajectory_path.with_suffix(trajectory_path.suffix + ".sidecar.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def run_one(row: dict, *, task_kwargs: dict) -> None:
    """Run a single instance and update ``row`` in place."""
    instance_id = row["instance_id"]
    print(f"\n===== {instance_id} (was: {row.get('status', 'pending')}) =====")
    row["error"] = None
    row.pop("traceback", None)
    try:
        result, _workspace = run_task(instance_id, **task_kwargs)
    except KeyboardInterrupt:
        row["status"] = "interrupted"
        raise
    except Exception as exc:
        row["status"] = "crashed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-1500:]
        traceback.print_exc()
        return

    trajectory_path = Path(task_kwargs["trajectories_dir"]) / f"{instance_id}.jsonl"
    sidecar_data = read_sidecar(trajectory_path)
    row.update({
        "status": classify_post_run(
            row,
            returncode=result.returncode,
            event_count=result.event_count,
            sidecar=sidecar_data,
        ),
        "trajectory_path": str(trajectory_path),
        "returncode": result.returncode,
        "event_count": result.event_count,
        "verified_pass": (sidecar_data or {}).get("verified_pass"),
        "grader_status": (sidecar_data or {}).get("grader_status"),
    })


def summarize(manifest: list[dict]) -> dict:
    """Return a lightweight summary of the manifest's terminal states."""
    by_status: dict[str, int] = {}
    by_verified: dict[str, int] = {"true": 0, "false": 0, "null": 0}
    by_grader: dict[str, int] = {}
    for row in manifest:
        status = row.get("status", "pending")
        by_status[status] = by_status.get(status, 0) + 1
        v = row.get("verified_pass")
        key = "true" if v is True else "false" if v is False else "null"
        by_verified[key] += 1
        g = row.get("grader_status") or "n/a"
        by_grader[g] = by_grader.get(g, 0) + 1
    return {"by_status": by_status, "by_verified": by_verified, "by_grader": by_grader}


def print_summary(manifest: list[dict]) -> None:
    summary = summarize(manifest)
    print("\n===== Batch summary =====")
    print(f"  Rows: {len(manifest)}")
    for status, count in sorted(summary["by_status"].items()):
        print(f"    status={status}: {count}")
    print(
        f"  verified_pass: true={summary['by_verified']['true']} "
        f"false={summary['by_verified']['false']} "
        f"null={summary['by_verified']['null']}"
    )
    print("  grader_status counts:")
    for status, count in sorted(summary["by_grader"].items(), key=lambda x: (-x[1], x[0])):
        print(f"    {status}: {count}")


def load_instance_list(args: argparse.Namespace) -> list[str]:
    instance_ids: list[str] = []
    if args.instances:
        instance_ids.extend(args.instances)
    if args.instances_file:
        text = Path(args.instances_file).read_text()
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                instance_ids.append(line)
    return instance_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="*", default=None, help="Instance ids to run")
    parser.add_argument(
        "--instances-file",
        default=None,
        help="Newline-delimited file of instance ids (# for comments)",
    )
    parser.add_argument("--runs-dir", default="data/swe/runs_v1")
    parser.add_argument("--manifest", default=None,
                        help="Override manifest path (default: <runs-dir>/_batch.json)")
    parser.add_argument("--tasks-dir", default="data/swe/tasks")
    parser.add_argument("--worktrees-dir", default="data/swe/worktrees")
    parser.add_argument("--agent", choices=["codex", "claude"], default="claude")
    parser.add_argument("--timeout", type=int, default=360,
                        help="Per-task agent timeout in seconds (default: 360)")
    parser.add_argument("--grader-timeout", type=int, default=600)
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--no-reset-workspace", action="store_true")
    parser.add_argument("--retry-agent-error", action="store_true",
                        help="Also retry rows in agent_error status")
    parser.add_argument("--force", nargs="*", default=None,
                        help="Force-retry these instance ids regardless of status")
    parser.add_argument("--sleep-between", type=int, default=0,
                        help="Seconds to sleep between rows (helps with rate limits)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run, then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    runs_dir = Path(args.runs_dir)
    manifest_path = Path(args.manifest) if args.manifest else runs_dir / MANIFEST_NAME
    instance_ids = load_instance_list(args)

    if not manifest_path.exists() and not instance_ids:
        print(
            "No manifest found and no --instances / --instances-file given.",
            file=sys.stderr,
        )
        return 2

    manifest = load_or_init_manifest(manifest_path, instance_ids)
    save_manifest(manifest_path, manifest)

    force_ids = set(args.force or [])
    pending = [
        row for row in manifest
        if should_run(row, retry_agent_error=args.retry_agent_error, force_ids=force_ids)
    ]

    print(f"Manifest: {manifest_path} ({len(manifest)} rows)")
    print(f"Already done: {sum(1 for r in manifest if r.get('status') in DONE_STATUSES)}")
    print(f"To run this invocation: {len(pending)}")
    if args.dry_run:
        for row in pending:
            print(f"  - {row['instance_id']} (was: {row.get('status', 'pending')})")
        return 0

    task_kwargs = {
        "tasks_dir": Path(args.tasks_dir),
        "worktrees_dir": Path(args.worktrees_dir),
        "trajectories_dir": runs_dir,
        "agent": args.agent,
        "timeout": args.timeout,
        "grader_timeout": args.grader_timeout,
        "grade": not args.no_grade,
        "reset_workspace": not args.no_reset_workspace,
    }

    try:
        for row in manifest:
            if not should_run(row, retry_agent_error=args.retry_agent_error, force_ids=force_ids):
                continue
            run_one(row, task_kwargs=task_kwargs)
            save_manifest(manifest_path, manifest)
            if args.sleep_between:
                time.sleep(args.sleep_between)
    except KeyboardInterrupt:
        save_manifest(manifest_path, manifest)
        print("\nInterrupted. Manifest saved. Re-run the same command to resume.")
        print_summary(manifest)
        return 130

    save_manifest(manifest_path, manifest)
    print_summary(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
