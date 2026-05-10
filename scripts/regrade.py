#!/usr/bin/env python3
"""Re-grade existing SWE-bench Lite trajectories against verified tests.

Reads each row's sidecar, replays the captured ``agent_diff`` into a
clean checkout of ``base_commit``, runs ``pip install -e .``, and
re-runs FAIL_TO_PASS / PASS_TO_PASS via pytest. Updates each sidecar
in place. Idempotent and resumable.

Usage:
    python scripts/regrade.py --runs-dir data/swe/runs_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swe import load_task, prepare_workspace, regrade_task  # noqa: E402


def update_sidecar(sidecar_path: Path, grade: dict) -> None:
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.update({
        "verified_pass": grade.get("verified_pass"),
        "grader_status": grade.get("grader_status"),
        "grader_message": grade.get("grader_message"),
        "fail_to_pass_results": grade.get("fail_to_pass_results"),
        "pass_to_pass_results": grade.get("pass_to_pass_results"),
        "test_patch_application": grade.get("test_patch_application"),
        "install_status": grade.get("install_status"),
    })
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))


def update_manifest_row(manifest_path: Path, instance_id: str, grade: dict) -> None:
    manifest = json.loads(manifest_path.read_text())
    for row in manifest:
        if row["instance_id"] == instance_id:
            row["verified_pass"] = grade.get("verified_pass")
            row["grader_status"] = grade.get("grader_status")
            break
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="data/swe/runs_v1")
    parser.add_argument("--tasks-dir", default="data/swe/tasks")
    parser.add_argument("--worktrees-dir", default="data/swe/worktrees")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--install-timeout", type=int, default=300)
    parser.add_argument("--grader-timeout", type=int, default=600)
    parser.add_argument(
        "--only-null-verified", action="store_true",
        help="Skip sidecars that already have a non-null verified_pass",
    )
    parser.add_argument(
        "--instances", nargs="*", default=None,
        help="Limit to these instance ids",
    )
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    manifest_path = runs_dir / "_batch.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    if args.instances:
        wanted = set(args.instances)
        manifest = [r for r in manifest if r["instance_id"] in wanted]

    print(f"Re-grading {len(manifest)} instances from {runs_dir}")

    for i, row in enumerate(manifest, start=1):
        instance_id = row["instance_id"]
        sidecar_path = runs_dir / f"{instance_id}.jsonl.sidecar.json"
        if not sidecar_path.exists():
            print(f"  [{i}/{len(manifest)}] {instance_id}: NO sidecar, skip")
            continue
        sidecar = json.loads(sidecar_path.read_text())

        if args.only_null_verified and sidecar.get("verified_pass") is not None:
            print(f"  [{i}/{len(manifest)}] {instance_id}: already verified={sidecar.get('verified_pass')}, skip")
            continue

        try:
            task = load_task(instance_id, args.tasks_dir)
        except FileNotFoundError as exc:
            print(f"  [{i}/{len(manifest)}] {instance_id}: {exc}, skip")
            continue

        # Don't reset here — regrade_task does its own reset.
        workspace = prepare_workspace(task, args.worktrees_dir, reset=False)
        agent_diff = sidecar.get("agent_diff") or ""
        size = len(agent_diff)
        print(f"  [{i}/{len(manifest)}] {instance_id}: regrading (agent_diff={size}B)")
        try:
            grade = regrade_task(
                task,
                workspace,
                agent_diff=agent_diff,
                install=not args.no_install,
                install_timeout=args.install_timeout,
                grader_timeout=args.grader_timeout,
            )
        except Exception as exc:  # never let one bad row kill the whole pass
            grade = {
                "verified_pass": None,
                "grader_status": "exception",
                "grader_message": f"{type(exc).__name__}: {exc}"[:1000],
            }
            print(f"    EXCEPTION: {grade['grader_message']}")

        update_sidecar(sidecar_path, grade)
        update_manifest_row(manifest_path, instance_id, grade)
        verdict = grade.get("verified_pass")
        label = "PASS" if verdict is True else ("FAIL" if verdict is False else "?")
        print(
            f"    -> {label} grader={grade.get('grader_status')} "
            f"install={grade.get('install_status')}"
        )

    final_manifest = json.loads(manifest_path.read_text())
    by_status = Counter(r.get("grader_status") for r in final_manifest)
    by_verified = Counter(r.get("verified_pass") for r in final_manifest)
    print("\n===== Re-grade summary =====")
    print(f"  rows: {len(final_manifest)}")
    print("  grader_status:")
    for s, c in by_status.most_common():
        print(f"    {s}: {c}")
    print("  verified_pass:")
    for v, c in by_verified.most_common():
        label = "true" if v is True else ("false" if v is False else "null")
        print(f"    {label}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
