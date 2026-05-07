"""SWE-bench Lite helpers for long-horizon agent trajectory capture and grading."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from runner import CodexRunResult, run_claude_trace, run_codex_trace


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "test"
DEFAULT_TASKS_DIR = Path("data/swe/tasks")
DEFAULT_WORKTREES_DIR = Path("data/swe/worktrees")
DEFAULT_TRAJECTORIES_DIR = Path("data/swe/trajectories")

GRADER_TIMEOUT_DEFAULT = 600


def fetch_tasks(
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    *,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    """Fetch SWE-bench Lite task rows via HuggingFace's JSON rows API."""
    target_dir = Path(tasks_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    query = urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": limit,
    })
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    print(f"Loading SWE-bench Lite tasks from {url}...")

    with urlopen(url) as response:
        payload = json.load(response)

    tasks = []
    for item in payload.get("rows", []):
        task = item["row"]
        task["row_idx"] = item.get("row_idx")
        filename = f"{task['instance_id']}.json"
        (target_dir / filename).write_text(
            json.dumps(task, indent=2, ensure_ascii=False)
        )
        tasks.append({
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "filename": filename,
            "row_idx": task.get("row_idx"),
        })
        print(f"  Saved: {task['instance_id']} ({task['repo']})")

    (target_dir / "manifest.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False)
    )
    print(f"\nDone. {len(tasks)} tasks saved to {target_dir.resolve()}")
    return tasks


def load_task(instance_id: str, tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> dict:
    """Load one fetched SWE task by instance id."""
    path = Path(tasks_dir) / f"{instance_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `trace-agent swe fetch` first.")
    with path.open("r") as f:
        return json.load(f)


def prepare_workspace(
    task: dict,
    worktrees_dir: str | Path = DEFAULT_WORKTREES_DIR,
    *,
    reset: bool = True,
) -> Path:
    """Clone the task repo at base_commit. If ``reset`` and a prior worktree
    exists, hard-reset it (or re-clone if reset fails) so the agent always
    starts from a clean ``base_commit`` checkout."""
    worktree = Path(worktrees_dir) / task["instance_id"]

    if worktree.exists():
        if reset:
            ok = _hard_reset_worktree(worktree, task["base_commit"])
            if not ok:
                print(f"Hard reset failed; re-cloning {worktree}")
                shutil.rmtree(worktree, ignore_errors=True)
            else:
                return worktree
        else:
            print(f"Workspace already exists: {worktree}")
            return worktree

    worktree.parent.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{task['repo']}.git"
    print(f"Cloning {repo_url} into {worktree}...")
    subprocess.run(["git", "clone", repo_url, str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "checkout", task["base_commit"]],
        check=True,
    )
    return worktree


def _hard_reset_worktree(worktree: Path, base_commit: str) -> bool:
    """Reset a worktree to ``base_commit`` and remove all untracked files."""
    try:
        subprocess.run(
            ["git", "-C", str(worktree), "reset", "--hard", base_commit],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(worktree), "clean", "-fdx"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_prompt(task: dict, *, include_test_patch: bool = False) -> str:
    """Build a long-horizon Codex prompt for a SWE-bench task."""
    parts = [
        f"SWE-bench Lite task: {task['instance_id']}",
        f"Repository: {task['repo']}",
        f"Base commit: {task['base_commit']}",
        "",
        "Problem statement:",
        task.get("problem_statement", "").strip(),
    ]

    hints = (task.get("hints_text") or "").strip()
    if hints:
        parts.extend(["", "Hints:", hints])

    fail_to_pass = (task.get("FAIL_TO_PASS") or "").strip()
    if fail_to_pass:
        parts.extend(["", "Known fail-to-pass tests:", fail_to_pass])

    if include_test_patch and task.get("test_patch"):
        parts.extend(["", "Reference regression test patch:", task["test_patch"]])

    parts.extend([
        "",
        "Instructions:",
        "- Inspect the repository before editing.",
        "- Fix the root cause in the implementation.",
        "- Add or update tests only when they are needed to verify the behavior.",
        "- Run the most relevant tests you can in this sandbox.",
        "- Stop when the fix is implemented and verification is complete.",
    ])
    return "\n".join(parts)


def apply_test_patch(workspace: str | Path, task: dict) -> Path | None:
    """Apply the task's public regression test patch to the prepared workspace."""
    patch = task.get("test_patch")
    if not patch:
        return None

    workspace_path = Path(workspace)
    patch_path = workspace_path / ".swe_test.patch"
    patch_path.write_text(patch)
    subprocess.run(["git", "-C", str(workspace_path), "apply", str(patch_path)], check=True)
    print(f"Applied test patch: {patch_path}")
    return patch_path


def _parse_test_id_list(value) -> list[str]:
    """SWE-bench stores test ids as either a JSON-encoded string or a list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except json.JSONDecodeError:
            pass
        return [text]
    return []


def grade_task(
    task: dict,
    workspace: str | Path,
    *,
    timeout: int = GRADER_TIMEOUT_DEFAULT,
    apply_test_patch_first: bool = True,
) -> dict:
    """Run FAIL_TO_PASS and PASS_TO_PASS tests, return a verified-grade dict.

    The returned dict has keys:
      - verified_pass:          bool | None
      - grader_status:          'ok' | 'collection_error' | 'timeout' | 'no_tests' | 'install_error'
      - grader_message:         str
      - fail_to_pass_results:   {passed:[], failed:[], errors:[], skipped:[]}
      - pass_to_pass_results:   same shape
      - agent_diff:             str   (git diff vs base_commit)
    """
    workspace_path = Path(workspace)
    base_commit = task.get("base_commit", "")
    fail_to_pass = _parse_test_id_list(task.get("FAIL_TO_PASS"))
    pass_to_pass = _parse_test_id_list(task.get("PASS_TO_PASS"))

    agent_diff = ""
    if base_commit:
        try:
            diff_proc = subprocess.run(
                ["git", "-C", str(workspace_path), "diff", base_commit, "--", ":(exclude).swe_test.patch"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            agent_diff = diff_proc.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            agent_diff = ""

    if apply_test_patch_first and task.get("test_patch"):
        test_patch_status = _ensure_test_patch_applied(workspace_path, task["test_patch"])
    else:
        test_patch_status = "skipped"

    result: dict = {
        "verified_pass": None,
        "grader_status": "skipped",
        "grader_message": "",
        "fail_to_pass_results": None,
        "pass_to_pass_results": None,
        "agent_diff": agent_diff,
        "test_patch_application": test_patch_status,
    }

    if not fail_to_pass and not pass_to_pass:
        result["grader_status"] = "no_tests"
        result["grader_message"] = "No FAIL_TO_PASS / PASS_TO_PASS test ids in task."
        return result

    f2p = _run_pytest(workspace_path, fail_to_pass, timeout=timeout) if fail_to_pass else _empty_pytest_result()
    p2p = _run_pytest(workspace_path, pass_to_pass, timeout=timeout) if pass_to_pass else _empty_pytest_result()
    result["fail_to_pass_results"] = f2p
    result["pass_to_pass_results"] = p2p

    statuses = {f2p["status"], p2p["status"]} - {"skipped"}
    if statuses == {"ok"}:
        passed_all_f2p = bool(fail_to_pass) and not f2p["failed"] and not f2p["errors"] and len(f2p["passed"]) == len(fail_to_pass)
        passed_all_p2p = (not pass_to_pass) or (not p2p["failed"] and not p2p["errors"] and len(p2p["passed"]) == len(pass_to_pass))
        if not fail_to_pass:
            passed_all_f2p = True
        result["verified_pass"] = bool(passed_all_f2p and passed_all_p2p)
        result["grader_status"] = "ok"
    elif "collection_error" in statuses:
        result["grader_status"] = "collection_error"
        result["grader_message"] = (f2p.get("message") or p2p.get("message") or "")[:1000]
    elif "timeout" in statuses:
        result["grader_status"] = "timeout"
    else:
        result["grader_status"] = next(iter(statuses), "skipped")

    return result


def _empty_pytest_result() -> dict:
    return {
        "status": "skipped",
        "passed": [],
        "failed": [],
        "errors": [],
        "skipped": [],
        "returncode": None,
        "message": "",
    }


def _ensure_test_patch_applied(workspace: Path, patch_text: str) -> str:
    """Apply the task's test patch if it is not already applied. Best-effort."""
    patch_path = workspace / ".swe_test.patch"
    patch_path.write_text(patch_text)
    check = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--check", str(patch_path)],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        return "already_applied_or_conflict"
    apply_proc = subprocess.run(
        ["git", "-C", str(workspace), "apply", str(patch_path)],
        capture_output=True,
        text=True,
    )
    if apply_proc.returncode == 0:
        return "applied"
    return f"apply_failed: {apply_proc.stderr.strip()[:200]}"


def _run_pytest(workspace: Path, test_ids: list[str], *, timeout: int) -> dict:
    """Run a list of pytest node ids and return a structured result."""
    if not test_ids:
        return _empty_pytest_result()

    junit = workspace / ".trace_agent_junit.xml"
    if junit.exists():
        junit.unlink()

    cmd = [
        sys.executable, "-m", "pytest",
        "--no-header", "-q", "--tb=short",
        f"--junitxml={junit}",
    ] + test_ids

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "passed": [], "failed": [], "errors": [], "skipped": [],
            "returncode": None,
            "message": f"pytest timed out after {timeout}s; partial stdout: {(exc.stdout or '')[-500:]}",
        }
    except FileNotFoundError:
        return {
            "status": "install_error",
            "passed": [], "failed": [], "errors": [], "skipped": [],
            "returncode": None,
            "message": "python -m pytest not available in this environment.",
        }

    if not junit.exists():
        return {
            "status": "collection_error",
            "passed": [], "failed": [], "errors": [], "skipped": [],
            "returncode": proc.returncode,
            "message": (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip(),
        }

    try:
        tree = ET.parse(junit)
    except ET.ParseError as exc:
        return {
            "status": "collection_error",
            "passed": [], "failed": [], "errors": [], "skipped": [],
            "returncode": proc.returncode,
            "message": f"Could not parse junit xml: {exc}",
        }

    passed, failed, errors, skipped = [], [], [], []
    for tc in tree.iter("testcase"):
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        node = f"{classname}::{name}" if classname else name
        if tc.find("failure") is not None:
            failed.append(node)
        elif tc.find("error") is not None:
            errors.append(node)
        elif tc.find("skipped") is not None:
            skipped.append(node)
        else:
            passed.append(node)

    junit.unlink(missing_ok=True)

    return {
        "status": "ok",
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "returncode": proc.returncode,
        "message": "",
    }


def write_sidecar(
    trajectory_path: str | Path,
    task: dict,
    grade: dict | None,
) -> Path:
    """Write a `<trajectory>.sidecar.json` next to the trajectory."""
    trajectory_path = Path(trajectory_path)
    sidecar_path = trajectory_path.with_suffix(trajectory_path.suffix + ".sidecar.json")
    payload = {
        "instance_id": task.get("instance_id"),
        "repo": task.get("repo"),
        "base_commit": task.get("base_commit"),
        "problem_statement": task.get("problem_statement"),
        "FAIL_TO_PASS": task.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": task.get("PASS_TO_PASS"),
    }
    if grade:
        payload.update({
            "verified_pass": grade.get("verified_pass"),
            "grader_status": grade.get("grader_status"),
            "grader_message": grade.get("grader_message"),
            "fail_to_pass_results": grade.get("fail_to_pass_results"),
            "pass_to_pass_results": grade.get("pass_to_pass_results"),
            "agent_diff": grade.get("agent_diff"),
            "test_patch_application": grade.get("test_patch_application"),
        })
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote sidecar: {sidecar_path}")
    return sidecar_path


def run_task(
    instance_id: str,
    *,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
    worktrees_dir: str | Path = DEFAULT_WORKTREES_DIR,
    trajectories_dir: str | Path = DEFAULT_TRAJECTORIES_DIR,
    sandbox: str = "workspace-write",
    model: str | None = None,
    profile: str | None = None,
    full_auto: bool = False,
    include_test_patch: bool = False,
    apply_tests: bool = False,
    timeout: int | None = None,
    eval_output: str | Path | None = None,
    agent: str = "codex",
    reset_workspace: bool = True,
    grade: bool = True,
    grader_timeout: int = GRADER_TIMEOUT_DEFAULT,
) -> tuple[CodexRunResult, Path]:
    """Prepare a SWE workspace, run an agent, capture the trajectory, and grade.

    Always hard-resets the workspace to ``base_commit`` before the agent runs
    (set ``reset_workspace=False`` to disable). After the run, executes the
    task's FAIL_TO_PASS and PASS_TO_PASS tests against the agent's working
    tree and writes a sidecar JSON next to the trajectory containing the
    verified outcome and task metadata.
    """
    task = load_task(instance_id, tasks_dir)
    workspace = prepare_workspace(task, worktrees_dir, reset=reset_workspace)
    if apply_tests:
        apply_test_patch(workspace, task)

    prompt = build_prompt(task, include_test_patch=include_test_patch)
    trajectory_path = Path(trajectories_dir) / f"{instance_id}.jsonl"

    timed_out = False
    try:
        if agent == "claude":
            result = run_claude_trace(
                prompt,
                output=trajectory_path,
                cwd=workspace,
                model=model,
                timeout=timeout,
            )
        else:
            result = run_codex_trace(
                prompt,
                output=trajectory_path,
                cwd=workspace,
                sandbox=sandbox,
                model=model,
                profile=profile,
                full_auto=full_auto,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        # Runner already wrote whatever it captured into trajectory_path.
        timed_out = True
        if trajectory_path.exists():
            event_count = sum(1 for line in trajectory_path.read_text().splitlines() if line.strip())
        else:
            event_count = 0
        result = CodexRunResult(
            trajectory_path=trajectory_path,
            returncode=124,
            event_count=event_count,
            stderr="agent run timed out",
        )
        print(f"WARNING: agent timed out after {timeout}s; recovered {event_count} events.")

    grade_result: dict | None = None
    if grade:
        try:
            grade_result = grade_task(task, workspace, timeout=grader_timeout)
        except Exception as exc:  # never let grading failure poison the run
            grade_result = {
                "verified_pass": None,
                "grader_status": "exception",
                "grader_message": f"{type(exc).__name__}: {exc}",
            }
        if grade_result is not None:
            if timed_out:
                grade_result.setdefault("grader_message", "")
                grade_result["grader_message"] = (
                    "agent_timed_out; " + (grade_result.get("grader_message") or "")
                ).strip("; ")
            verdict = grade_result.get("verified_pass")
            label = "PASS" if verdict is True else ("FAIL" if verdict is False else "?")
            print(
                f"Grader: verified_pass={verdict} ({label}) "
                f"status={grade_result.get('grader_status')}"
            )

    write_sidecar(trajectory_path, task, grade_result)
    return result, workspace
