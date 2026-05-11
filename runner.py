"""Run Codex in a configured sandbox and capture JSONL trajectories."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodexRunResult:
    """Result metadata for a captured Codex run."""
    trajectory_path: Path
    returncode: int
    event_count: int
    stderr: str = ""


def build_codex_exec_command(
    prompt: str,
    *,
    cwd: str | Path,
    sandbox: str = "workspace-write",
    model: str | None = None,
    profile: str | None = None,
    add_dirs: list[str] | None = None,
    full_auto: bool = False,
    skip_git_repo_check: bool = False,
    ephemeral: bool = False,
) -> list[str]:
    """Build the codex exec command used by the runner."""
    command = ["codex", "exec", "--json", "-C", str(cwd), "--sandbox", sandbox]

    if model:
        command.extend(["--model", model])
    if profile:
        command.extend(["--profile", profile])
    for directory in add_dirs or []:
        command.extend(["--add-dir", directory])
    if full_auto:
        command.append("--full-auto")
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if ephemeral:
        command.append("--ephemeral")

    command.append(prompt)
    return command


def run_codex_trace(
    prompt: str,
    *,
    output: str | Path,
    cwd: str | Path = ".",
    sandbox: str = "workspace-write",
    model: str | None = None,
    profile: str | None = None,
    add_dirs: list[str] | None = None,
    full_auto: bool = False,
    skip_git_repo_check: bool = False,
    ephemeral: bool = False,
    timeout: int | None = None,
) -> CodexRunResult:
    """Run Codex and write its JSONL event stream to output."""
    command = build_codex_exec_command(
        prompt,
        cwd=cwd,
        sandbox=sandbox,
        model=model,
        profile=profile,
        add_dirs=add_dirs,
        full_auto=full_auto,
        skip_git_repo_check=skip_git_repo_check,
        ephemeral=ephemeral,
    )
    print(f"Running Codex in {Path(cwd).resolve()} with sandbox={sandbox}...")
    return _run_and_capture(command, output=output, cwd=cwd, timeout=timeout, cli_name="codex")


def build_claude_exec_command(
    prompt: str,
    *,
    model: str | None = None,
) -> list[str]:
    """Build the claude command used by the Claude runner."""
    command = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]
    if model:
        command.extend(["--model", model])
    return command


def run_claude_trace(
    prompt: str,
    *,
    output: str | Path,
    cwd: str | Path = ".",
    model: str | None = None,
    timeout: int | None = None,
) -> CodexRunResult:
    """Run Claude Code CLI and write its JSONL event stream to output."""
    command = build_claude_exec_command(prompt, model=model)
    print(f"Running Claude Code in {Path(cwd).resolve()}...")
    return _run_and_capture(command, output=output, cwd=cwd, timeout=timeout, cli_name="claude")


def _run_and_capture(
    command: list[str],
    *,
    output: str | Path,
    cwd: str | Path,
    timeout: int | None,
    cli_name: str,
) -> CodexRunResult:
    """Shared subprocess runner for agent CLIs."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except FileNotFoundError:
        print(f"ERROR: '{cli_name}' CLI not found. Is it installed?", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            output_path.write_text(_as_text(exc.stdout))
        raise

    output_path.write_text(result.stdout)
    event_count = len([line for line in result.stdout.splitlines() if line.strip()])

    if result.stderr:
        stderr_path = output_path.with_suffix(output_path.suffix + ".stderr")
        stderr_path.write_text(result.stderr)
        print(f"{cli_name} stderr saved to {stderr_path}")

    print(f"Saved {event_count} events to {output_path}")
    if result.returncode != 0:
        print(f"WARNING: {cli_name} exited with {result.returncode}")

    return CodexRunResult(
        trajectory_path=output_path,
        returncode=result.returncode,
        event_count=event_count,
        stderr=result.stderr,
    )


def _as_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
