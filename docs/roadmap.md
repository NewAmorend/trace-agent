# Roadmap to v0.1.0 — real `verified_pass` + calibrated analyzer

## Why this exists

The infrastructure compiles, has 108 passing tests, and supports a Codex /
Claude trajectory eval pipeline plus a SWE-bench Lite runner with hard-reset
workspaces and a verified grader. Two gaps stop this from being a credible
public artifact:

1. The marquee claim — `verified_pass` from a real grader — is `null` on
   every run today. The harness env has no per-task deps installed, so the
   grader returns `collection_error`.
2. The analyzer's risk score is hand-coded heuristics with no calibration
   against ground truth. There is no number anyone can point at.

The single sentence we're working towards:

> Trajectory analyzer for coding agents (Codex / Claude on SWE-bench-Lite).
> Risk score predicts verified-fail at AUC = X.X across N = Y runs.

Until Tier 1 is done, that sentence is unwriteable.

---

## Tier 1 — must-have

- [ ] **T1.1 — Make the SWE-bench grader actually grade.**
  `swe.py` (`grade_task`, new `_install_repo()` helper). Add a best-effort
  `pip install -e .` in the workspace before pytest, with
  `--grader-install-timeout` (default 300 s). On install failure, record
  `grader_status="install_failed"` with stderr captured. Optional stretch:
  `--grader-mode {local,docker}` shim that runs the official
  `princeton-nlp/sweb.eval.x86_64.<instance>` image when Docker is present.
  **Done when:** `python main.py swe run pallets__flask-4045 --agent claude
  --timeout 600` produces a sidecar with `grader_status="ok"` and a
  non-`null` `verified_pass`.

- [ ] **T1.2 — Run ≥ 30 trajectories with mixed outcomes.**
  New `scripts/run_batch.py`. Aim for 4–5 small repos (flask, requests,
  pylint, pytest, sphinx) × 6–8 instances each. Commit only the sidecar
  JSONs (not the trajectories) under `data/swe/runs_v1/`. Target a 30–60 %
  verified pass rate so we have signal in both classes.
  **Done when:** `ls data/swe/runs_v1/*.sidecar.json | wc -l` ≥ 30, with
  at least one `verified_pass=true` and one `verified_pass=false`.

- [ ] **T1.3 — Calibrate the analyzer against `verified_pass`.**
  New `scripts/calibrate.py`. Join each sidecar's `verified_pass` against
  `evaluator.evaluate_file()` output (`risk_level`, `max_suspicious_score`).
  Emit `out/calibration/confusion_matrix.json`, `out/calibration/roc.png`,
  and a one-page `out/calibration/summary.md` (commit the summary).
  **Done when:** AUC > 0.6. If AUC ≤ 0.5, tune `analyzer.PATTERNS` weights
  and re-run.

## Tier 2 — clone-and-believable

- [ ] **T2.1 — GitHub Actions CI.**
  `.github/workflows/test.yml` runs `python -m unittest discover` on push
  and PR, Python 3.10 + 3.11 matrix. Add the badge to the README.

- [ ] **T2.2 — 5-minute quickstart that produces a real result.**
  `docs/demo/` with one full demo set (trajectory + sidecar + eval outputs)
  for a small completed task. `Makefile` with `make demo`. Top of README
  points there.

- [ ] **T2.3 — Differentiate in the README.**
  Replace the "Codex-first" framing — Claude is now first-class. Add a
  two-sentence "Why this vs. existing tools" section naming the
  differentiator: step-level suspicious-pattern scoring with replay-branch
  suggestions.

- [ ] **T2.4 — Cover real Claude tools.**
  `adapters/claude_adapter.py` currently handles ~4 of ~54 Claude Code
  tools. Add explicit handling + fixtures for `Task` (subagents),
  `MultiEdit`, `NotebookEdit`, `Glob`, `apply_patch`-style raw edits.
  Tests in `tests/test_claude_eval.py`.

## Tier 3 — polish

- [ ] **T3.1 — Tag `v0.1.0`** with the calibration numbers in the release
  notes and a link to `out/calibration/summary.md`.

- [ ] **T3.2 — Short writeup** at `docs/findings.md`: one table (confusion
  matrix, AUC), one chart (`roc.png`), one paragraph on what the analyzer
  catches. This is what the resume bullet links to.

- [ ] **T3.3 — Repositioning.** Decide between `agent-trace-eval` vs.
  `swe-bench-trace-analyzer` framing in the README header. Keep the
  package name; only change the marketing.

---

## Critical files (reuse, don't fork)

| File | Existing utility | Used by |
|---|---|---|
| `swe.py` | `grade_task`, `_run_pytest`, `write_sidecar` | T1.1, T1.2 |
| `analyzer.py` | `PATTERNS` (weighted registry) | T1.3 (tune weights) |
| `evaluator.py` | `evaluate_file`, `summarize_batch` | T1.3 |
| `parser.py` | `_load_sidecar`, `_enrich_with_sidecar` | already wired |
| `report.py` | `format_batch_summary_md` | T1.3 (verified-pass section) |

## How to track progress

Tick the boxes above as each item lands. Each tier item gets its own short
plan + PR. The resume bullet is unblocked once Tier 1 is fully checked.
