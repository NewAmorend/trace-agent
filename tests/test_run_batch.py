"""Unit tests for scripts/run_batch.py — manifest, resume, classification."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# scripts/ isn't on sys.path by default; make the module importable.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_batch  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_init_creates_pending_rows_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_batch.json"
            manifest = run_batch.load_or_init_manifest(path, ["a", "b", "c"])
        self.assertEqual([row["instance_id"] for row in manifest], ["a", "b", "c"])
        for row in manifest:
            self.assertEqual(row["status"], "pending")

    def test_existing_rows_preserved_on_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_batch.json"
            path.write_text(json.dumps([
                {"instance_id": "a", "status": "done", "verified_pass": True},
                {"instance_id": "b", "status": "crashed", "error": "boom"},
            ]))
            merged = run_batch.load_or_init_manifest(path, ["b", "c"])
        # a + b stay as-is; c appended pending
        self.assertEqual([row["instance_id"] for row in merged], ["a", "b", "c"])
        self.assertEqual(merged[0]["status"], "done")
        self.assertTrue(merged[0]["verified_pass"])
        self.assertEqual(merged[1]["status"], "crashed")
        self.assertEqual(merged[1]["error"], "boom")
        self.assertEqual(merged[2]["status"], "pending")

    def test_save_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deeper" / "_batch.json"
            rows = [{"instance_id": "x", "status": "done"}]
            run_batch.save_manifest(path, rows)
            self.assertEqual(json.loads(path.read_text()), rows)


class ResumePolicyTests(unittest.TestCase):
    def _row(self, status, instance_id="x"):
        return {"instance_id": instance_id, "status": status}

    def test_done_skipped_by_default(self):
        self.assertFalse(run_batch.should_run(self._row("done"), retry_agent_error=False, force_ids=set()))

    def test_pending_runs(self):
        self.assertTrue(run_batch.should_run(self._row("pending"), retry_agent_error=False, force_ids=set()))

    def test_crashed_runs_by_default(self):
        self.assertTrue(run_batch.should_run(self._row("crashed"), retry_agent_error=False, force_ids=set()))

    def test_interrupted_runs_by_default(self):
        self.assertTrue(run_batch.should_run(self._row("interrupted"), retry_agent_error=False, force_ids=set()))

    def test_agent_error_only_with_flag(self):
        row = self._row("agent_error")
        self.assertFalse(run_batch.should_run(row, retry_agent_error=False, force_ids=set()))
        self.assertTrue(run_batch.should_run(row, retry_agent_error=True, force_ids=set()))

    def test_force_overrides_done(self):
        self.assertTrue(run_batch.should_run(
            self._row("done", "abc"), retry_agent_error=False, force_ids={"abc"},
        ))


class ClassificationTests(unittest.TestCase):
    def _classify(self, *, returncode, event_count, sidecar=None):
        row = {"instance_id": "demo"}
        return run_batch.classify_post_run(
            row, returncode=returncode, event_count=event_count, sidecar=sidecar,
        )

    def test_clean_exit_is_done(self):
        self.assertEqual(self._classify(returncode=0, event_count=42), "done")

    def test_substantive_trajectory_is_done_even_if_nonzero(self):
        # Timeout returns 124 but produced a real trajectory — still useful.
        self.assertEqual(self._classify(returncode=124, event_count=80), "done")

    def test_short_failed_run_marked_agent_error(self):
        # Likely a rate-limit or auth error: agent died before doing real work.
        self.assertEqual(self._classify(returncode=1, event_count=2), "agent_error")


class SummaryTests(unittest.TestCase):
    def test_summary_counts_status_and_verified(self):
        manifest = [
            {"instance_id": "a", "status": "done", "verified_pass": True, "grader_status": "ok"},
            {"instance_id": "b", "status": "done", "verified_pass": False, "grader_status": "ok"},
            {"instance_id": "c", "status": "done", "verified_pass": None, "grader_status": "collection_error"},
            {"instance_id": "d", "status": "crashed"},
            {"instance_id": "e", "status": "pending"},
        ]
        summary = run_batch.summarize(manifest)
        self.assertEqual(summary["by_status"]["done"], 3)
        self.assertEqual(summary["by_status"]["crashed"], 1)
        self.assertEqual(summary["by_status"]["pending"], 1)
        self.assertEqual(summary["by_verified"]["true"], 1)
        self.assertEqual(summary["by_verified"]["false"], 1)
        self.assertEqual(summary["by_verified"]["null"], 3)
        self.assertEqual(summary["by_grader"]["ok"], 2)
        self.assertEqual(summary["by_grader"]["collection_error"], 1)


class CLITests(unittest.TestCase):
    def test_dry_run_prints_pending_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            instances = runs_dir / "ids.txt"
            instances.write_text("a\nb\n# comment line\nc\n")
            rc = run_batch.main([
                "--instances-file", str(instances),
                "--runs-dir", str(runs_dir),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            manifest_path = runs_dir / run_batch.MANIFEST_NAME
            self.assertTrue(manifest_path.exists())
            rows = json.loads(manifest_path.read_text())
            self.assertEqual([r["instance_id"] for r in rows], ["a", "b", "c"])

    def test_no_inputs_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = run_batch.main(["--runs-dir", tmp])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
