"""Tests for harchoc.queue_notify — audit log must never contain recipient secrets."""

from __future__ import annotations

import json
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path

from harchoc.queue_notify import (
    DEFAULT_NOTIFY_LOG,
    load_notify_config,
    notify_event,
    notify_matrix_row,
    notify_queue_job,
)


class QueueNotifyTests(unittest.TestCase):
    def test_load_notify_config_from_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            secret = "secret-recipient@example.com"
            (repo / "configs" / "local").mkdir(parents=True)
            (repo / "configs/local/queue_notify.json").write_text(
                json.dumps({"email": secret, "smtp": {"host": "smtp.test", "password": "pw"}}),
                encoding="utf-8",
            )
            cfg = load_notify_config(repo_root=repo)
            assert cfg is not None
            self.assertEqual(cfg.email, secret)
            self.assertEqual(cfg.smtp_password, "pw")

    def test_notify_log_never_contains_email(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "configs" / "local").mkdir(parents=True)
            secret = "hidden@example.com"
            (repo / "configs/local/queue_notify.json").write_text(
                json.dumps({"email": secret}),
                encoding="utf-8",
            )
            captured: list[EmailMessage] = []

            def _fake_sender(_cfg, msg: EmailMessage) -> None:
                captured.append(msg)
                self.assertEqual(msg["To"], secret)

            result = notify_event(
                repo_root=repo,
                event="test_ping",
                subject="test",
                body="hello",
                context={"job_id": "zoo_matrix_p0_5"},
                sender=_fake_sender,
            )
            self.assertTrue(result.get("delivered"))
            log_path = repo / DEFAULT_NOTIFY_LOG
            self.assertTrue(log_path.is_file())
            raw = log_path.read_text(encoding="utf-8")
            self.assertNotIn(secret, raw)
            self.assertNotIn("@", raw)
            row = json.loads(raw.strip().splitlines()[0])
            self.assertEqual(row["event"], "test_ping")
            self.assertEqual(row["job_id"], "zoo_matrix_p0_5")
            self.assertNotIn("email", row)

    def test_notify_queue_job_failed_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "configs" / "local").mkdir(parents=True)
            (repo / "configs/local/queue_notify.json").write_text(
                json.dumps({"email": "a@b.com"}),
                encoding="utf-8",
            )

            def _noop(_cfg, _msg: EmailMessage) -> None:
                pass

            job = {"id": "zoo_matrix_p0_5", "kind": "zoo_matrix_train"}
            notify_queue_job(
                repo_root=repo,
                job=job,
                status="failed",
                stage_id="train",
                exit_code=1,
                hint="OOM",
            )
            log = (repo / DEFAULT_NOTIFY_LOG).read_text(encoding="utf-8")
            self.assertIn("queue_job_failed", log)
            self.assertIn("zoo_matrix_p0_5", log)
            self.assertNotIn("a@b.com", log)

    def test_notify_matrix_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "configs" / "local").mkdir(parents=True)
            (repo / "configs/local/queue_notify.json").write_text(
                json.dumps({"email": "z@y.com"}),
                encoding="utf-8",
            )
            notify_matrix_row(
                repo_root=repo,
                row_name="yolov8m_default",
                status="ok",
                matrix_group="zoo_core_8gb",
                test_count_mae=61.3,
                parent_job_id="zoo_matrix_p0_5",
            )
            log = (repo / DEFAULT_NOTIFY_LOG).read_text(encoding="utf-8")
            self.assertIn("matrix_row_complete", log)
            self.assertIn("yolov8m_default", log)
            self.assertNotIn("z@y.com", log)
            self.assertNotIn("@", log)


if __name__ == "__main__":
    unittest.main()
