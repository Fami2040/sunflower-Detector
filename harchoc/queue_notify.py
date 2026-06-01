"""GPU queue email notifications — recipient lives in gitignored local config or env only."""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable

DEFAULT_NOTIFY_LOG = "reports/gpu_queue/notify_log.jsonl"
DEFAULT_LOCAL_CONFIG = "configs/local/queue_notify.json"
_SENSITIVE_KEYS = frozenset({"email", "password", "smtp_password", "user", "smtp_user"})


@dataclass(frozen=True)
class NotifyConfig:
    email: str
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    from_addr: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_notify_config(*, repo_root: Path) -> NotifyConfig | None:
    """Resolve notify settings; never logs or returns secrets to callers that persist them."""
    email = (os.environ.get("HARCHOC_NOTIFY_EMAIL") or "").strip()
    smtp_host = (os.environ.get("HARCHOC_NOTIFY_SMTP_HOST") or "").strip() or None
    smtp_port = int(os.environ.get("HARCHOC_NOTIFY_SMTP_PORT") or "587")
    smtp_user = (os.environ.get("HARCHOC_NOTIFY_SMTP_USER") or "").strip() or None
    smtp_password = os.environ.get("HARCHOC_NOTIFY_SMTP_PASSWORD") or None
    smtp_use_tls = os.environ.get("HARCHOC_NOTIFY_SMTP_TLS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    smtp_use_ssl = os.environ.get("HARCHOC_NOTIFY_SMTP_SSL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    from_addr = (os.environ.get("HARCHOC_NOTIFY_FROM") or "").strip() or None

    local_path = repo_root / DEFAULT_LOCAL_CONFIG
    if local_path.is_file():
        try:
            obj = _read_json(local_path)
            if not email:
                email = str(obj.get("email") or "").strip()
            smtp = obj.get("smtp") if isinstance(obj.get("smtp"), dict) else {}
            if isinstance(smtp, dict):
                smtp_host = smtp_host or (str(smtp.get("host") or "").strip() or None)
                if smtp.get("port") is not None:
                    smtp_port = int(smtp["port"])
                smtp_user = smtp_user or (str(smtp.get("user") or "").strip() or None)
                smtp_password = smtp_password or smtp.get("password")
                if smtp.get("use_tls") is not None:
                    smtp_use_tls = bool(smtp["use_tls"])
                if smtp.get("use_ssl") is not None:
                    smtp_use_ssl = bool(smtp["use_ssl"])
                elif int(smtp.get("port") or smtp_port) == 465:
                    smtp_use_ssl = True
                    smtp_use_tls = False
            from_addr = from_addr or (str(obj.get("from") or "").strip() or None)
        except Exception:
            pass

    if not email:
        return None
    return NotifyConfig(
        email=email,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user or email,
        smtp_password=smtp_password,
        smtp_use_tls=smtp_use_tls,
        smtp_use_ssl=smtp_use_ssl,
        from_addr=from_addr or smtp_user or email,
    )


def notify_enabled(*, repo_root: Path) -> bool:
    if os.environ.get("HARCHOC_NOTIFY_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return False
    return load_notify_config(repo_root=repo_root) is not None


def _append_notify_log(repo_root: Path, record: dict[str, Any]) -> None:
    """Append audit record — must not contain email, passwords, or message bodies."""
    safe = {k: v for k, v in record.items() if k not in _SENSITIVE_KEYS}
    for k, v in list(safe.items()):
        if isinstance(v, str) and "@" in v and k not in ("job_id", "row_name", "matrix_group"):
            # Drop stray address-like strings except known id fields.
            if k.endswith("_email") or k in ("to", "recipient", "from"):
                safe.pop(k, None)
    log_path = repo_root / DEFAULT_NOTIFY_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe, sort_keys=True) + "\n")


def _send_email(
    cfg: NotifyConfig,
    *,
    subject: str,
    body: str,
    sender: Callable[[NotifyConfig, EmailMessage], None] | None = None,
) -> tuple[bool, str | None]:
    if os.environ.get("HARCHOC_NOTIFY_DRY_RUN", "").strip().lower() in ("1", "true", "yes"):
        return True, "dry_run"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.from_addr or cfg.smtp_user or cfg.email
    msg["To"] = cfg.email
    msg.set_content(body)

    if sender is not None:
        sender(cfg, msg)
        return True, None

    if not cfg.smtp_host:
        return False, "smtp_not_configured"

    try:
        ctx = ssl.create_default_context()
        if cfg.smtp_use_ssl:
            with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=30, context=ctx) as smtp:
                if cfg.smtp_password:
                    smtp.login(cfg.smtp_user or cfg.email, cfg.smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as smtp:
                if cfg.smtp_use_tls:
                    smtp.starttls(context=ctx)
                if cfg.smtp_password:
                    smtp.login(cfg.smtp_user or cfg.email, cfg.smtp_password)
                smtp.send_message(msg)
        return True, None
    except Exception as ex:
        return False, str(ex)


def notify_event(
    *,
    repo_root: Path,
    event: str,
    subject: str,
    body: str,
    context: dict[str, Any] | None = None,
    dry_run: bool = False,
    sender: Callable[[NotifyConfig, EmailMessage], None] | None = None,
) -> dict[str, Any]:
    """Send notification if configured; always append sanitized audit log when enabled."""
    rr = repo_root.resolve()
    cfg = load_notify_config(repo_root=rr)
    env_dry = os.environ.get("HARCHOC_NOTIFY_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
    effective_dry = dry_run or env_dry
    base: dict[str, Any] = {
        "ts": _utc_now(),
        "event": event,
        "delivered": False,
        "dry_run": effective_dry,
    }
    if context:
        base.update({k: v for k, v in context.items() if k not in _SENSITIVE_KEYS})

    if cfg is None:
        base["skipped"] = "notify_not_configured"
        return base

    if effective_dry:
        base["delivered"] = True
        base["channel"] = "dry_run"
        _append_notify_log(rr, base)
        return base

    ok, err = _send_email(cfg, subject=subject, body=body, sender=sender)
    base["delivered"] = ok
    if err:
        base["delivery_error"] = err
    _append_notify_log(rr, base)
    return base


def notify_queue_job(
    *,
    repo_root: Path,
    job: dict[str, Any],
    status: str,
    dry_run: bool = False,
    stage_id: str | None = None,
    exit_code: int | None = None,
    hint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job_id = str(job.get("id") or "")
    kind = str(job.get("kind") or "")
    event = "queue_job_complete" if status == "complete" else "queue_job_failed"
    subject = f"[harchoc] {job_id} {status}"
    lines = [
        f"job_id: {job_id}",
        f"kind: {kind}",
        f"status: {status}",
    ]
    if stage_id:
        lines.append(f"stage: {stage_id}")
    if exit_code is not None:
        lines.append(f"exit_code: {exit_code}")
    if hint:
        lines.append(f"hint: {hint}")
    if extra:
        for k, v in extra.items():
            if k not in _SENSITIVE_KEYS:
                lines.append(f"{k}: {v}")
    ctx: dict[str, Any] = {"job_id": job_id, "kind": kind, "status": status}
    if stage_id:
        ctx["stage_id"] = stage_id
    if exit_code is not None:
        ctx["exit_code"] = exit_code
    if extra:
        ctx.update({k: v for k, v in extra.items() if k not in _SENSITIVE_KEYS})
    return notify_event(
        repo_root=repo_root,
        event=event,
        subject=subject,
        body="\n".join(lines),
        context=ctx,
        dry_run=dry_run,
    )


def notify_matrix_row(
    *,
    repo_root: Path,
    row_name: str,
    status: str,
    matrix_group: str | None = None,
    test_count_mae: float | None = None,
    detail: str | None = None,
    parent_job_id: str | None = None,
) -> dict[str, Any]:
    event = "matrix_row_complete" if status in ("ok", "complete", "skipped") else "matrix_row_failed"
    subject = f"[harchoc] zoo row {row_name} {status} (queue continues)"
    lines = [f"row: {row_name}", f"status: {status}"]
    if matrix_group:
        lines.append(f"group: {matrix_group}")
    if test_count_mae is not None:
        lines.append(f"test_count_mae: {test_count_mae:.3f}")
    if detail:
        lines.append(f"detail: {detail}")
    if parent_job_id:
        lines.append(f"parent_job: {parent_job_id}")
    ctx: dict[str, Any] = {"row_name": row_name, "status": status}
    if matrix_group:
        ctx["matrix_group"] = matrix_group
    if test_count_mae is not None:
        ctx["test_count_mae"] = test_count_mae
    if parent_job_id:
        ctx["parent_job_id"] = parent_job_id
    return notify_event(
        repo_root=repo_root,
        event=event,
        subject=subject,
        body="\n".join(lines),
        context=ctx,
        dry_run=False,
    )


def notify_queue_manifest_complete(
    *,
    repo_root: Path,
    manifest_path: str,
    completed: list[str],
    skipped: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    subject = "[harchoc] GPU queue manifest complete"
    body = "\n".join(
        [
            f"manifest: {manifest_path}",
            f"completed: {len(completed)}",
            f"skipped: {len(skipped)}",
            "jobs: " + ", ".join(completed[:20]) + ("…" if len(completed) > 20 else ""),
        ]
    )
    return notify_event(
        repo_root=repo_root,
        event="queue_manifest_complete",
        subject=subject,
        body=body,
        context={
            "manifest": manifest_path,
            "completed_count": len(completed),
            "skipped_count": len(skipped),
        },
        dry_run=dry_run,
    )
