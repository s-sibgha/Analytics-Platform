"""
utils/error_logging.py — Centralized, Silent, Server-Side Error Telemetry
(NEW MODULE)

Provides a single production-grade entry point, `log_exception`, that every
try/except block across the platform's frontend (components/, pages/) and
backend (core/, engine/, visualization/) layers can call to persist a full,
structured, correlatable error record to a rotating log file plus stderr —
without ever surfacing internals to the end user and without ever raising
itself, per the platform's "never crash the UI" mandate.

This module is intentionally dependency-light (stdlib `logging` only, no
Streamlit import) so it can be safely imported from any layer, including
core/engine modules that must remain UI-framework-agnostic.

Design:
    - A single, lazily-initialized, process-wide logger named
      "kesco_platform" backed by a RotatingFileHandler (logs/platform.log,
      5 MB x 5 backups) plus a StreamHandler at WARNING+ for console
      visibility during local development.
    - Every call to `log_exception` is assigned a short, correlatable
      incident_id (UUID4, first 8 hex chars) that the caller may optionally
      surface to the end user (e.g. "Reference ID: a1b2c3d4") so a support
      engineer can grep the log file for the exact incident without ever
      exposing a stack trace or internal detail in the UI.
    - `log_exception` NEVER raises: a failure inside the logging pipeline
      itself (e.g. an unwritable log directory) is silently swallowed after
      one best-effort fallback to stderr, consistent with the platform's
      absolute "never propagate an unhandled exception to the UI" contract.
"""
from __future__ import annotations

import logging
import sys
import traceback
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

_LOGGER_NAME: str = "kesco_platform"
_LOG_DIR: Path = Path("logs")
_LOG_FILE: Path = _LOG_DIR / "platform.log"
_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per rotated file
_BACKUP_COUNT: int = 5

_VALID_SEVERITIES: frozenset = frozenset({"debug", "info", "warning", "error", "critical"})

_logger: Optional[logging.Logger] = None


def _initialize_logger() -> logging.Logger:
    """
    Lazily constructs and caches the process-wide "kesco_platform" logger on
    first use. Idempotent — safe to call from every module/thread without
    duplicating handlers across Streamlit reruns. Never raises: if the log
    directory/file cannot be created (e.g. read-only filesystem in a
    constrained container), the logger degrades to a console-only handler
    rather than failing to initialize.
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=str(_LOG_FILE),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:  # noqa: BLE001
            # Filesystem unavailable/read-only — degrade to console-only
            # rather than raising during logger initialization.
            pass

        try:
            console_handler = logging.StreamHandler(stream=sys.stderr)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception:  # noqa: BLE001
            pass

    _logger = logger
    return _logger


def _new_incident_id() -> str:
    """Generates a short, human-shareable correlation identifier (first 8
    hex characters of a UUID4) suitable for display in a support-facing
    error message without exposing internal detail. Never raises."""
    try:
        return uuid.uuid4().hex[:8]
    except Exception:  # noqa: BLE001
        return "00000000"


def log_exception(
    component: str,
    exc: BaseException,
    *,
    severity: str = "error",
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Centralized, silent, server-side exception logger. Every try/except
    block across the platform (frontend components, pages, and backend
    engines) should call this from its `except Exception as exc:` clause in
    addition to whatever graceful degradation (st.warning/st.caption/
    structured-fallback-metadata) it already performs, so that production
    defects leave a durable, correlatable server-side trace instead of
    vanishing into a single user-facing string.

    This function NEVER raises. A failure inside the logging pipeline
    itself is caught and, as an absolute last resort, written directly to
    stderr — it can never become the cause of a second, masking exception
    inside an except block that is itself the platform's last line of
    defense against crashing the UI.

    Args:
        component: A short, stable identifier of the call site (e.g.
            "sidebar.activate_staged_file", "dashboard.render_chart_row",
            "schema_mapping.apply_type_override"). Used to group and filter
            log entries by subsystem.
        exc: The caught exception instance. Its type, message, and full
            traceback are captured.
        severity: One of "debug" | "info" | "warning" | "error" |
            "critical" (case-insensitive). Defaults to "error". An
            unrecognized value degrades to "error" rather than raising.
        context: Optional structured metadata relevant to the failure
            (e.g. {"filename": ..., "workspace": ..., "chart_type": ...}).
            Values are coerced to `str` defensively so a non-serializable
            object passed here can never itself cause a logging failure.

    Returns:
        A short incident_id (str) that the caller may optionally surface
        in the user-facing message (e.g. "Reference ID: a1b2c3d4") so a
        support engineer can grep logs/platform.log for the exact incident
        without the UI ever displaying a raw stack trace.
    """
    incident_id = _new_incident_id()
    try:
        logger = _initialize_logger()

        severity_normalized = severity.strip().lower() if isinstance(severity, str) else "error"
        if severity_normalized not in _VALID_SEVERITIES:
            severity_normalized = "error"
        log_level = getattr(logging, severity_normalized.upper(), logging.ERROR)

        safe_context: Dict[str, str] = {}
        if context:
            for key, value in context.items():
                try:
                    safe_context[str(key)] = str(value)
                except Exception:  # noqa: BLE001
                    safe_context[str(key)] = "<unrepresentable>"

        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

        message = (
            f"incident_id={incident_id} | component={component} | "
            f"exception_type={type(exc).__name__} | message={exc} | "
            f"context={safe_context}\n{traceback_text}"
        )
        logger.log(log_level, message)
        return incident_id
    except Exception as logging_failure:  # noqa: BLE001 — absolute final safety net
        try:
            sys.stderr.write(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"kesco_platform logging pipeline failure "
                f"(incident_id={incident_id}, component={component}): "
                f"{logging_failure}\n"
            )
        except Exception:  # noqa: BLE001
            pass
        return incident_id


def get_logger() -> logging.Logger:
    """
    Returns the process-wide "kesco_platform" logger, initializing it on
    first call. Exposed for call sites that want to emit a non-exception
    diagnostic (e.g. logger.info(...)) using the same centralized handler
    configuration as `log_exception`, without needing to catch an actual
    exception first. Never raises — falls back to a bare, handler-less
    logger instance if initialization fails, so a `.info()`/`.warning()`
    call on the returned object can never itself raise due to missing
    handlers.
    """
    try:
        return _initialize_logger()
    except Exception:  # noqa: BLE001
        return logging.getLogger(_LOGGER_NAME)