from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DEFAULT_PREVIEW_LIMIT, PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "logs"
CHAT_LOG_FILE = LOG_DIR / "chat.jsonl"
ERROR_LOG_FILE = LOG_DIR / "errors.jsonl"

KEY_PATTERNS = [
    re.compile(r"\bsk-or-v1-[A-Za-z0-9._-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9._-]{20,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(openrouter[_-]?api[_-]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]{12,}"),
]

SECRET_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(value: object, extra_secrets: list[str] | None = None) -> str:
    text = str(value)
    secrets = [os.getenv(name) for name in SECRET_ENV_NAMES]
    if extra_secrets:
        secrets.extend(extra_secrets)

    for secret in secrets:
        if secret and secret.strip():
            text = text.replace(secret, "[REDACTED]")

    for pattern in KEY_PATTERNS:
        text = pattern.sub(lambda match: _mask_regex_match(match), text)

    return text


def preview_text(value: object, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
    sanitized = sanitize_text(value).replace("\r", " ").replace("\n", " ").strip()
    if len(sanitized) <= limit:
        return sanitized

    return sanitized[: limit - 3].rstrip() + "..."


def log_chat(
    model: str,
    user_message: str,
    assistant_response: str,
    success: bool,
) -> None:
    _append_jsonl(
        CHAT_LOG_FILE,
        {
            "timestamp": utc_timestamp(),
            "model": sanitize_text(model),
            "user_message_preview": preview_text(user_message),
            "assistant_response_preview": preview_text(assistant_response),
            "success": success,
        },
    )


def log_error(model: str, error: BaseException | str) -> None:
    error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
    error_message = str(error)
    _append_jsonl(
        ERROR_LOG_FILE,
        {
            "timestamp": utc_timestamp(),
            "model": sanitize_text(model),
            "error_type": sanitize_text(error_type),
            "error_message": preview_text(error_message),
        },
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _mask_regex_match(match: re.Match[str]) -> str:
    if match.lastindex:
        return match.group(1) + "[REDACTED]"

    return "[REDACTED]"
