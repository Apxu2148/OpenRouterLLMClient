import json

import logger


def test_log_error_writes_cancelled_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logger, "ERROR_LOG_FILE", tmp_path / "errors.jsonl")

    logger.log_error(
        "deepseek/deepseek-chat",
        KeyboardInterrupt("stopped"),
        web_search=False,
        user_message="please stop",
        cancelled=True,
    )

    payload = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8"))

    assert payload["cancelled"] is True
    assert payload["error_type"] == "KeyboardInterrupt"
    assert payload["user_message_preview"] == "please stop"


def test_log_error_writes_timeout_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logger, "ERROR_LOG_FILE", tmp_path / "errors.jsonl")

    logger.log_error(
        "deepseek/deepseek-chat",
        TimeoutError("too slow"),
        web_search=True,
        user_message="slow question",
        timeout=True,
    )

    payload = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8"))

    assert payload["timeout"] is True
    assert payload["web_search"] is True
    assert payload["error_type"] == "TimeoutError"
