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


def test_chat_and_error_logs_include_generation_parameters(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logger, "CHAT_LOG_FILE", tmp_path / "chat.jsonl")
    monkeypatch.setattr(logger, "ERROR_LOG_FILE", tmp_path / "errors.jsonl")

    logger.log_chat(
        "deepseek/deepseek-chat",
        "hello",
        "ok",
        success=True,
        temperature=0.2,
        max_tokens=4000,
    )
    logger.log_error(
        "deepseek/deepseek-chat",
        TimeoutError("too slow"),
        temperature=0.2,
        max_tokens=4000,
    )

    chat_payload = json.loads((tmp_path / "chat.jsonl").read_text(encoding="utf-8"))
    error_payload = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8"))

    assert chat_payload["temperature"] == 0.2
    assert chat_payload["max_tokens"] == 4000
    assert error_payload["temperature"] == 0.2
    assert error_payload["max_tokens"] == 4000
