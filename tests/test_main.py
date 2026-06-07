from collections.abc import Iterator
import time

import pytest

import main
from config import AppConfig, WebSearchConfig


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, str, bool, float | None]] = []

    def chat(
        self,
        model: str,
        user_message: str,
        use_web_search: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        self.chat_calls.append((model, user_message, use_web_search, timeout_seconds))
        return "ok"


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        api_key="test-key",
        http_referer=None,
        app_title=None,
        default_model="deepseek/deepseek-chat",
        request_timeout_seconds=120.0,
        request_defaults={},
        web_search=WebSearchConfig(
            enabled_by_default=False,
            tool_type="openrouter:web_search",
        ),
    )


@pytest.fixture(autouse=True)
def disable_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "log_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "log_error", lambda *args, **kwargs: None)


def test_web_on_changes_state(app_config: AppConfig, capsys: pytest.CaptureFixture[str]) -> None:
    result = main.handle_command(
        "/web on",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        web_search_enabled=False,
    )

    assert result.web_search_enabled is True
    assert "Web search: on" in capsys.readouterr().out


def test_web_off_changes_state(app_config: AppConfig, capsys: pytest.CaptureFixture[str]) -> None:
    result = main.handle_command(
        "/web off",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        web_search_enabled=True,
    )

    assert result.web_search_enabled is False
    assert "Web search: off" in capsys.readouterr().out


def test_askweb_does_not_change_global_web_state(app_config: AppConfig) -> None:
    client = FakeClient()

    result = main.handle_command(
        "/askweb latest OpenRouter web search news",
        client,
        app_config,
        current_model="deepseek/deepseek-chat",
        web_search_enabled=False,
    )

    assert result.web_search_enabled is False
    assert client.chat_calls == [
        (
            "deepseek/deepseek-chat",
            "latest OpenRouter web search news",
            True,
            120.0,
        )
    ]


def test_timeout_shows_current_value(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/timeout",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        request_timeout_seconds=45,
    )

    assert result.request_timeout_seconds == 45
    assert "Current request timeout: 45 seconds" in capsys.readouterr().out


def test_timeout_command_changes_state(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/timeout 30",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        request_timeout_seconds=120,
    )

    assert result.request_timeout_seconds == 30
    assert "Request timeout set to 30 seconds" in capsys.readouterr().out


@pytest.mark.parametrize("raw_command", ["/timeout abc", "/timeout -5", "/timeout 0"])
def test_timeout_command_rejects_invalid_values(
    raw_command: str,
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        raw_command,
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        request_timeout_seconds=120,
    )

    assert result.request_timeout_seconds == 120
    assert "Invalid timeout. Please provide a positive number of seconds." in capsys.readouterr().out


def test_run_repl_sends_normal_messages_with_enabled_web_state(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    inputs: Iterator[str] = iter(["/web on", "/timeout 30", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = main.run_repl(client, app_config, "deepseek/deepseek-chat")

    assert exit_code == 0
    assert client.chat_calls == [("deepseek/deepseek-chat", "hello", True, 30.0)]


def test_application_timeout_wrapper_returns_fast_result() -> None:
    result = main.run_with_application_timeout(lambda: "ok", timeout_seconds=1)

    assert result == "ok"


def test_application_timeout_wrapper_times_out_slow_call() -> None:
    def slow_call() -> str:
        time.sleep(0.2)
        return "late"

    with pytest.raises(main.ApplicationTimeout) as exc_info:
        main.run_with_application_timeout(slow_call, timeout_seconds=0.01)

    assert "Request timed out after 0.01 seconds. Returned to prompt." in str(exc_info.value)


def test_send_chat_message_logs_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class InterruptingClient:
        def chat(self, *args, **kwargs) -> str:
            raise KeyboardInterrupt

    error_calls: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "log_error", lambda *args, **kwargs: error_calls.append(kwargs))
    monkeypatch.setattr(main, "log_chat", lambda *args, **kwargs: chat_calls.append(kwargs))

    main.send_chat_message(
        client=InterruptingClient(),
        current_model="deepseek/deepseek-chat",
        user_message="long request",
        use_web_search=False,
        request_timeout_seconds=120,
    )

    output = capsys.readouterr().out
    assert "Request interrupted by user. Returned to prompt." in output
    assert error_calls[0]["cancelled"] is True
    assert chat_calls[0]["cancelled"] is True


def test_send_chat_message_logs_application_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SlowClient:
        def chat(self, *args, **kwargs) -> str:
            time.sleep(0.2)
            return "late"

    error_calls: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "log_error", lambda *args, **kwargs: error_calls.append(kwargs))
    monkeypatch.setattr(main, "log_chat", lambda *args, **kwargs: chat_calls.append(kwargs))

    main.send_chat_message(
        client=SlowClient(),
        current_model="deepseek/deepseek-chat",
        user_message="slow request",
        use_web_search=True,
        request_timeout_seconds=0.01,
    )

    output = capsys.readouterr().out
    assert "Request timed out after 0.01 seconds. Returned to prompt." in output
    assert "provider-side processing may already have started" in output
    assert error_calls[0]["timeout"] is True
    assert chat_calls[0]["timeout"] is True


def test_send_chat_message_logs_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TimeoutClient:
        def chat(self, *args, **kwargs) -> str:
            raise main.OpenRouterTimeoutError("Request timed out. Returned to prompt.")

    error_calls: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "log_error", lambda *args, **kwargs: error_calls.append(kwargs))
    monkeypatch.setattr(main, "log_chat", lambda *args, **kwargs: chat_calls.append(kwargs))

    main.send_chat_message(
        client=TimeoutClient(),
        current_model="deepseek/deepseek-chat",
        user_message="slow request",
        use_web_search=True,
        request_timeout_seconds=1,
    )

    output = capsys.readouterr().out
    assert "Request timed out. Returned to prompt." in output
    assert error_calls[0]["timeout"] is True
    assert chat_calls[0]["timeout"] is True
