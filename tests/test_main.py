from collections.abc import Iterator
import time

import pytest

import main
from config import AppConfig, WebSearchConfig


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, str, bool, float | None, float | None, int | None]] = []

    def chat(
        self,
        model: str,
        user_message: str,
        use_web_search: bool = False,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.chat_calls.append((model, user_message, use_web_search, timeout_seconds, temperature, max_tokens))
        return "ok"


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        api_key="test-key",
        http_referer=None,
        app_title=None,
        default_model="deepseek/deepseek-chat",
        request_timeout_seconds=120.0,
        request_defaults={"temperature": 0.7, "max_tokens": 2000, "stream": False},
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
            0.7,
            2000,
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


def test_tokens_shows_current_value(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/tokens",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_max_tokens=1234,
    )

    assert result.current_max_tokens == 1234
    assert "Current max tokens: 1234" in capsys.readouterr().out


def test_tokens_command_changes_state(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/tokens 4000",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_max_tokens=2000,
    )

    assert result.current_max_tokens == 4000
    assert "Max tokens set to 4000" in capsys.readouterr().out


@pytest.mark.parametrize("raw_command", ["/tokens abc", "/tokens 0", "/tokens -10", "/tokens 999999999"])
def test_tokens_command_rejects_invalid_values(
    raw_command: str,
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        raw_command,
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_max_tokens=2000,
    )

    assert result.current_max_tokens == 2000
    assert "Invalid max tokens. Please provide an integer between 1 and 32000." in capsys.readouterr().out


def test_temp_shows_current_value(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/temp",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_temperature=0.2,
    )

    assert result.current_temperature == 0.2
    assert "Current temperature: 0.2" in capsys.readouterr().out


def test_temp_command_changes_state(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        "/temp 0.2",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_temperature=0.7,
    )

    assert result.current_temperature == 0.2
    assert "Temperature set to 0.2" in capsys.readouterr().out


@pytest.mark.parametrize("raw_command", ["/temp abc", "/temp -1", "/temp 3"])
def test_temp_command_rejects_invalid_values(
    raw_command: str,
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main.handle_command(
        raw_command,
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        current_temperature=0.7,
    )

    assert result.current_temperature == 0.7
    assert "Invalid temperature. Please provide a number between 0.0 and 2.0." in capsys.readouterr().out


def test_params_prints_current_request_parameters(
    app_config: AppConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main.handle_command(
        "/params",
        FakeClient(),
        app_config,
        current_model="deepseek/deepseek-chat",
        request_timeout_seconds=120,
        current_temperature=0.2,
        current_max_tokens=4000,
        web_search_enabled=True,
    )

    output = capsys.readouterr().out
    assert "Current request parameters:" in output
    assert "Model: deepseek/deepseek-chat" in output
    assert "Temperature: 0.2" in output
    assert "Max tokens: 4000" in output
    assert "Timeout: 120 seconds" in output
    assert "Web search: on" in output
    assert "Web search tool: openrouter:web_search" in output
    assert "Stream: false" in output


def test_run_repl_sends_normal_messages_with_enabled_web_state(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    inputs: Iterator[str] = iter(["/web on", "/timeout 30", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = main.run_repl(client, app_config, "deepseek/deepseek-chat")

    assert exit_code == 0
    assert client.chat_calls == [("deepseek/deepseek-chat", "hello", True, 30.0, 0.7, 2000)]


def test_run_repl_sends_normal_messages_with_runtime_generation_values(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    inputs: Iterator[str] = iter(["/temp 0.2", "/tokens 4000", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = main.run_repl(client, app_config, "deepseek/deepseek-chat")

    assert exit_code == 0
    assert client.chat_calls == [("deepseek/deepseek-chat", "hello", False, 120.0, 0.2, 4000)]


def test_askweb_uses_runtime_generation_values(app_config: AppConfig) -> None:
    client = FakeClient()

    result = main.handle_command(
        "/askweb latest OpenRouter web search news",
        client,
        app_config,
        current_model="deepseek/deepseek-chat",
        current_temperature=0.2,
        current_max_tokens=4000,
    )

    assert result.web_search_enabled is False
    assert client.chat_calls == [
        (
            "deepseek/deepseek-chat",
            "latest OpenRouter web search news",
            True,
            120.0,
            0.2,
            4000,
        )
    ]


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
        temperature=0.7,
        max_tokens=2000,
    )

    output = capsys.readouterr().out
    assert "Request interrupted by user. Returned to prompt." in output
    assert error_calls[0]["cancelled"] is True
    assert error_calls[0]["temperature"] == 0.7
    assert error_calls[0]["max_tokens"] == 2000
    assert chat_calls[0]["cancelled"] is True
    assert chat_calls[0]["temperature"] == 0.7
    assert chat_calls[0]["max_tokens"] == 2000


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
        temperature=0.2,
        max_tokens=4000,
    )

    output = capsys.readouterr().out
    assert "Request timed out after 0.01 seconds. Returned to prompt." in output
    assert "provider-side processing may already have started" in output
    assert error_calls[0]["timeout"] is True
    assert error_calls[0]["temperature"] == 0.2
    assert error_calls[0]["max_tokens"] == 4000
    assert chat_calls[0]["timeout"] is True
    assert chat_calls[0]["temperature"] == 0.2
    assert chat_calls[0]["max_tokens"] == 4000


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
        temperature=0.2,
        max_tokens=4000,
    )

    output = capsys.readouterr().out
    assert "Request timed out. Returned to prompt." in output
    assert error_calls[0]["timeout"] is True
    assert error_calls[0]["temperature"] == 0.2
    assert error_calls[0]["max_tokens"] == 4000
    assert chat_calls[0]["timeout"] is True
    assert chat_calls[0]["temperature"] == 0.2
    assert chat_calls[0]["max_tokens"] == 4000
