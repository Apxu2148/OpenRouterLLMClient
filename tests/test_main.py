from collections.abc import Iterator

import pytest

import main
from config import AppConfig, WebSearchConfig


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, str, bool]] = []

    def chat(self, model: str, user_message: str, use_web_search: bool = False) -> str:
        self.chat_calls.append((model, user_message, use_web_search))
        return "ok"


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        api_key="test-key",
        http_referer=None,
        app_title=None,
        default_model="deepseek/deepseek-chat",
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
        )
    ]


def test_run_repl_sends_normal_messages_with_enabled_web_state(
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    inputs: Iterator[str] = iter(["/web on", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    exit_code = main.run_repl(client, app_config, "deepseek/deepseek-chat")

    assert exit_code == 0
    assert client.chat_calls == [("deepseek/deepseek-chat", "hello", True)]
