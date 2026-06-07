from pathlib import Path

import pytest

from config import is_missing_api_key, load_config


def test_config_loads_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = load_config()

    assert config.default_model == "deepseek/deepseek-chat"
    assert config.request_defaults["temperature"] == 0.7
    assert config.request_defaults["max_tokens"] == 2000
    assert config.web_search.enabled_by_default is False
    assert config.web_search.tool_type == "openrouter:web_search"


def test_empty_api_key_is_missing() -> None:
    assert is_missing_api_key(None)
    assert is_missing_api_key("")
    assert is_missing_api_key("   ")
    assert not is_missing_api_key("sk-or-v1-example")


def test_config_reads_environment_without_real_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")

    config = load_config(env_file=env_file)

    assert config.api_key == "test-key"
