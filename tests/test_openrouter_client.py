from openrouter_client import build_chat_kwargs


def test_build_chat_kwargs_omits_tools_without_web_search() -> None:
    kwargs = build_chat_kwargs(
        model="deepseek/deepseek-chat",
        user_message="hello",
        request_defaults={
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "tools": [{"type": "should-not-leak"}],
            "timeout_seconds": 999,
        },
        extra_headers={"X-Title": "OpenRouterLLMClient"},
        use_web_search=False,
        timeout_seconds=30,
    )

    assert "tools" not in kwargs
    assert "timeout_seconds" not in kwargs
    assert kwargs["stream"] is False
    assert kwargs["timeout"] == 30
    assert kwargs["model"] == "deepseek/deepseek-chat"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["extra_headers"] == {"X-Title": "OpenRouterLLMClient"}


def test_build_chat_kwargs_adds_web_search_server_tool() -> None:
    kwargs = build_chat_kwargs(
        model="deepseek/deepseek-chat",
        user_message="hello",
        request_defaults={"temperature": 0.7},
        use_web_search=True,
    )

    assert kwargs["tools"] == [{"type": "openrouter:web_search"}]
    assert kwargs["timeout"] == 120.0


def test_build_chat_kwargs_runtime_generation_values_override_defaults() -> None:
    kwargs = build_chat_kwargs(
        model="deepseek/deepseek-chat",
        user_message="hello",
        request_defaults={"temperature": 0.7, "max_tokens": 2000},
        temperature=0.2,
        max_tokens=4000,
    )

    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 4000
