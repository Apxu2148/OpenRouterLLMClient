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
        },
        extra_headers={"X-Title": "OpenRouterLLMClient"},
        use_web_search=False,
    )

    assert "tools" not in kwargs
    assert kwargs["stream"] is False
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
