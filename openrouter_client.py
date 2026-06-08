from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI

from config import (
    AppConfig,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_WEB_SEARCH_TOOL_TYPE,
    OPENROUTER_BASE_URL,
    is_missing_api_key,
)
from logger import sanitize_text
from models import ModelInfo


class OpenRouterClientError(RuntimeError):
    pass


class MissingAPIKeyError(OpenRouterClientError):
    pass


class EmptyResponseError(OpenRouterClientError):
    pass


class OpenRouterTimeoutError(OpenRouterClientError):
    pass


class OpenRouterLLMClient:
    def __init__(self, config: AppConfig) -> None:
        if is_missing_api_key(config.api_key):
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY not found.\n"
                "Create a .env file based on .env.example and put your OpenRouter API key there."
            )

        self.config = config
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=config.api_key,
        )

    def chat(
        self,
        model: str,
        user_message: str,
        use_web_search: bool = False,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                **build_chat_kwargs(
                    model=model,
                    user_message=user_message,
                    request_defaults=self.config.request_defaults,
                    extra_headers=self._extra_headers(),
                    use_web_search=use_web_search,
                    web_search_tool_type=self.config.web_search.tool_type,
                    timeout_seconds=timeout_seconds or self.config.request_timeout_seconds,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        except AuthenticationError as exc:
            raise OpenRouterClientError("Authentication failed. Check your OpenRouter API key.") from exc
        except APITimeoutError as exc:
            raise OpenRouterTimeoutError("Request timed out. Returned to prompt.") from exc
        except APIConnectionError as exc:
            raise OpenRouterClientError("Network error. Check your internet connection.") from exc
        except APIError as exc:
            raise OpenRouterClientError(f"OpenRouter API error: {sanitize_text(exc)}") from exc
        except Exception as exc:
            raise OpenRouterClientError(f"Request failed: {sanitize_text(exc)}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise EmptyResponseError("Model returned an empty response.") from exc

        if isinstance(content, list):
            text = "".join(str(part) for part in content).strip()
        else:
            text = str(content or "").strip()

        if not text:
            raise EmptyResponseError("Model returned an empty response.")

        return text

    def list_models(
        self,
        text_filter: str | None = None,
        timeout_seconds: float | None = None,
    ) -> list[ModelInfo]:
        try:
            models_response = self._client.models.list(
                extra_headers=self._extra_headers(),
                timeout=timeout_seconds or self.config.request_timeout_seconds,
            )
        except AuthenticationError as exc:
            raise OpenRouterClientError("Authentication failed. Check your OpenRouter API key.") from exc
        except APITimeoutError as exc:
            raise OpenRouterTimeoutError("Request timed out. Returned to prompt.") from exc
        except APIConnectionError as exc:
            raise OpenRouterClientError("Network error. Check your internet connection.") from exc
        except APIError as exc:
            raise OpenRouterClientError(f"OpenRouter models API error: {sanitize_text(exc)}") from exc
        except Exception as exc:
            raise OpenRouterClientError(f"Could not load models: {sanitize_text(exc)}") from exc

        raw_items = getattr(models_response, "data", models_response)
        model_infos = [
            model_info
            for item in raw_items
            if (model_info := ModelInfo.from_api_item(item)).id
        ]
        return [model for model in model_infos if model.matches(text_filter)]

    def _extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.app_title:
            headers["X-Title"] = self.config.app_title
        return headers


def build_chat_kwargs(
    model: str,
    user_message: str,
    request_defaults: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
    use_web_search: bool = False,
    web_search_tool_type: str = DEFAULT_WEB_SEARCH_TOOL_TYPE,
    timeout_seconds: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(request_defaults)
    kwargs.pop("stream", None)
    kwargs.pop("tools", None)
    kwargs.pop("timeout_seconds", None)
    kwargs["stream"] = False
    kwargs["model"] = model
    kwargs["messages"] = [{"role": "user", "content": user_message}]
    kwargs["extra_headers"] = extra_headers or {}

    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if use_web_search:
        kwargs["tools"] = [{"type": web_search_tool_type}]

    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds

    return kwargs
