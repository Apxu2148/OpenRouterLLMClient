from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, AuthenticationError, OpenAI

from config import AppConfig, OPENROUTER_BASE_URL, is_missing_api_key
from logger import sanitize_text
from models import ModelInfo


class OpenRouterClientError(RuntimeError):
    pass


class MissingAPIKeyError(OpenRouterClientError):
    pass


class EmptyResponseError(OpenRouterClientError):
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

    def chat(self, model: str, user_message: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_message}],
                extra_headers=self._extra_headers(),
                **self._request_defaults(),
            )
        except AuthenticationError as exc:
            raise OpenRouterClientError("Authentication failed. Check your OpenRouter API key.") from exc
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

    def list_models(self, text_filter: str | None = None) -> list[ModelInfo]:
        try:
            models_response = self._client.models.list(extra_headers=self._extra_headers())
        except AuthenticationError as exc:
            raise OpenRouterClientError("Authentication failed. Check your OpenRouter API key.") from exc
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

    def _request_defaults(self) -> dict[str, Any]:
        defaults = dict(self.config.request_defaults)
        defaults.pop("stream", None)
        defaults["stream"] = False
        return defaults

    def _extra_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.app_title:
            headers["X-Title"] = self.config.app_title
        return headers
