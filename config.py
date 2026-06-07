from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
MODELS_CONFIG_FILE = PROJECT_ROOT / "config" / "models_config.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PREVIEW_LIMIT = 300
DEFAULT_WEB_SEARCH_TOOL_TYPE = "openrouter:web_search"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class WebSearchConfig:
    enabled_by_default: bool = False
    tool_type: str = DEFAULT_WEB_SEARCH_TOOL_TYPE


@dataclass(frozen=True)
class AppConfig:
    api_key: str | None
    http_referer: str | None
    app_title: str | None
    default_model: str
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    request_defaults: dict[str, Any] = field(default_factory=dict)
    model_presets: dict[str, dict[str, str]] = field(default_factory=dict)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)


def is_missing_api_key(api_key: str | None) -> bool:
    return api_key is None or api_key.strip() == ""


def _read_models_config(config_path: Path = MODELS_CONFIG_FILE) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Models config must contain a YAML object.")

    return loaded


def load_config(
    env_file: Path = ENV_FILE,
    models_config_file: Path = MODELS_CONFIG_FILE,
) -> AppConfig:
    load_dotenv(dotenv_path=env_file, override=False)
    models_config = _read_models_config(models_config_file)

    default_model = str(models_config.get("default_model") or "").strip()
    if not default_model:
        raise ValueError("default_model is required in models_config.yaml")

    request_defaults = models_config.get("request_defaults") or {}
    if not isinstance(request_defaults, dict):
        raise ValueError("request_defaults must be a YAML object.")
    request_defaults = dict(request_defaults)
    request_timeout_seconds = _parse_positive_seconds(
        request_defaults.pop("timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS),
        "request_defaults.timeout_seconds",
    )

    model_presets = models_config.get("model_presets") or {}
    if not isinstance(model_presets, dict):
        raise ValueError("model_presets must be a YAML object.")

    web_search = _parse_web_search_config(models_config.get("web_search") or {})

    return AppConfig(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        app_title=os.getenv("OPENROUTER_APP_TITLE"),
        default_model=default_model,
        request_timeout_seconds=request_timeout_seconds,
        request_defaults=request_defaults,
        model_presets=model_presets,
        web_search=web_search,
    )


def _parse_web_search_config(raw_config: object) -> WebSearchConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("web_search must be a YAML object.")

    enabled_by_default = raw_config.get("enabled_by_default", False)
    if not isinstance(enabled_by_default, bool):
        raise ValueError("web_search.enabled_by_default must be true or false.")

    tool_type = str(raw_config.get("tool_type") or DEFAULT_WEB_SEARCH_TOOL_TYPE).strip()
    if not tool_type:
        raise ValueError("web_search.tool_type must not be empty.")

    return WebSearchConfig(
        enabled_by_default=enabled_by_default,
        tool_type=tool_type,
    )


def _parse_positive_seconds(value: object, field_name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number of seconds.") from exc

    if seconds <= 0:
        raise ValueError(f"{field_name} must be a positive number of seconds.")

    return seconds
