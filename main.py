from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import TypeVar

from config import AppConfig, RuntimeLimits, is_missing_api_key, load_config
from logger import log_chat, log_error, sanitize_text
from openrouter_client import (
    EmptyResponseError,
    MissingAPIKeyError,
    OpenRouterClientError,
    OpenRouterLLMClient,
    OpenRouterTimeoutError,
)


HELP_TEXT = """Commands:
  /help              Show this help.
  /model             Show the current model.
  /use <model_id>    Switch the current model.
  /models            List available OpenRouter models.
  /models <filter>   List models matching id or name.
  /timeout           Show the current request timeout.
  /timeout <seconds> Set local wait timeout for later API calls.
  /tokens            Show current max output tokens.
  /tokens <number>   Set max output tokens for this session.
  /temp              Show current temperature.
  /temp <number>     Set temperature for this session.
  /params            Show current model and request parameters.
  /web               Show web search status.
  /web on            Enable web search for normal messages.
  /web off           Disable web search for normal messages.
  /web status        Show web search status.
  /askweb <question> Send one question with web search.
  /exit, /quit       Exit the program.

Ctrl+C while waiting for a response interrupts local waiting and returns to prompt.
"""

MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY not found.\n"
    "Create a .env file based on .env.example and put your OpenRouter API key there."
)

T = TypeVar("T")
DEFAULT_RUNTIME_TEMPERATURE = 0.7
DEFAULT_RUNTIME_MAX_TOKENS = 2000


class ApplicationTimeout(TimeoutError):
    pass


def main() -> int:
    try:
        app_config = load_config()
    except Exception as exc:
        print(f"Configuration error: {sanitize_text(exc)}")
        return 1

    current_model = app_config.default_model
    print("OpenRouter LLM Client")
    print(f"Current model: {current_model}")
    print("Type /help for commands.")
    print()

    if is_missing_api_key(app_config.api_key):
        print(MISSING_KEY_MESSAGE)
        return 1

    try:
        client = OpenRouterLLMClient(app_config)
    except MissingAPIKeyError:
        print(MISSING_KEY_MESSAGE)
        return 1

    return run_repl(client, app_config, current_model)


def run_repl(client: OpenRouterLLMClient, app_config: AppConfig, current_model: str) -> int:
    web_search_enabled = app_config.web_search.enabled_by_default
    request_timeout_seconds = app_config.request_timeout_seconds
    current_temperature = initial_temperature(app_config)
    current_max_tokens = initial_max_tokens(app_config)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            command_result = handle_command(
                user_input,
                client,
                app_config,
                current_model,
                web_search_enabled,
                request_timeout_seconds,
                current_temperature,
                current_max_tokens,
            )
            current_model = command_result.current_model
            web_search_enabled = command_result.web_search_enabled
            request_timeout_seconds = command_result.request_timeout_seconds
            current_temperature = command_result.current_temperature
            current_max_tokens = command_result.current_max_tokens
            if command_result.should_exit:
                print("Goodbye.")
                return 0
            continue

        send_chat_message(
            client=client,
            current_model=current_model,
            user_message=user_input,
            use_web_search=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
            temperature=current_temperature,
            max_tokens=current_max_tokens,
        )


class CommandResult:
    def __init__(
        self,
        current_model: str,
        should_exit: bool = False,
        web_search_enabled: bool = False,
        request_timeout_seconds: float = 120.0,
        current_temperature: float = DEFAULT_RUNTIME_TEMPERATURE,
        current_max_tokens: int = DEFAULT_RUNTIME_MAX_TOKENS,
    ) -> None:
        self.current_model = current_model
        self.should_exit = should_exit
        self.web_search_enabled = web_search_enabled
        self.request_timeout_seconds = request_timeout_seconds
        self.current_temperature = current_temperature
        self.current_max_tokens = current_max_tokens


def handle_command(
    raw_command: str,
    client: OpenRouterLLMClient,
    app_config: AppConfig,
    current_model: str,
    web_search_enabled: bool = False,
    request_timeout_seconds: float | None = None,
    current_temperature: float | None = None,
    current_max_tokens: int | None = None,
) -> CommandResult:
    if request_timeout_seconds is None:
        request_timeout_seconds = app_config.request_timeout_seconds
    if current_temperature is None:
        current_temperature = initial_temperature(app_config)
    if current_max_tokens is None:
        current_max_tokens = initial_max_tokens(app_config)

    def command_result(
        model: str = current_model,
        should_exit: bool = False,
        web_enabled: bool = web_search_enabled,
        timeout_seconds: float = request_timeout_seconds,
        temperature: float = current_temperature,
        max_tokens: int = current_max_tokens,
    ) -> CommandResult:
        return CommandResult(
            model,
            should_exit=should_exit,
            web_search_enabled=web_enabled,
            request_timeout_seconds=timeout_seconds,
            current_temperature=temperature,
            current_max_tokens=max_tokens,
        )

    command, _, argument = raw_command.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command == "/help":
        print(HELP_TEXT)
        return command_result()

    if command == "/model":
        print(f"Current model: {current_model}")
        return command_result()

    if command == "/use":
        if not argument:
            print("Usage: /use <model_id>")
            return command_result()
        print(f"Current model: {argument}")
        return command_result(model=argument)

    if command == "/models":
        print_models(client, current_model, argument or None, request_timeout_seconds)
        return command_result()

    if command == "/timeout":
        if not argument:
            print(f"Current request timeout: {format_seconds(request_timeout_seconds)} seconds")
            return command_result()

        try:
            new_timeout_seconds = parse_timeout_seconds(argument)
        except ValueError:
            print("Invalid timeout. Please provide a positive number of seconds.")
            return command_result()

        print(f"Request timeout set to {format_seconds(new_timeout_seconds)} seconds")
        return command_result(timeout_seconds=new_timeout_seconds)

    if command == "/tokens":
        if not argument:
            print(f"Current max tokens: {current_max_tokens}")
            return command_result()

        try:
            new_max_tokens = parse_max_tokens(argument, app_config.runtime_limits)
        except ValueError:
            print(invalid_max_tokens_message(app_config.runtime_limits))
            return command_result()

        print(f"Max tokens set to {new_max_tokens}")
        return command_result(max_tokens=new_max_tokens)

    if command == "/temp":
        if not argument:
            print(f"Current temperature: {format_temperature(current_temperature)}")
            return command_result()

        try:
            new_temperature = parse_temperature(argument, app_config.runtime_limits)
        except ValueError:
            print(invalid_temperature_message(app_config.runtime_limits))
            return command_result()

        print(f"Temperature set to {format_temperature(new_temperature)}")
        return command_result(temperature=new_temperature)

    if command == "/params":
        print_request_params(
            current_model=current_model,
            temperature=current_temperature,
            max_tokens=current_max_tokens,
            timeout_seconds=request_timeout_seconds,
            web_search_enabled=web_search_enabled,
            web_search_tool_type=app_config.web_search.tool_type,
        )
        return command_result()

    if command == "/web":
        argument = argument.lower()
        if argument in {"", "status"}:
            print_web_status(web_search_enabled, app_config.web_search.tool_type)
            return command_result()
        if argument == "on":
            print_web_status(True, app_config.web_search.tool_type)
            return command_result(web_enabled=True)
        if argument == "off":
            print_web_status(False, app_config.web_search.tool_type)
            return command_result(web_enabled=False)

        print("Usage: /web [on|off|status]")
        return command_result()

    if command == "/askweb":
        if not argument:
            print("Usage: /askweb <question>")
            return command_result()

        send_chat_message(
            client=client,
            current_model=current_model,
            user_message=argument,
            use_web_search=True,
            request_timeout_seconds=request_timeout_seconds,
            temperature=current_temperature,
            max_tokens=current_max_tokens,
        )
        return command_result()

    if command in {"/exit", "/quit"}:
        return command_result(should_exit=True)

    print("Unknown command. Type /help for commands.")
    return command_result()


def send_chat_message(
    client: OpenRouterLLMClient,
    current_model: str,
    user_message: str,
    use_web_search: bool,
    request_timeout_seconds: float,
    temperature: float,
    max_tokens: int,
) -> None:
    try:
        assistant_response = run_with_application_timeout(
            lambda: client.chat(
                current_model,
                user_message,
                use_web_search=use_web_search,
                timeout_seconds=request_timeout_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            request_timeout_seconds,
        )
    except KeyboardInterrupt:
        message = "Request interrupted by user. Returned to prompt."
        print(message)
        print("Note: provider-side processing may already have started.")
        log_error(
            current_model,
            KeyboardInterrupt(message),
            web_search=use_web_search,
            user_message=user_message,
            cancelled=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        log_chat(
            current_model,
            user_message,
            message,
            success=False,
            web_search=use_web_search,
            cancelled=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return
    except (ApplicationTimeout, OpenRouterTimeoutError) as exc:
        message = sanitize_text(exc)
        print(message)
        print("Note: provider-side processing may already have started.")
        log_error(
            current_model,
            exc,
            web_search=use_web_search,
            user_message=user_message,
            timeout=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        log_chat(
            current_model,
            user_message,
            message,
            success=False,
            web_search=use_web_search,
            timeout=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return
    except (EmptyResponseError, OpenRouterClientError) as exc:
        message = sanitize_text(exc)
        print(f"Error: {message}")
        log_error(
            current_model,
            exc,
            web_search=use_web_search,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        log_chat(
            current_model,
            user_message,
            message,
            success=False,
            web_search=use_web_search,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return

    print(assistant_response)
    print()
    log_chat(
        current_model,
        user_message,
        assistant_response,
        success=True,
        web_search=use_web_search,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_with_application_timeout(call: Callable[[], T], timeout_seconds: float) -> T:
    result_queue: queue.Queue[tuple[str, T | BaseException]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("result", call()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, name="openrouter-request", daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ApplicationTimeout(
                f"Request timed out after {format_seconds(timeout_seconds)} seconds. Returned to prompt."
            )

        try:
            result_type, value = result_queue.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            continue

        if result_type == "error":
            raise value

        return value


def print_web_status(enabled: bool, tool_type: str) -> None:
    if enabled:
        print("Web search: on")
        print(f"Tool: {tool_type}")
        return

    print("Web search: off")


def initial_temperature(app_config: AppConfig) -> float:
    try:
        temperature = float(app_config.request_defaults.get("temperature", DEFAULT_RUNTIME_TEMPERATURE))
    except (TypeError, ValueError):
        return DEFAULT_RUNTIME_TEMPERATURE

    if not isfinite(temperature):
        return DEFAULT_RUNTIME_TEMPERATURE

    return temperature


def initial_max_tokens(app_config: AppConfig) -> int:
    value = app_config.request_defaults.get("max_tokens", DEFAULT_RUNTIME_MAX_TOKENS)
    if isinstance(value, bool):
        return DEFAULT_RUNTIME_MAX_TOKENS

    try:
        max_tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RUNTIME_MAX_TOKENS

    if max_tokens < 1:
        return DEFAULT_RUNTIME_MAX_TOKENS

    return max_tokens


def parse_max_tokens(raw_value: str, limits: RuntimeLimits) -> int:
    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise ValueError("Invalid max tokens.") from exc

    if max_tokens < limits.max_tokens_min or max_tokens > limits.max_tokens_max:
        raise ValueError("Invalid max tokens.")

    return max_tokens


def parse_temperature(raw_value: str, limits: RuntimeLimits) -> float:
    try:
        temperature = float(raw_value)
    except ValueError as exc:
        raise ValueError("Invalid temperature.") from exc

    if (
        not isfinite(temperature)
        or temperature < limits.temperature_min
        or temperature > limits.temperature_max
    ):
        raise ValueError("Invalid temperature.")

    return temperature


def invalid_max_tokens_message(limits: RuntimeLimits) -> str:
    return (
        "Invalid max tokens. Please provide an integer between "
        f"{limits.max_tokens_min} and {limits.max_tokens_max}."
    )


def invalid_temperature_message(limits: RuntimeLimits) -> str:
    return (
        "Invalid temperature. Please provide a number between "
        f"{format_temperature(limits.temperature_min)} and "
        f"{format_temperature(limits.temperature_max)}."
    )


def print_request_params(
    current_model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    web_search_enabled: bool,
    web_search_tool_type: str,
) -> None:
    print("Current request parameters:")
    print(f"Model: {current_model}")
    print(f"Temperature: {format_temperature(temperature)}")
    print(f"Max tokens: {max_tokens}")
    print(f"Timeout: {format_seconds(timeout_seconds)} seconds")
    print(f"Web search: {'on' if web_search_enabled else 'off'}")
    if web_search_enabled:
        print(f"Web search tool: {web_search_tool_type}")
    print("Stream: false")


def parse_timeout_seconds(raw_value: str) -> float:
    try:
        seconds = float(raw_value)
    except ValueError as exc:
        raise ValueError("Invalid timeout.") from exc

    if seconds <= 0:
        raise ValueError("Invalid timeout.")

    return seconds


def format_seconds(seconds: float) -> str:
    seconds_float = float(seconds)
    if seconds_float.is_integer():
        return str(int(seconds_float))

    return str(seconds_float)


def format_temperature(temperature: float) -> str:
    return str(float(temperature))


def print_models(
    client: OpenRouterLLMClient,
    current_model: str,
    text_filter: str | None,
    request_timeout_seconds: float,
) -> None:
    try:
        models = run_with_application_timeout(
            lambda: client.list_models(text_filter, timeout_seconds=request_timeout_seconds),
            request_timeout_seconds,
        )
    except (ApplicationTimeout, OpenRouterTimeoutError) as exc:
        print(sanitize_text(exc))
        print("Note: provider-side processing may already have started.")
        log_error(current_model, exc, timeout=True)
        return
    except OpenRouterClientError as exc:
        print(f"Could not load models: {sanitize_text(exc)}")
        log_error(current_model, exc)
        return

    if not models:
        print("No models found.")
        return

    for model in models:
        label = model.id
        if model.name:
            label += f" - {model.name}"
        print(label)


if __name__ == "__main__":
    raise SystemExit(main())
