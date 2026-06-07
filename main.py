from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from config import AppConfig, is_missing_api_key, load_config
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
            )
            current_model = command_result.current_model
            web_search_enabled = command_result.web_search_enabled
            request_timeout_seconds = command_result.request_timeout_seconds
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
        )


class CommandResult:
    def __init__(
        self,
        current_model: str,
        should_exit: bool = False,
        web_search_enabled: bool = False,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        self.current_model = current_model
        self.should_exit = should_exit
        self.web_search_enabled = web_search_enabled
        self.request_timeout_seconds = request_timeout_seconds


def handle_command(
    raw_command: str,
    client: OpenRouterLLMClient,
    app_config: AppConfig,
    current_model: str,
    web_search_enabled: bool = False,
    request_timeout_seconds: float | None = None,
) -> CommandResult:
    if request_timeout_seconds is None:
        request_timeout_seconds = app_config.request_timeout_seconds

    command, _, argument = raw_command.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command == "/help":
        print(HELP_TEXT)
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command == "/model":
        print(f"Current model: {current_model}")
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command == "/use":
        if not argument:
            print("Usage: /use <model_id>")
            return CommandResult(
                current_model,
                web_search_enabled=web_search_enabled,
                request_timeout_seconds=request_timeout_seconds,
            )
        print(f"Current model: {argument}")
        return CommandResult(
            argument,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command == "/models":
        print_models(client, current_model, argument or None, request_timeout_seconds)
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command == "/timeout":
        if not argument:
            print(f"Current request timeout: {format_seconds(request_timeout_seconds)} seconds")
            return CommandResult(
                current_model,
                web_search_enabled=web_search_enabled,
                request_timeout_seconds=request_timeout_seconds,
            )

        try:
            new_timeout_seconds = parse_timeout_seconds(argument)
        except ValueError:
            print("Invalid timeout. Please provide a positive number of seconds.")
            return CommandResult(
                current_model,
                web_search_enabled=web_search_enabled,
                request_timeout_seconds=request_timeout_seconds,
            )

        print(f"Request timeout set to {format_seconds(new_timeout_seconds)} seconds")
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=new_timeout_seconds,
        )

    if command == "/web":
        argument = argument.lower()
        if argument in {"", "status"}:
            print_web_status(web_search_enabled, app_config.web_search.tool_type)
            return CommandResult(
                current_model,
                web_search_enabled=web_search_enabled,
                request_timeout_seconds=request_timeout_seconds,
            )
        if argument == "on":
            print_web_status(True, app_config.web_search.tool_type)
            return CommandResult(
                current_model,
                web_search_enabled=True,
                request_timeout_seconds=request_timeout_seconds,
            )
        if argument == "off":
            print_web_status(False, app_config.web_search.tool_type)
            return CommandResult(
                current_model,
                web_search_enabled=False,
                request_timeout_seconds=request_timeout_seconds,
            )

        print("Usage: /web [on|off|status]")
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command == "/askweb":
        if not argument:
            print("Usage: /askweb <question>")
            return CommandResult(
                current_model,
                web_search_enabled=web_search_enabled,
                request_timeout_seconds=request_timeout_seconds,
            )

        send_chat_message(
            client=client,
            current_model=current_model,
            user_message=argument,
            use_web_search=True,
            request_timeout_seconds=request_timeout_seconds,
        )
        return CommandResult(
            current_model,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    if command in {"/exit", "/quit"}:
        return CommandResult(
            current_model,
            should_exit=True,
            web_search_enabled=web_search_enabled,
            request_timeout_seconds=request_timeout_seconds,
        )

    print("Unknown command. Type /help for commands.")
    return CommandResult(
        current_model,
        web_search_enabled=web_search_enabled,
        request_timeout_seconds=request_timeout_seconds,
    )


def send_chat_message(
    client: OpenRouterLLMClient,
    current_model: str,
    user_message: str,
    use_web_search: bool,
    request_timeout_seconds: float,
) -> None:
    try:
        assistant_response = run_with_application_timeout(
            lambda: client.chat(
                current_model,
                user_message,
                use_web_search=use_web_search,
                timeout_seconds=request_timeout_seconds,
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
        )
        log_chat(
            current_model,
            user_message,
            message,
            success=False,
            web_search=use_web_search,
            cancelled=True,
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
        )
        log_chat(
            current_model,
            user_message,
            message,
            success=False,
            web_search=use_web_search,
            timeout=True,
        )
        return
    except (EmptyResponseError, OpenRouterClientError) as exc:
        message = sanitize_text(exc)
        print(f"Error: {message}")
        log_error(current_model, exc, web_search=use_web_search, user_message=user_message)
        log_chat(current_model, user_message, message, success=False, web_search=use_web_search)
        return

    print(assistant_response)
    print()
    log_chat(current_model, user_message, assistant_response, success=True, web_search=use_web_search)


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
