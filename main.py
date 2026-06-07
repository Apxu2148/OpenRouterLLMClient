from __future__ import annotations

from config import AppConfig, is_missing_api_key, load_config
from logger import log_chat, log_error, sanitize_text
from openrouter_client import (
    EmptyResponseError,
    MissingAPIKeyError,
    OpenRouterClientError,
    OpenRouterLLMClient,
)


HELP_TEXT = """Commands:
  /help              Show this help.
  /model             Show the current model.
  /use <model_id>    Switch the current model.
  /models            List available OpenRouter models.
  /models <filter>   List models matching id or name.
  /web               Show web search status.
  /web on            Enable web search for normal messages.
  /web off           Disable web search for normal messages.
  /web status        Show web search status.
  /askweb <question> Send one question with web search.
  /exit, /quit       Exit the program.
"""

MISSING_KEY_MESSAGE = (
    "OPENROUTER_API_KEY not found.\n"
    "Create a .env file based on .env.example and put your OpenRouter API key there."
)


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
            )
            current_model = command_result.current_model
            web_search_enabled = command_result.web_search_enabled
            if command_result.should_exit:
                print("Goodbye.")
                return 0
            continue

        send_chat_message(
            client=client,
            current_model=current_model,
            user_message=user_input,
            use_web_search=web_search_enabled,
        )


class CommandResult:
    def __init__(
        self,
        current_model: str,
        should_exit: bool = False,
        web_search_enabled: bool = False,
    ) -> None:
        self.current_model = current_model
        self.should_exit = should_exit
        self.web_search_enabled = web_search_enabled


def handle_command(
    raw_command: str,
    client: OpenRouterLLMClient,
    app_config: AppConfig,
    current_model: str,
    web_search_enabled: bool = False,
) -> CommandResult:
    command, _, argument = raw_command.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command == "/help":
        print(HELP_TEXT)
        return CommandResult(current_model, web_search_enabled=web_search_enabled)

    if command == "/model":
        print(f"Current model: {current_model}")
        return CommandResult(current_model, web_search_enabled=web_search_enabled)

    if command == "/use":
        if not argument:
            print("Usage: /use <model_id>")
            return CommandResult(current_model, web_search_enabled=web_search_enabled)
        print(f"Current model: {argument}")
        return CommandResult(argument, web_search_enabled=web_search_enabled)

    if command == "/models":
        print_models(client, current_model, argument or None)
        return CommandResult(current_model, web_search_enabled=web_search_enabled)

    if command == "/web":
        argument = argument.lower()
        if argument in {"", "status"}:
            print_web_status(web_search_enabled, app_config.web_search.tool_type)
            return CommandResult(current_model, web_search_enabled=web_search_enabled)
        if argument == "on":
            print_web_status(True, app_config.web_search.tool_type)
            return CommandResult(current_model, web_search_enabled=True)
        if argument == "off":
            print_web_status(False, app_config.web_search.tool_type)
            return CommandResult(current_model, web_search_enabled=False)

        print("Usage: /web [on|off|status]")
        return CommandResult(current_model, web_search_enabled=web_search_enabled)

    if command == "/askweb":
        if not argument:
            print("Usage: /askweb <question>")
            return CommandResult(current_model, web_search_enabled=web_search_enabled)

        send_chat_message(
            client=client,
            current_model=current_model,
            user_message=argument,
            use_web_search=True,
        )
        return CommandResult(current_model, web_search_enabled=web_search_enabled)

    if command in {"/exit", "/quit"}:
        return CommandResult(
            current_model,
            should_exit=True,
            web_search_enabled=web_search_enabled,
        )

    print("Unknown command. Type /help for commands.")
    return CommandResult(current_model, web_search_enabled=web_search_enabled)


def send_chat_message(
    client: OpenRouterLLMClient,
    current_model: str,
    user_message: str,
    use_web_search: bool,
) -> None:
    try:
        assistant_response = client.chat(
            current_model,
            user_message,
            use_web_search=use_web_search,
        )
    except (EmptyResponseError, OpenRouterClientError) as exc:
        message = sanitize_text(exc)
        print(f"Error: {message}")
        log_error(current_model, exc, web_search=use_web_search)
        log_chat(current_model, user_message, message, success=False, web_search=use_web_search)
        return

    print(assistant_response)
    print()
    log_chat(current_model, user_message, assistant_response, success=True, web_search=use_web_search)


def print_web_status(enabled: bool, tool_type: str) -> None:
    if enabled:
        print("Web search: on")
        print(f"Tool: {tool_type}")
        return

    print("Web search: off")


def print_models(client: OpenRouterLLMClient, current_model: str, text_filter: str | None) -> None:
    try:
        models = client.list_models(text_filter)
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
