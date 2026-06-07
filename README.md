# OpenRouterLLMClient

Minimal reusable Python console client for the OpenRouter API. It uses the OpenAI-compatible endpoint at `https://openrouter.ai/api/v1`, reads the API key from `.env`, sends chat messages to the selected model, and lets you switch models from the console.

The default model is `deepseek/deepseek-chat`, but the client is OpenRouter-first and accepts any OpenRouter model id.

## Setup

Create the project-local virtual environment:

```bat
cd /d C:\Python\OpenRouterLLMClient
python -m venv .venv
```

Install dependencies only into that environment:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

You can also run:

```bat
setup.bat
```

## Configure API Key

Copy `.env.example` to `.env`:

```bat
copy .env.example .env
```

Edit `.env` and replace `put_your_openrouter_api_key_here` with your OpenRouter API key.

Get an API key from your OpenRouter account dashboard. Do not commit `.env` and do not paste real API keys into source files, logs, issues, or chat.

If the key is missing, the program prints:

```text
OPENROUTER_API_KEY not found.
Create a .env file based on .env.example and put your OpenRouter API key there.
```

## Run

```bat
run.bat
```

Or directly:

```bat
.venv\Scripts\python.exe main.py
```

At startup the console shows:

```text
OpenRouter LLM Client
Current model: deepseek/deepseek-chat
Type /help for commands.
```

## Console Commands

```text
/help              Show commands.
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
/exit              Exit the program.
/quit              Exit the program.
```

Example model switch:

```text
/use deepseek/deepseek-r1
```

Example test request:

```text
Hello, summarize what OpenRouter is in one sentence.
```

## Web Search Mode

Web search is off by default. Enable it for later normal messages with:

```text
/web on
```

Turn it off with:

```text
/web off
```

Check the current state with `/web` or `/web status`. Use `/askweb <question>` for a one-shot web-search request without changing the current web mode.

Web search may cost more than a normal request because search requests can have their own cost and found context can add input tokens. See `DEVELOPER_GUIDE.md` for implementation and reuse details.

## Timeout And Interrupts

The default request timeout is 120 seconds. Use `/timeout` to show the current value, or `/timeout <seconds>` to set a positive local application-level timeout for waiting on later model responses.

When the timeout fires, the CLI stops waiting, logs `timeout: true`, and returns to the prompt. The timeout returns control to the local CLI; it does not guarantee cancellation of provider-side processing or billing.

While waiting for a model response, press Ctrl+C to stop local waiting and return to the prompt. Local cancellation has the same provider-side limitation.

## Tests

After dependencies are installed:

```bat
.venv\Scripts\python.exe -m pytest
```

Tests do not require a real API key.

## Logs

Runtime logs are written to:

```text
logs/chat.jsonl
logs/errors.jsonl
```

Only short sanitized previews are logged. API keys and likely secret values are masked. The `.gitignore` excludes real logs and `.env`, while `logs/.gitkeep` keeps the folder in the project.

Chat and error log rows include `web_search: true` or `web_search: false`. Cancelled and timed-out requests are logged with matching flags.

## Limits

This project is intentionally simple: no GUI, no agent framework, no RAG, no local LLMs, and no processing of user files.
