OpenRouterLLMClient — инструкция для переиспользования
1. Назначение проекта

OpenRouterLLMClient — это минимальный переиспользуемый Python-клиент для работы с языковыми моделями через OpenRouter.

Проект позволяет:

- подключаться к OpenRouter API;
- использовать OpenAI-compatible endpoint;
- отправлять сообщения в выбранную модель;
- получать ответ модели в Python-консоли;
- переключать модель командой /use;
- просматривать доступные модели командой /models;
- использовать DeepSeek как модель по умолчанию;
- безопасно хранить API-ключ в локальном .env.

Проект не является агентом. В нём нет EidosAgent, RAG, GUI, локальной LLM, работы с файлами пользователя или самообучения. Это только универсальный LLM-клиент.

2. Расположение проекта

Рекомендуемый путь:

C:\Python\OpenRouterLLMClient
3. Основная архитектура
OpenRouterLLMClient/
├─ main.py
├─ openrouter_client.py
├─ config.py
├─ models.py
├─ logger.py
├─ requirements.txt
├─ run.bat
├─ setup.bat
├─ README.md
├─ .env.example
├─ .gitignore
├─ config/
│  └─ models_config.yaml
├─ logs/
│  └─ .gitkeep
└─ tests/
   ├─ test_config.py
   └─ test_sanitize.py
4. Назначение файлов
main.py

Консольный интерфейс проекта.

Отвечает за:

- запуск REPL;
- чтение пользовательского ввода;
- обработку команд /help, /model, /use, /models, /exit, /quit;
- отправку обычных сообщений в OpenRouterClient;
- вывод ответа модели в консоль.
openrouter_client.py

Главный API-клиент.

Отвечает за:

- подключение к OpenRouter через OpenAI-compatible API;
- отправку chat completion запросов;
- получение ответа модели;
- получение списка моделей;
- обработку API/сетевых ошибок.

Использует endpoint:

https://openrouter.ai/api/v1
config.py

Модуль загрузки конфигурации.

Отвечает за:

- чтение .env;
- чтение config/models_config.yaml;
- получение OPENROUTER_API_KEY;
- получение OPENROUTER_HTTP_REFERER;
- получение OPENROUTER_APP_TITLE;
- получение default_model;
- получение request_defaults.
models.py

Вспомогательный модуль для работы с моделями.

Отвечает за:

- хранение текущей модели;
- переключение модели;
- фильтрацию списка моделей;
- работу с model_id.
logger.py

Логирование и очистка секретов.

Отвечает за:

- запись logs/chat.jsonl;
- запись logs/errors.jsonl;
- обрезку preview до 300 символов;
- маскирование API-ключей и похожих секретов;
- предотвращение попадания OPENROUTER_API_KEY в логи.
config/models_config.yaml

Основной конфиг моделей.

Пример:

default_model: deepseek/deepseek-chat

request_defaults:
  temperature: 0.7
  max_tokens: 2000
  stream: false

model_presets:
  deepseek_chat:
    model: deepseek/deepseek-chat
    description: Default general-purpose DeepSeek chat model

  deepseek_r1:
    model: deepseek/deepseek-r1
    description: DeepSeek reasoning model, if available
.env.example

Шаблон локального .env.

OPENROUTER_API_KEY=put_your_openrouter_api_key_here
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=OpenRouterLLMClient
.env

Локальный файл с реальным ключом.

Не должен попадать в git.

run.bat

Запускает клиент через локальное виртуальное окружение:

.venv\Scripts\python.exe main.py
setup.bat

Создаёт .venv и устанавливает зависимости из requirements.txt.

tests/

Минимальные тесты проекта:

- загрузка конфигурации;
- default_model;
- обработка отсутствующего ключа;
- sanitizer;
- masking секретов.
5. Установка на новом компьютере
1. Склонировать проект
cd "C:\Python"
git clone https://github.com/<username>/OpenRouterLLMClient.git
cd "C:\Python\OpenRouterLLMClient"
2. Создать виртуальное окружение
.\setup.bat

Или вручную:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
3. Создать .env
Copy-Item ".env.example" ".env"
notepad .env

Вставить реальный ключ:

OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=OpenRouterLLMClient
4. Запустить
.\run.bat
6. Команды консоли
/help

Показывает список команд.

/model

Показывает текущую выбранную модель.

Пример:

/model
/use <model_id>

Переключает модель.

Пример:

/use deepseek/deepseek-r1
/models

Запрашивает список моделей OpenRouter.

Пример:

/models
/models <filter>

Запрашивает список моделей и фильтрует по строке.

Пример:

/models deepseek
/exit или /quit

Завершает программу.

7. Пример использования

После запуска:

OpenRouter LLM Client
Current model: deepseek/deepseek-chat
Type /help for commands.

Проверить модель:

/model

Отправить запрос:

Ответь одним предложением: соединение с OpenRouter работает.

Переключиться на DeepSeek R1:

/use deepseek/deepseek-r1

Проверить ответ:

Ответь одним предложением: ты работаешь через DeepSeek R1.
8. Как переиспользовать в другом Python-проекте

Минимальный сценарий:

from config import load_config
from openrouter_client import OpenRouterClient

config = load_config()

client = OpenRouterClient(
    api_key=config.openrouter_api_key,
    http_referer=config.openrouter_http_referer,
    app_title=config.openrouter_app_title,
)

response = client.chat(
    model=config.default_model,
    user_message="Привет. Ответь одним предложением.",
    temperature=0.7,
    max_tokens=1000,
)

print(response)

Если структура импортов отличается после будущих изменений, смотреть актуальные сигнатуры в openrouter_client.py.

9. Как использовать проект в будущем EidosAgent

Будущий EidosAgent может использовать этот проект как основу для модуля:

core/llm_backends/openrouter_backend.py

Рекомендуемая логика переноса:

1. Взять openrouter_client.py как основу API-клиента.
2. Взять config.py как основу загрузки .env и models_config.yaml.
3. Взять logger.py как основу sanitizer.
4. Оставить CLI main.py только для ручного тестирования.
5. В EidosAgent сделать LLM Router, который вызывает OpenRouterClient.

Будущий интерфейс Eidos может выглядеть так:

EidosAgent
↓
llm_router.py
↓
openrouter_backend.py
↓
OpenRouter API
↓
выбранная модель
10. Что важно для будущего агента

Этот клиент уже решает несколько задач, которые не нужно писать заново:

- безопасная загрузка API-ключа;
- OpenRouter base_url;
- переключение модели;
- получение списка моделей;
- обработка ошибок;
- логирование без секретов;
- базовый config через YAML;
- локальное .venv;
- run.bat/setup.bat.

При создании EidosAgent не нужно повторно проектировать подключение к OpenRouter. Нужно переиспользовать этот проект как проверенный прототип.

11. Что не надо делать с этим проектом

Не превращать OpenRouterLLMClient в агента.

Не добавлять сюда:

- EidosAgent runtime;
- self-modification;
- RAG;
- vector DB;
- локальные модели;
- управление мышью/клавиатурой;
- OCR;
- Telegram parsing;
- транскрибацию аудио;
- работу с файлами пользователя;
- agent memory.

Этот проект должен оставаться маленьким и переиспользуемым.

12. Рекомендации по git

Перед коммитом проверять:

git status

.env не должен попадать в commit.

Проверка:

git check-ignore -v .env

Ожидаемо:

.gitignore:2:.env .env

Обычный workflow:

git pull
# работа
git status
git add .
git commit -m "Meaningful commit message"
git push
13. Что делать при ошибках
API key not found

Проверить:

.env существует?
OPENROUTER_API_KEY заполнен?
.env находится в корне проекта?
Модель недоступна

Проверить:

/model
/models deepseek
/use <другая_модель>
Ошибка сети

Проверить интернет, VPN, доступность OpenRouter.

Большие расходы

Смотреть usage/cost в интерфейсе OpenRouter. Не хранить финансовую логику в этом проекте, пока интерфейс OpenRouter удобен.

14. Текущий статус проекта
Статус: рабочий
Проверено:
- реальный API-запрос проходит;
- модель отвечает;
- переключение моделей работает;
- DeepSeek R1 выбирается;
- /models выводит большой список моделей;
- проект перенесён через GitHub на другой ноутбук.
15. Роль проекта в развитии Eidos

OpenRouterLLMClient — первый технический орган будущего Eidos:

канал связи локального Python-кода с внешней языковой моделью

Он не является Eidos, но будет использован как заготовка для его reasoning backend.

16. Web search enhancement

OpenRouterLLMClient now has an opt-in web search mode built on OpenRouter server tools.
The normal chat path still runs without web search by default.

CLI commands:

```text
/web
/web on
/web off
/web status
/askweb <question>
```

Command behavior:

- `/web` and `/web status` print the current web search state.
- `/web on` enables web search for later normal messages in the current REPL session.
- `/web off` disables web search for later normal messages.
- `/askweb <question>` sends exactly one request with web search enabled.
- `/askweb <question>` does not change the global `/web on` or `/web off` state.

Configuration lives in `config/models_config.yaml`:

```yaml
web_search:
  enabled_by_default: false
  tool_type: openrouter:web_search
```

`enabled_by_default` must remain `false` unless the project intentionally changes the default user experience.
`tool_type` is read through `AppConfig.web_search.tool_type`, with `openrouter:web_search` as the supported server tool.

Implementation notes:

- `main.py` stores the current REPL state in `web_search_enabled`.
- Normal messages call `client.chat(..., use_web_search=web_search_enabled)`.
- `/askweb <question>` calls `client.chat(..., use_web_search=True)` for that single question only.
- `logger.py` records `web_search: true` or `web_search: false` in chat and error log entries.
- Logs keep sanitized previews only and must not include API keys or full external search context.

`openrouter_client.py` is responsible for request payload construction.
The helper `build_chat_kwargs(...)` removes any inherited `tools` value from request defaults.
It adds the OpenRouter server tool only when `use_web_search=True`:

```python
tools = [{"type": "openrouter:web_search"}]
```

When `use_web_search=False`, the payload must not include a `tools` key at all.
This preserves the previous normal chat behavior.

Cost and provider limitations:

- Web search may cost more than a normal model request.
- OpenRouter may charge for the search request in addition to model token usage.
- Search results can add extra input tokens to the model prompt.
- OpenRouter server tools are beta, so behavior can change.
- Availability and quality may depend on the selected model and provider.
- The client should surface API errors without breaking the REPL.

Manual comparison examples:

```text
/use deepseek/deepseek-chat
/web off
What changed in OpenRouter web search recently?

/askweb What changed in OpenRouter web search recently?
```

Same model with persistent web mode:

```text
/use deepseek/deepseek-chat
/web off
Answer in one sentence: what is OpenRouter web search?

/web on
Answer in one sentence: what is OpenRouter web search?

/web off
```

Future reuse:

- Reuse `OpenRouterLLMClient.chat(..., use_web_search=...)` instead of duplicating payload logic.
- Reuse `load_config()` and `AppConfig.web_search` for default state and tool type.
- Keep web search as an explicit routing option in future projects.
- In EidosAgent, the OpenRouter backend should accept a per-request web-search flag from the LLM router.
- EidosAgent should not make all requests web-enabled by default; route only tasks that need fresh external information through `use_web_search=True`.
- Future projects should keep the same safety rule: never log full API keys, full search context, or large external result dumps.

17. Application-level timeout and local cancellation

OpenRouterLLMClient applies a configurable local wait timeout around model calls and handles local interruption while waiting for a model response.
The SDK/http timeout is still passed to the OpenAI-compatible client, but the user-facing guarantee comes from an application-level wrapper in `main.py`.

Configuration:

```yaml
request_defaults:
  temperature: 0.7
  max_tokens: 2000
  stream: false
  timeout_seconds: 120
```

`config.py` parses `request_defaults.timeout_seconds` into `AppConfig.request_timeout_seconds`.
The parsed value must be a positive number.
`timeout_seconds` is removed from `request_defaults` before those defaults are used as API payload defaults, so it is not sent as an unsupported chat parameter.

Why SDK timeout alone is not enough:

- The OpenAI SDK `timeout` can behave like a transport timeout for connect/read/write operations.
- A long generation can keep the connection active and still run longer than the value the user set with `/timeout`.
- The CLI needs a wall-clock limit for how long the REPL waits before returning control to the user.

OpenRouter client behavior:

- `OpenRouterLLMClient.chat(..., timeout_seconds=...)` accepts a per-request timeout override.
- `OpenRouterLLMClient.list_models(..., timeout_seconds=...)` also accepts a timeout override.
- `openrouter_client.build_chat_kwargs(...)` still converts the runtime value into the OpenAI SDK `timeout` request option as a transport guard.
- The timeout applies to normal chat and web-search chat because both paths use the same chat helper.
- SDK timeout errors are converted into `OpenRouterTimeoutError`.

Application-level wrapper:

- `main.run_with_application_timeout(...)` runs the blocking model call in a daemon thread.
- The main REPL thread waits on a result queue until the configured wall-clock deadline.
- If the call returns before the deadline, its result or exception is propagated normally.
- If the deadline passes first, `ApplicationTimeout` is raised in the REPL thread.
- On timeout, the daemon worker may continue in the background until the underlying SDK/http call returns.
- The daemon worker is intentionally daemonized so a timed-out request does not prevent `/quit` or process exit.
- Ctrl+C while waiting is still handled by the REPL thread and returns to the prompt without a traceback.

REPL behavior:

```text
/timeout
Current request timeout: 120 seconds

/timeout 30
Request timeout set to 30 seconds
```

Invalid values such as `/timeout abc`, `/timeout -5`, and `/timeout 0` print:

```text
Invalid timeout. Please provide a positive number of seconds.
```

The invalid input does not exit the REPL and does not change the current timeout.
The timeout setting applies to later chat, `/askweb`, and `/models` API calls in the current REPL session.

Ctrl+C handling:

- Ctrl+C at the input prompt still exits the client.
- Ctrl+C while waiting for a model response is handled by `send_chat_message(...)`.
- The client prints `Request interrupted by user. Returned to prompt.`
- The client also prints `Note: provider-side processing may already have started.`
- The REPL returns to the prompt without printing a traceback.

Timeout handling:

- When the application-level timeout fires, the client prints `Request timed out after N seconds. Returned to prompt.`
- The client also prints `Note: provider-side processing may already have started.`
- When the SDK reports a timeout first, the client still handles it without a traceback.
- The REPL returns to the prompt without a traceback.
- The same handling is used for normal requests and web-search requests.

Logging:

- `logger.log_error(...)` accepts `user_message`, `cancelled`, and `timeout` fields.
- `logger.log_chat(...)` accepts `cancelled` and `timeout` fields for failed chat rows.
- Cancelled requests are logged with `cancelled: true`.
- Timed-out requests are logged with `timeout: true`.
- Application-level timeout error rows use `error_type: "ApplicationTimeout"`.
- User prompts are stored only as sanitized previews.
- API keys, full prompts, full web-search context, and large external result dumps must not be logged.

Cancellation limitations:

- Cancellation is local only.
- Application-level timeout is also local only.
- OpenRouter or the upstream provider may continue processing after the client stops waiting.
- Billing may already have started.
- Exact cancellation behavior can depend on the OpenAI SDK and HTTP transport.
- Server tools and web search can still be slower than normal requests.
- A timed-out background request may finish later, but its result is ignored by the CLI.
- Based on manual testing notes, `deepseek/deepseek-r1` with web search is currently not recommended.

Future reuse:

- Future projects should treat cancellation as local control-flow interruption.
- Future projects should treat application-level timeout as local control-flow interruption too.
- EidosAgent must not assume that OpenRouter or the provider cancelled remote generation or billing.
- EidosAgent should pass timeout and cancellation state through its LLM router/backend boundary explicitly.
- Reuse `OpenRouterLLMClient.chat(..., timeout_seconds=..., use_web_search=...)` instead of duplicating request construction.
