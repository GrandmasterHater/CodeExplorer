# CodebaseExplorer

Локальный агент для вопросов по кодовой базе. Индексация поддерживает `.cs` и `.java` (UTF-8).

## Запуск

Установите зависимости через `uv sync`. Запустите Ollama и подготовьте chat-модель с поддержкой инструментов и embedding-модель. По умолчанию используются `qwen3.5-9b-32k:latest` и `qwen3-embedding:0.6b`; обе должны быть доступны в Ollama.

```sh
uv run codebaseexplorer index --project "C:/Projects/MyProject"
uv run codebaseexplorer search --project "C:/Projects/MyProject" --query "Где реализована авторизация?"
uv run codebaseexplorer ask --project "C:/Projects/MyProject" --query "Где определён PaymentService?"
```

- `--ollama URL` — адрес Ollama, по умолчанию `http://localhost:11434`.
- `--chat-model NAME` и `--embedding-model NAME` — выбор моделей. При поиске используйте ту же embedding-модель, что и при индексации.
- `index --rebuild` — заменить существующий индекс; `--no-cache` дополнительно пересоздаёт описания.
- Индекс и кэш сохраняются в `.explorer` внутри анализируемого проекта. После изменения исходников индекс нужно перестроить.
- Для поиска по графу нужен доступный в `PATH` CodeGraph CLI с командами `status` и `explore` и заранее подготовленный граф проекта. Семантическая индексация граф не создаёт.

Команды можно запускать из терминала или задачи IDE.
