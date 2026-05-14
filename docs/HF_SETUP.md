# Настройка Hugging Face CLI и доступа

## 1. Python-зависимости

Из корня проекта:

```powershell
cd "c:\Users\Kudarov Umar\Desktop\search system"
python -m pip install -r requirements.txt
```

После этого в PATH появится команда **`hf`** (через `huggingface_hub`).

Проверка:

```powershell
python -m huggingface_hub.commands.huggingface_cli login --help
```

или при установке entry points:

```powershell
hf --version
hf auth whoami
```

## 2. Токен Hub

Создайте токен в [HF Settings → Access Tokens](https://huggingface.co/settings/tokens).

Рекомендуемые scope для этого проекта:

- **read** — загрузка публичных датасетов и базовые модели
- **write** — push датасетов/моделей (при необходимости замените на fine-grained)

Варианты авторизации:

1. Интерактивно:

   ```powershell
   hf auth login
   ```

2. Переменная окружения (CI / без интерактива):

   ```powershell
   setx HF_TOKEN "hf_xxxxxxxx"
   ```

Не коммитьте токены.

## 3. MCP-сервер в Cursor (`plugin-huggingface-skills-huggingface-skills`)

Для вызова инструментов Hugging Face из чата MCP должен быть **аутентифицирован**:

- выполните в агенте инструмент **`mcp_auth`** для сервера `plugin-huggingface-skills-huggingface-skills` с аргументами `{}` и завершите login в браузере, когда Cursor попросит.

Если аутентификация была пропущена, MCP-операции в IDE будут недоступны до повторной авторизации.

## 4. Первые команды пайплана

После авторизации см. корневой [README](../README.md): подготовка прокси-данных (`scripts/ingest_proxy_text.py`), обучение текстовой головы (`training/train_verbal_classifier.py`).
