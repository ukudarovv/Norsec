# HF verbal layer — KZ/RU

Этот репозиторий реализует **ML-слой на Hugging Face** из утверждённого плана: датасеты, прокси-ингestion, текстовый multi-label классификатор, карточка моделей, документация ASR/WER и JSON-контракт перед fusion-сервисом.

## Структура

| Путь | Назначение |
|------|-----------|
| [docs/HF_SETUP.md](docs/HF_SETUP.md) | CLI `hf`, токены, MCP `mcp_auth` |
| [docs/DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md](docs/DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md) | utterance/episode, gated/public, матрица ПДн |
| [docs/ASR_BASELINE_AND_WER_PROTOCOL.md](docs/ASR_BASELINE_AND_WER_PROTOCOL.md) | Whisper vs NeMo/ISSAI, WER-срезы, публикация аудио |
| [hf_ml_verbal/schemas.py](hf_ml_verbal/schemas.py) | Pydantic-модели записей Hub |
| [config/proxy_sources.yaml](config/proxy_sources.yaml) | Реестр открытых прокси-источников |
| [scripts/ingest_proxy_text.py](scripts/ingest_proxy_text.py) | Нормализация HF прокси / синтетики → Dataset on-disk |
| [training/train_verbal_classifier.py](training/train_verbal_classifier.py) | fine-tuning + метрики + `MODEL_CARD.generated.md` |
| [demo_gradio.py](demo_gradio.py) | Веб-UI: батч ASR, онлайн, **Этап 1 — YOLO детекция людей (bbox + JSON)**; violence-proxy по кадрам |
| [inference/person_detector.py](inference/person_detector.py) | Этап 1: `PersonDetector`, `draw_person_boxes`, `analyze_live_frame_people` (RGB) |
| [inference/video_person_analyzer.py](inference/video_person_analyzer.py) | Этап 1: выборка кадров из видео + JSON |
| [inference/](inference/) | Извлечение аудио, ASR по окнам, инференс вербальных скорингов |
| [bullying_ai/](bullying_ai/) | **Phase 1+** real-time CV: YOLO люди → ByteTrack → YOLO-pose (опционально: `pip install -r requirements-bullying.txt`) |
| [contracts/verbal_signal_v1.schema.json](contracts/verbal_signal_v1.schema.json) | Контракт JSON для регионального узла |
| [fixtures/synthetic_utterances.jsonl](fixtures/synthetic_utterances.jsonl) | Микро-демонстрационный синтетический набор |

## Быстрый старт

```powershell
cd "c:\Users\Kudarov Umar\Desktop\search system"
python -m pip install -r requirements.txt
hf auth login
```

Подготовка датасета (синтетика или публичные прокси — см. флаги скрипта):

```powershell
python scripts\ingest_proxy_text.py --source synthetic --output data\processed\proxy\synthetic
```

Обучение (CPU/GPU; для smoke достаточно CPU, но будет медленнее):

```powershell
python training\train_verbal_classifier.py --dataset-path data\processed\proxy\synthetic --output-dir checkpoints\verbal-smoke --epochs 8 --batch-size 2 --max-length 96
```

Артефакты чекпоинта и метрики: `checkpoints/verbal-smoke/eval_metrics.json`, `MODEL_CARD.generated.md`.

## Демо: веб-камера / файл и оценка вербального риска

В **батч**-вкладке по умолчанию включён опциональный **визуальный прокси физической агрессии**: равномерная выборка кадров и HF image-classification (по умолчанию `locih/violence_classification`; в поле UI можно указать другой id). Репозитории **без поля `model_type` в config.json** (как старые выкладки ViT) с **transformers 5.x** могут не загрузиться. Это **не** полный школьный буллинг и не замена вербальному анализу. Поток со **звуком**: **ffmpeg** извлекает **аудио 16 kHz** → **Whisper** по окнам → **текстовая** голова из `checkpoints/...`.

Нужен чекпоинт в `checkpoints/verbal-latest` или путь в поле UI.

```powershell
python -m pip install -r requirements.txt
python demo_gradio.py
```

Онлайн-Whisper по микрофону использует **torchaudio** для ресемплинга (пакет уже входит в `requirements.txt` после `pip install -r`).

В браузере откройте URL из консоли (порт может сдвигаться), разрешите доступ к **камере и микрофону**.

- **Файл / веб-камера (батч)** — запись или файл, затем «Запустить разбор».
- **Этап 1 — Детекция людей** — только **YOLO** (`ultralytics`, по умолчанию `yolov8n.pt`): bbox, количество людей, JSON; без Whisper и без violence-proxy.
- **Онлайн — микрофон + камера** — поток микрофона (~2 с) и с камеры (~3 с): **violence-proxy (ViT)** и **прокси опасных предметов (CLIP zero-shot, ~350 MB при первом запуске)** — см. галочки во вкладке; не детектор с рамками.

Юнит-тесты визуального proxy (без скачивания весов) и опционально smoke с Hub:

```powershell
python -m unittest discover -s tests -p "test*.py" -v
# или только Этап 1:
pytest tests/test_person_detector.py
```

Полный HF-smoke (CPU, скачивание модели ~350MB): `set RUN_HF_SLOW_TESTS=1` затем снова команду выше.

## Hub push

- Датасет: `--push-to-hub username/dataset` в `scripts/ingest_proxy_text.py` после `hf auth login`.
- Модель: `--push-to-hub-model` в training-скрипте.
