# Подробное описание реализованных возможностей (обзор кода)

Документ фиксирует основные доработки пайплайна «видео / микрофон → ASR → вербальная голова → визуальные прокси» в репозитории **search system**: что сделано, в каких файлах и как это запускается. Не заменяет `README.md`, а дополняет его для разработчика.

---

## 1. Общая архитектура

| Режим | Вход | Поток |
|--------|------|--------|
| **Батч** | Файл или запись с веб-камеры в Gradio | FFmpeg → (при наличии звука) WAV 16 kHz → Whisper по окнам → вербальная голова (multilabel) → опционально выборка кадров → ViT violence / сводка инцидента |
| **Онлайн** | Поток микрофона + поток превью камеры | Периодически: окно аудио → Whisper → verbal; кадр камеры → violence ViT, CLIP «опасные предметы», Haar + эмоции; сводка «драка и конфликт» сверху лога |

Точка входа UI: [`demo_gradio.py`](../demo_gradio.py).

---

## 2. Аудио и FFmpeg

**Файл:** [`inference/audio_extract.py`](../inference/audio_extract.py)

- Определение наличия аудиодорожки в контейнере (через FFmpeg).
- Извлечение моно WAV 16 kHz для ASR.
- **Shim для Gradio/ffmpy:** бинарь в PATH должен называться `ffmpeg.exe`; при необходимости копируется исполняемый файл из `imageio-ffmpeg`, чтобы `ffmpy` находил его на Windows.

---

## 3. Разбор видео (батч): `analyze_media`

**Файл:** [`inference/analyze_media.py`](../inference/analyze_media.py)

- Если **нет аудиодорожки:** отчёт без ASR и без вербальных скоров по сегментам; при включённой галочке — визуальный слой по кадрам; в JSON — `skipped_asr_reason: "no_audio_stream"`.
- Если **звук есть:** чекпоинт вербальной головы обязателен (`checkpoints/verbal-latest` или путь из UI), извлечение WAV, pipeline Whisper, `verbal_scores_batch` по каждому текстовому окну, таблица сегментов, опционально `analyze_visual_aggression`.
- **Сводка «драка и конфликт»** (после максимумов по классам и до таблицы сегментов при наличии звука; для видео без звука — после блока «вербальный слой недоступен» и до детального визуального раздела):
  - логика в [`inference/incident_signals.py`](../inference/incident_signals.py);
  - в JSON добавляется объект **`incident`** с полями вроде `incident_physical_proxy`, `incident_verbal_conflict`, `incident_verbal_escalation`, `incident_verbal_aggregate`, `incident_escalation_note`.

---

## 4. Визуальный прокси агрессии (драка / насилие)

**Файл:** [`inference/visual_aggression.py`](../inference/visual_aggression.py)

- Равномерная выборка кадров из видео (OpenCV), для каждого кадра — HF `image-classification` (по умолчанию **`locih/violence_classification`**).
- Из top-k предсказаний считается **`violence_probability`**: эвристика по подстрокам в названии класса (агрессия vs non-violence / safe / neutral и т.д.).
- Расширенный набор подсказок в названиях классов включает в том числе: **fight, fighting, brawl, punch, scuffle, riot, assault, unsafe, violence, …** — чтобы разные Hub-модели давали согласованный прокси-скор.
- Функции **`analyze_visual_aggression`**, **`visual_results_markdown_section`**, дисклеймер в markdown.

**Онлайн:** тот же классификатор по одному кадру в [`inference/live_visual.py`](../inference/live_visual.py) с дедупликацией по грубой сигнатуре кадра (`_frame_signature`).

---

## 5. Опасные предметы (CLIP, zero-shot)

**Файл:** [`inference/dangerous_objects.py`](../inference/dangerous_objects.py)

- Один вызов пайплайна **`zero-shot-image-classification`** с набором кандидатов: одна «безопасная» сцена + **несколько** англоязычных промптов под оружие/опасные предметы (нож, огнестрел, дубинка, баллончик и т.д.).
- **`weapon_proxy`** = сумма softmax-вероятностей по всем «weapon family» промптам; отдельно выводятся топ-1 по всем классам и подсказка **`weapon_hint`** по лучшему weapon-промпту.
- По умолчанию модель: **`openai/clip-vit-base-patch32`**.

---

## 6. Лица и настроение (онлайн)

**Файл:** [`inference/people_mood.py`](../inference/people_mood.py)

- **Количество анфасных лиц:** каскад Haar OpenCV (`haarcascade_frontalface_default.xml`) — это не полный подсчёт людей в кадре.
- **Эмоция** по **крупнейшему** вырезу лица: HF ViT по умолчанию **`dima806/facial_emotions_image_detection`**.
- Ограничения задокументированы в markdown (не диагноз, не анализ позы).

---

## 7. Ускорение онлайн-инференса по кадру

**Файл:** [`inference/frame_resize.py`](../inference/frame_resize.py)

- **`downscale_rgb_max_long_side`:** уменьшение длинной стороны кадра (OpenCV `INTER_AREA`) перед подачей в ViT / CLIP / Haar.

**Файл:** [`inference/live_visual.py`](../inference/live_visual.py)

- После валидации RGB кадра дедуп по сигнатуре считается по **полному** разрешению; инференс идёт на **`rgb_inf`**, полученном даунскейлом.
- Переменная окружения **`LIVE_CAMERA_MAX_SIDE`** (по умолчанию **448**); значение **0** отключает даунскейл.

**Файл:** [`demo_gradio.py`](../demo_gradio.py)

- Интервал стрима камеры: **`LIVE_CAM_STREAM_SEC`** (секунды, с разумным clamp), читается в **`_live_cam_stream_every_seconds()`**.

---

## 8. Потоковый микрофон (онлайн)

**Файл:** [`inference/live_mic.py`](../inference/live_mic.py)

- Загрузка Whisper + вербальной головы с кэшированием.
- **`torchaudio`** / окружение для языка (**`LIVE_WHISPER_LANGUAGE`**, по умолчанию русский), **`LIVE_MIC_RMS_FLOOR`** для отсечения тишины.
- Фильтр **`_likely_script_hallucination`** против типичного мусора Whisper (CJK и т.п.) на длинных отрезках без кириллицы.
- Декодирование Whisper с **`task=transcribe`** и указанием языка.

---

## 9. Обучение вербальной головы и итеративное улучшение

**Файл:** [`training/train_verbal_classifier.py`](../training/train_verbal_classifier.py)

- Обучение **multi-label** классификации (классы из [`hf_ml_verbal/label_config.py`](../hf_ml_verbal/label_config.py)).
- При **`--improve-max-rounds > 1`**: один длинный **`trainer.train()`** с суммарным числом эпох **`epochs × improve-max-rounds`**, **`EarlyStoppingCallback`** по метрике **`macro_f1_report`** (patience и min-delta задаются флагами).
- Для этого режима по умолчанию **`save_only_model=True`** в `TrainingArguments`, чтобы реже ловить сбои **`torch.save`** оптимизатора под Windows; при необходимости полных чекпоинтов — **`--save-full-checkpoints`**.
- **`dataloader_pin_memory=False`** для тихих прогонов на CPU.
- Метрики и **`iteration_log`** пишутся в **`eval_metrics.json`**.
- Отдельно сохраняются лучшие веса в подкаталог **`best_macro_f1`** и копия **`exported_best_macro_f1`**.

Ранее использовавшийся цикл из нескольких **`train(resume_from_checkpoint)`** заменён из-за ошибок состояния Trainer и сохранений на диске.

---

## 10. Драки и конфликты — явная сводка

**Файл:** [`inference/incident_signals.py`](../inference/incident_signals.py)

- **`verbal_conflict_score`**: агрегаты по четырём вербальным классам (фокус «спор/оскорбление» vs «угроза/принуждение», общий max).
- **`fuse_batch_incident`**: объединение максимального видеопрокси (`visual.summary.max` или эквивалент) и max-скоров по сегментам / нулей при отсутствии звука.
- **`prepend_online_incident`**: блок markdown в начале онлайн-отчёта в Gradio (макс по речевому журналу + макс `violence_probability` по журналу камеры).

**Интеграция:**

- Батч: [`analyze_media.py`](../inference/analyze_media.py) → markdown + `payload["incident"]`.
- Онлайн: [`demo_gradio.py`](../demo_gradio.py) оборачивает вывод **`format_quad_live_markdown`** через **`prepend_online_incident`**.

**Документация схемы данных:** дополнена строка в [`docs/DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md`](DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md) про связку вербальных меток и видеопрокси без bbox.

---

## 11. Тесты

| Файл | Назначение |
|------|------------|
| [`tests/test_visual_aggression.py`](../tests/test_visual_aggression.py) | Эвристики и при необходимости медленные HF-smoke (`RUN_HF_SLOW_TESTS`) |
| [`tests/test_dangerous_objects.py`](../tests/test_dangerous_objects.py) | Агрегирование мультиклассового CLIP, обратная совместимость `weapon_proxy_score_from_topk` |
| [`tests/test_people_mood.py`](../tests/test_people_mood.py) | Haar: пустой кадр без лиц |
| [`tests/test_incident_signals.py`](../tests/test_incident_signals.py) | Вербальная сводка, fuse, журнал violence онлайн |

---

## 12. Переменные окружения (сводка)

| Переменная | Где используется | Смысл |
|------------|------------------|--------|
| `LIVE_CAMERA_MAX_SIDE` | `live_visual` | Даунскейл длинной стороны кадра (0 = выкл.) |
| `LIVE_CAM_STREAM_SEC` | `demo_gradio` | Период обновления превью камеры в онлайне |
| `LIVE_WHISPER_LANGUAGE` | `live_mic` | Язык декодирования Whisper в онлайне |
| `LIVE_MIC_RMS_FLOOR` | `live_mic` | Порог громкости; ниже — тик пропускается |
| `HF_TOKEN` / `huggingface-cli login` | Hub | Загрузка моделей без лишних ограничений rate limit |

---

## 13. Ограничения и честность перед заказчиком

- Визуальный слой — **классификация всего кадра**, без детекции людей и ударов; спорт и быстрое движение дают ложные срабатывания.
- CLIP по текстовым промптам — не промышленный детектор оружия.
- Вербальный конфликт возможен только **после ASR**; качество зависит от микрофона и модели Whisper.
- Метки эмоций и «лица» — прокси, не медицина и не правовая квалификация.

---

## 14. Как воспроизвести ключевые сценарии

**Gradio:**

```powershell
cd "путь\к\search system"
python demo_gradio.py
```

**Обучение вербальной головы (пример из docstring скрипта):**

```powershell
python training\train_verbal_classifier.py --dataset-path data\processed\proxy\synthetic --output-dir checkpoints\verbal-smoke --epochs 2 ...
```

**Инжест синтетического датасета (если нужен):**

```powershell
python scripts\ingest_proxy_text.py --source synthetic --output data\processed\proxy\synthetic
```

---

*Документ создан для навигации по реализации; при изменении API обновляйте соответствующие разделы и таблицы.*
