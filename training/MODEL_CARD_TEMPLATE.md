---
library_name: transformers
license: apache-2.0
base_model: __BASE_MODEL_PLACEHOLDER__
tags:
  - multilabel-classification
  - verbal-safety
  - ru
  - kz
---

# Verbal risk head (multi-label)

Этот файл шаблон генерируется в `MODEL_CARD.generated.md` после `training/train_verbal_classifier.py`.
Замените `__MODEL_REPO__` и `__METRICS_JSON__` вручную, если запускаете без скрипта.

**Model repo (плейсхолдер):** `__MODEL_REPO__`

## Intended use

- Вход: фрагмент **транскрипта** (после ASR), не сырой аудиопоток на публичном Hub.
- Выход: многометочная оценка классов `neutral_conflict`, `insult_humiliation`, `explicit_threat`, `coercion_harassment`.
- Тревога в продукте: **только совместно** с видео-/аудио-эвристиками и human review (см. `docs/DATASET_SCHEMA_AND_PRIVACY_KZ_RU.md`).

## Evaluation (JSON)

```json
__METRICS_JSON__
```

## Limitations

Прокси-датасеты (например `tweet_eval` → грубая проекция) и синтетика не покрывают bullying intent во времени. Для KZ/RU production нужен доменный gated corpус.
