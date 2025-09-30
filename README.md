# X5 NER Service

FastAPI сервис для выделения сущностей в коротких товарных описаниях (NER). Загружает совместимые с HuggingFace артефакты из `ARTIFACTS_DIR` и предоставляет REST API.

## Содержание
- Установка и запуск (локально и в Docker)
- Конфигурация (переменные окружения)
- Эндпоинты и примеры запросов
- Артефакты модели и внешние источники
- Нагрузочное тестирование и оффлайн-оценка

## Установка и запуск

### Требования
- Python 3.10+
- Linux/macOS/Windows
- Для Docker-режима: Docker 24+ и docker compose

### Локальный запуск (CPU)
1) Создайте и активируйте виртуальное окружение
```
python -m venv .venv && source .venv/bin/activate
```
2) Установите зависимости
```
pip install -r requirements-prod.txt   # минимальный рантайм
# или для разработки (ноутбуки, тесты, линтеры):
pip install -r requirements-dev.txt
```
3) Подготовьте артефакты модели
```
mkdir -p artifacts
# Скопируйте файлы модели в ./artifacts или укажите путь через ARTIFACTS_DIR
```
4) Запустите сервер
```
uvicorn service.main:app --host 0.0.0.0 --port 8000
```
5) Откройте документацию
```
http://localhost:8000/docs
```

### Запуск в Docker
Сборка и запуск:
```
docker build -t x5-ner:local .
docker run --rm -p 8000:8000 \
  -e ARTIFACTS_DIR=/app/artifacts/rubert-base-cased/20250930-113112 \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$PWD/artifacts:/app/artifacts:ro" x5-ner:local
```
Docker Compose (рекомендуется для локальных проверок):
```
docker compose up --build
```
См. `compose.yaml` для переменных окружения, volume и healthcheck.

## Конфигурация
Ключевые переменные окружения (полный список — в `docs/ENV_VARS.md`):
- ARTIFACTS_DIR — путь к артефактам модели (по умолчанию `./artifacts`)
- X5_FORCE_DEVICE или FORCE_DEVICE — `cpu`/`cuda`/`mps` (если доступно)
- PREDICT_MAX_CONCURRENCY — ограничение параллелизма инференса на процесс
- MICRO_BATCH_ENABLED — `true|false`
- MICRO_BATCH_MAX_SIZE, MICRO_BATCH_MAX_WAIT_MS, MICRO_BATCH_HARD_TIMEOUT_MS — тюнинг микробатчинга
- CACHE_ENABLED, CACHE_MAX_SIZE, CACHE_TTL_SECONDS — параметры кэша
- TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS, MKL_NUM_THREADS, TORCH_NUM_THREADS, TORCH_NUM_INTEROP_THREADS — тюнинг потоков

## Эндпоинты
- GET `/health` — статус сервиса, устройство, путь к артефактам и статистика кэша
- POST `/api/predict` — предсказание для одной строки
- POST `/api/predict_batch` — предсказания для списка строк
- POST `/warmup` — опциональный прогрев модели
- GET `/cache/stats`, DELETE `/cache/clear`, GET `/cache/info` — кэш

Примеры запросов:
```
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"cola 500ml 5%"}'

curl -s -X POST http://localhost:8000/api/predict_batch \
  -H 'Content-Type: application/json' \
  -d '{"inputs":["cola 500ml","фанта 1л 6%"]}'
```
Ответы содержат список BIO-спанов на уровне слов:
```
[
  {"start_index":0,"end_index":4,"entity":"B-BRAND"},
  {"start_index":5,"end_index":10,"entity":"B-VOLUME"},
  {"start_index":11,"end_index":13,"entity":"B-PERCENT"}
]
```

## Артефакты модели и внешние источники
- Предобученная модель: `DeepPavlov/rubert-base-cased` (HuggingFace)
- Формат артефактов (совместим с HF):
  - config.json, tokenizer.json, tokenizer_config.json, vocab.txt, special_tokens_map.json
  - model.safetensors (или иные веса, поддерживаемые Transformers)
  - (опц.) id2label в config.json или отдельный mapping
- Папки с примерами артефактов: `artifacts/` (см. README в `docs/ARTIFACTS.md`)

## Нагрузочное тестирование
Есть утилита для нагрузочного теста `/api/predict`:
```
python scripts/load_test_predict.py \
  --base_url http://localhost:8000 \
  --input examples/sample_input.csv \
  --concurrency 100 --requests-per-client 50 --timeout 1.0 \
  --log_requests eval_out/load_requests.csv
```

## Оффлайн-оценка
Скрипт оффлайн-оценки предсказаний сервиса:
```
python scripts/evaluate_service.py \
  --input examples/annotated_sample.csv \
  --output_dir eval_out \
  --base_url http://localhost:8000 \
  --batch_size 32
```
Выходы: `eval_results.csv`, `eval_report.html`, `eval_stats.json`.

## Лицензия
<!-- Добавьте файл лицензии (MIT/Apache-2.0) в корень репозитория. -->
