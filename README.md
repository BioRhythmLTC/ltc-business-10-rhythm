# X5 NER Service

FastAPI сервис для выделения сущностей в коротких товарных описаниях (NER). Загружает совместимые с HuggingFace артефакты из `ARTIFACTS_DIR` и предоставляет REST API.

## Содержание
- Установка и запуск (локально и в Docker)
- Конфигурация (переменные окружения)
- Эндпоинты и примеры запросов
- Артефакты модели и внешние источники
- Нагрузочное тестирование и оффлайн-оценка
- Полная API-спецификация и гайды

## Установка и запуск

### Требования
- Python 3.9.6
- Linux/macOS/Windows
- Для Docker-режима: Docker 24+ и docker compose

### Локальный запуск (CPU)
1) Создайте и активируйте виртуальное окружение
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```
2) Установите зависимости
```
pip install -r requirements-prod.txt   # минимальный рантайм для API сервиса
# или для разработки/исследований (ноутбуки, метрики и т.п.):
pip install -r requirements.txt        # полный стек для .ipynb
# для локальной разработки сервиса (линтеры/тесты):
pip install -r requirements-dev.txt
```
3) Подготовьте артефакты модели

- 1) Скачайте архив модели с диска `rubert_base_cased_20251002_111705.zip` из [Google Drive](https://drive.google.com/drive/folders/13WxzEEXwLt8el3-_sm_XkO_0YqUde5EA?usp=sharing).
- 2) Разархивируйте его в каталог проекта так, чтобы итоговый путь был: `./artifacts/rubert-base-cased/20251002-111705`.

```bash
mkdir -p artifacts
# Переместите распакованную папку rubert-base-cased/20251002-111705 в ./artifacts
```

Проверьте, что в папке есть файлы:
- `./artifacts/rubert-base-cased/20251002-111705/config.json`
- `./artifacts/rubert-base-cased/20251002-111705/model.safetensors`
- `./artifacts/rubert-base-cased/20251002-111705/tokenizer.json`
- `./artifacts/rubert-base-cased/20251002-111705/vocab.txt`
- `./artifacts/rubert-base-cased/20251002-111705/special_tokens_map.json`
4) Запустите сервер
```bash
cd /path/to/X5

export ARTIFACTS_DIR="$(pwd)/artifacts/rubert-base-cased/20251002-111705"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
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
  -e ARTIFACTS_DIR=/app/artifacts/rubert-base-cased/20251002-111705 \
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
- PREDICT_FAIL_SAFE — если `true`, при неожиданных ошибках эндпоинты вернут пустой результат (200)
- MAX_INPUT_CHARS — ограничение длины входной строки (обрезка)
- ROOT_PUBLIC — если `true`, корень `/` отдаёт информацию о сервисе, иначе 404
- MICRO_BATCH_ENABLED — `true|false`
- MICRO_BATCH_MAX_SIZE, MICRO_BATCH_MAX_WAIT_MS, MICRO_BATCH_HARD_TIMEOUT_MS, MICRO_BATCH_QUEUE_MAXSIZE — тюнинг микробатчинга
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

## Полная документация
- Архитектура решения: см. [docs/solution.md](docs/solution.md)
- API-спецификация: см. [docs/API.md](docs/API.md)
- Переменные окружения: см. [docs/ENV_VARS.md](docs/ENV_VARS.md)
- Артефакты модели: см. [docs/ARTIFACTS.md](docs/ARTIFACTS.md) и [docs/ARTIFACTS_DOWNLOAD.md](docs/ARTIFACTS_DOWNLOAD.md)
- Развёртывание: см. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Настройка производительности: см. [docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md)

## Разделение requirements
- `requirements-prod.txt` — минимальные зависимости для API сервиса (FastAPI, Transformers, Torch и т.д.).
- `requirements.txt` — расширенные зависимости для исследований и ноутбуков (.ipynb), метрик, обучения и анализа.
- `requirements-dev.txt` — инструменты разработки (pytest, линтеры и прочее), опционально поверх выбранного набора выше.


