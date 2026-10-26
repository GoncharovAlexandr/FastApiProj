# Agro Scoring API — Практическая работа №2

Продолжение проекта агроскоринга из ПЗ №1. Входные поля и правила расчёта риска сохранены; добавлены сквозная идентификация, SQLite, batching, очередь, backpressure, timeout и LLM.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Swagger UI: <http://127.0.0.1:8000/docs>

По умолчанию база создаётся в файле `agro_scoring.db`. Путь и параметры можно переопределить переменными окружения: `AGRO_SCORING_DB`, `INFERENCE_TIMEOUT_SECONDS`, `QUEUE_MAX_SIZE`, `MAX_BATCH_SIZE`, `MAX_BATCH_WAIT_MS`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS`.

## Основные endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict` — одиночный инференс с записью в SQLite
- `POST /batch-predict` — explicit batch (1–8 хозяйств; единый request_id; одна запись транзакцией)
- `POST /predict-queued` — динамическая очередь и пакетная обработка
- `GET /queue-status` — состояние очереди и параметры
- `GET /predictions?limit=10&offset=0` — список из SQLite
- `GET /predictions/{request_id}` — отдельный результат; `409`, если request_id содержит несколько результатов batch
- `GET /requests/{request_id}/predictions` — все результаты batch-запроса
- `GET /predictions/by-weekday?weekday=monday&page=1&per_page=5` — пагинация по дню недели
- `POST /llm/chat` — прямой запрос к LLM
- `POST /predict-with-llm` — неизменяемый скоринг + текстовое объяснение LLM

Каждый HTTP-ответ содержит заголовки `X-Request-ID` и `X-Process-Time`. Переданный клиентом `X-Request-ID` сохраняется; если заголовок не передан, создаётся UUID.

## Пример `POST /predict`

```json
{
  "farm_id": "FARM-001",
  "region": "Krasnodar",
  "crop_type": "wheat",
  "area_ha": 500,
  "temperature_avg": 22,
  "precipitation_mm": 300,
  "payment_delay_days": 0,
  "previous_defaults": 0,
  "debt": 1000000
}
```

## Пример `POST /batch-predict`

```json
{
  "items": [
    {
      "farm_id": "FARM-001", "region": "Krasnodar", "crop_type": "wheat",
      "area_ha": 500, "temperature_avg": 22, "precipitation_mm": 300,
      "payment_delay_days": 0, "previous_defaults": 0, "debt": 1000000
    },
    {
      "farm_id": "FARM-002", "region": "Rostov", "crop_type": "sunflower",
      "area_ha": 800, "temperature_avg": 24, "precipitation_mm": 50,
      "payment_delay_days": 45, "previous_defaults": 0, "debt": 1000000
    }
  ]
}
```

## LLM

Используется OpenAI-совместимый endpoint `https://vireonix.ai/v1/chat/completions` с `model=auto` без API-ключа, как указано в методичке. Внешний провайдер может быть временно недоступен или применять ограничения. Ошибка провайдера преобразуется в HTTP `502`, временная недоступность/лимит — `503`, timeout — `504`. Не отправляйте через внешний API секреты или чувствительные данные.

## Проверка

```bash
pytest -q
```

Инференс модели здесь намеренно имитирован детерминированными правилами ПЗ №1 и небольшой асинхронной задержкой, чтобы демонстрировать batch, timeout и dynamic batching без отдельной ML-модели.
