import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import time
import uuid

import httpx
from fastapi import Body, FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import (
    DB_PATH,
    INFERENCE_TIMEOUT_SECONDS,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    MAX_BATCH_SIZE,
    MAX_BATCH_WAIT_MS,
    QUEUE_MAX_SIZE,
)
from llm_client import LLMServiceError, chat_completion
from model import model_predict_batch
from schemas import (
    BatchPredictionResponse,
    BatchRequest,
    ErrorResponse,
    FarmRequest,
    HealthResponse,
    LLMChatRequest,
    LLMChatResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictWithLLMResponse,
    QueueStatusResponse,
    WeekdayPageResponse,
)
from services import ALLOWED_REGIONS, VALID_RISK_LEVELS, WEEKDAYS, build_prediction
from storage import Database, StorageUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agro_scoring")


class QueuedInference:
    def __init__(self, request_id: str, farm: FarmRequest, future: asyncio.Future):
        self.request_id = request_id
        self.farm = farm
        self.future = future


async def inference_worker(app: FastAPI) -> None:
    """Собирает динамический batch и обрабатывает его одним вызовом модели."""
    queue: asyncio.Queue = app.state.inference_queue
    while True:
        first: QueuedInference = await queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + (MAX_BATCH_WAIT_MS / 1000)

        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            batch.append(item)

        try:
            scores = await asyncio.wait_for(
                model_predict_batch([item.farm for item in batch]),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
            results = [
                build_prediction(item.request_id, item.farm, score)
                for item, score in zip(batch, scores)
            ]
            await app.state.db.save_many(results, source="predict-queued")
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)
        except asyncio.TimeoutError:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(
                        HTTPException(
                            status_code=504,
                            detail="Inference timeout",
                        )
                    )
        except StorageUnavailable:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(
                        HTTPException(
                            status_code=503,
                            detail="Prediction storage is unavailable",
                        )
                    )
        except Exception:
            logger.exception("Dynamic batch processing failed")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(
                        HTTPException(status_code=500, detail="Inference failed")
                    )
        finally:
            for _ in batch:
                queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(DB_PATH)
    await db.initialize()
    app.state.db = db
    app.state.llm_client = httpx.AsyncClient(
        base_url=LLM_BASE_URL.rstrip("/") + "/",
        timeout=LLM_TIMEOUT_SECONDS,
        headers={"Content-Type": "application/json"},
    )
    app.state.inference_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    worker = asyncio.create_task(inference_worker(app), name="dynamic-batching-worker")
    app.state.inference_worker = worker
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        await app.state.llm_client.aclose()


app = FastAPI(
    title="Agro Scoring API — Practical Work 2",
    description=(
        "Расширенный AI-сервис агроскоринга: request_id middleware, SQLite, "
        "explicit/dynamic batching, backpressure, timeout и внешняя LLM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID")
    request_id = supplied_id.strip() if supplied_id and supplied_id.strip() else str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error request_id=%s method=%s path=%s",
            request_id, request.method, request.url.path,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.6f}"
    logger.info(
        "request_id=%s method=%s path=%s status=%s process_time=%.6f",
        request_id, request.method, request.url.path,
        response.status_code, process_time,
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "request_id": request_id},
    )


@app.exception_handler(StorageUnavailable)
async def storage_error_handler(request: Request, exc: StorageUnavailable):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "request_id": request_id},
    )


@app.exception_handler(LLMServiceError)
async def llm_error_handler(request: Request, exc: LLMServiceError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc), "request_id": request_id},
    )


@app.get(
    "/health", response_model=HealthResponse,
    summary="Проверка доступности API",
)
async def health():
    return {"status": "ok"}


@app.get(
    "/model-info", response_model=ModelInfoResponse,
    summary="Информация о модели агроскоринга",
)
async def model_info():
    return {
        "model_name": "agro-risk-model",
        "model_version": "1.0",
        "model_type": "rule-based risk-scoring",
        "status": "ready",
    }


async def validate_region(request: FarmRequest) -> None:
    if request.region not in ALLOWED_REGIONS:
        raise HTTPException(status_code=400, detail="Unknown region")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценка риска одного хозяйства",
    description="Считает риск, сохраняет результат в SQLite и возвращает прогноз.",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def predict(request: Request, data: FarmRequest = Body(...)):
    await validate_region(data)
    request_id = request.state.request_id
    try:
        scores = await asyncio.wait_for(
            model_predict_batch([data]), timeout=INFERENCE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    result = build_prediction(request_id, data, scores[0])
    await request.app.state.db.save_many([result], source="predict")
    return result


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Явный пакетный инференс",
    description=f"Обрабатывает до {MAX_BATCH_SIZE} хозяйств одним вызовом модели и одной транзакцией SQLite.",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def batch_predict(request: Request, body: BatchRequest):
    for farm in body.items:
        await validate_region(farm)
    try:
        scores = await asyncio.wait_for(
            model_predict_batch(body.items), timeout=INFERENCE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    request_id = request.state.request_id
    results = [
        build_prediction(request_id, farm, score)
        for farm, score in zip(body.items, scores)
    ]
    await request.app.state.db.save_many(results, source="batch-predict")
    return {"request_id": request_id, "items": results}


@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Инференс через очередь с dynamic batching",
    description="Принимает одно хозяйство; worker объединяет запросы в batch до 8 элементов или на 50 мс.",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               429: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
               504: {"model": ErrorResponse}},
)
async def predict_queued(request: Request, data: FarmRequest):
    await validate_region(data)
    queue: asyncio.Queue = request.app.state.inference_queue
    if queue.full():
        raise HTTPException(status_code=429, detail="Inference queue is full")
    future = asyncio.get_running_loop().create_future()
    item = QueuedInference(request.state.request_id, data, future)
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Inference queue is full")
    return await future


@app.get(
    "/queue-status", response_model=QueueStatusResponse,
    summary="Состояние очереди dynamic batching",
)
async def queue_status(request: Request):
    return {
        "queue_size": request.app.state.inference_queue.qsize(),
        "queue_max_size": QUEUE_MAX_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_batch_wait_ms": MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
    }


@app.get(
    "/predictions", response_model=list[PredictionResponse],
    summary="Чтение прогнозов из SQLite",
)
async def predictions(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    risk_level_filter: str | None = Query(default=None, alias="risk_level"),
):
    if risk_level_filter is not None and risk_level_filter not in VALID_RISK_LEVELS:
        raise HTTPException(status_code=400, detail="Invalid risk_level. Use low, medium or high.")
    # Читаем из БД. Фильтр по риску применяем после ограниченного чтения,
    # а при указанном фильтре расширяем запрос в рамках лимита до 100 строк.
    return await request.app.state.db.list_predictions(
        limit=limit, offset=offset, risk_level=risk_level_filter
    )


@app.get(
    "/predictions/by-weekday", response_model=WeekdayPageResponse,
    summary="Пагинация прогнозов по дню недели",
    description="Offset-пагинация через LIMIT/OFFSET и COUNT(*). Дни недели указаны на английском.",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
)
async def predictions_by_weekday(
    request: Request,
    weekday: str = Query(..., description="monday, tuesday, wednesday, thursday, friday, saturday, sunday"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=5, ge=1, le=100),
):
    weekday = weekday.lower()
    if weekday not in WEEKDAYS:
        raise HTTPException(status_code=400, detail="Unknown weekday")
    return await request.app.state.db.list_by_weekday(weekday, page, per_page)


@app.get(
    "/predictions/{request_id}", response_model=PredictionResponse,
    summary="Получение одного прогноза по request_id",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
)
async def get_prediction(request_id: str, request: Request):
    rows = await request.app.state.db.get_by_request_id(request_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if len(rows) > 1:
        raise HTTPException(
            status_code=409,
            detail="request_id is associated with multiple predictions; use /requests/{request_id}/predictions",
        )
    return rows[0]


@app.get(
    "/requests/{request_id}/predictions", response_model=list[PredictionResponse],
    summary="Получение всех результатов batch-запроса",
)
async def get_request_predictions(request_id: str, request: Request):
    rows = await request.app.state.db.get_by_request_id(request_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return rows



@app.post(
    "/llm/chat", response_model=LLMChatResponse,
    summary="Запрос к внешней LLM",
    responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
               503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def llm_chat(request: Request, body: LLMChatRequest):
    result = await chat_completion(
        request.app.state.llm_client,
        body.prompt,
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    return {"request_id": request.state.request_id, **result}


@app.post(
    "/predict-with-llm", response_model=PredictWithLLMResponse,
    summary="Агроскоринг с текстовым объяснением LLM",
    description="Числовой скор и категория рассчитываются приложением; LLM формирует только объяснение и не меняет результат скоринга.",
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               502: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
               504: {"model": ErrorResponse}},
)
async def predict_with_llm(request: Request, data: FarmRequest):
    await validate_region(data)
    request_id = request.state.request_id
    try:
        scores = await asyncio.wait_for(
            model_predict_batch([data]), timeout=INFERENCE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    # Оценка фиксируется детерминированной моделью до обращения к LLM.
    result = build_prediction(request_id, data, scores[0])
    prompt = (
        "Объясни результат агроскоринга для оператора банка в 2-4 предложениях. "
        "Не меняй и не пересчитывай risk_score или risk_level. "
        f"Данные: farm_id={data.farm_id}; регион={data.region}; культура={data.crop_type}; "
        f"площадь={data.area_ha} га; средняя температура={data.temperature_avg}; "
        f"осадки={data.precipitation_mm} мм; задержка платежа={data.payment_delay_days} дней; "
        f"предыдущие дефолты={data.previous_defaults}; долг={data.debt}. "
        f"Результат модели: risk_score={result['risk_score']}, "
        f"risk_level={result['risk_level']}, рекомендация={result['recommendation']}."
    )
    explanation = await chat_completion(
        request.app.state.llm_client,
        prompt,
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=300,
    )
    await request.app.state.db.save_many([result], source="predict-with-llm")
    return {
        **result,
        "explanation": explanation["answer"],
        "requested_model": explanation["requested_model"],
        "returned_model": explanation["returned_model"],
        "latency_ms": explanation["latency_ms"],
    }
