import uuid

from fastapi import Body, FastAPI, HTTPException, Query, status

from model import calculate_risk
from schemas import (
    FarmRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from services import recommendation, risk_level
from storage import (
    get_prediction,
    get_predictions,
    save_prediction,
)


app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API сервиса оценки кредитного риска "
        "сельскохозяйственных предприятий"
    ),
    version="1.0.0",
)

ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol",
}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    description=(
        "Проверяет доступность API. "
        "При успешной работе возвращает статус ok."
    ),
)
def health() -> HealthResponse:
    return {
        "status": "ok"
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description=(
        "Возвращает название, версию и тип модели "
        "агроскоринга."
    ),
)
def model_info() -> ModelInfoResponse:
    return {
        "model_name": "agro-risk-model",
        "model_version": "1.0",
        "model_type": "risk-scoring",
        "status": "ready",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Оценка риска хозяйства",
    description=(
        "Выполняет расчёт риска сельскохозяйственного "
        "предприятия на основе финансовых и погодных показателей."
    ),
    responses={
        400: {
            "description": "Некорректный регион"
        },
        422: {
            "description": "Ошибка валидации входных данных"
        },
    },
)
def predict(
    request: FarmRequest = Body(
        ...,
        openapi_examples={
            "low_risk": {
                "summary": "Низкий риск",
                "description": (
                    "Нет задержек платежей, "
                    "нет предыдущих дефолтов, "
                    "умеренный долг и достаточное количество осадков."
                ),
                "value": {
                    "farm_id": "FARM-001",
                    "region": "Krasnodar",
                    "crop_type": "wheat",
                    "area_ha": 500,
                    "temperature_avg": 22,
                    "precipitation_mm": 300,
                    "payment_delay_days": 0,
                    "previous_defaults": 0,
                    "debt": 1_000_000,
                },
            },
            "medium_risk": {
                "summary": "Средний риск",
                "description": (
                    "Есть существенная задержка платежа "
                    "и недостаточное количество осадков."
                ),
                "value": {
                    "farm_id": "FARM-002",
                    "region": "Rostov",
                    "crop_type": "sunflower",
                    "area_ha": 800,
                    "temperature_avg": 24,
                    "precipitation_mm": 50,
                    "payment_delay_days": 45,
                    "previous_defaults": 0,
                    "debt": 1_000_000,
                },
            },
            "high_risk": {
                "summary": "Высокий риск",
                "description": (
                    "Есть длительная задержка платежей, "
                    "предыдущий дефолт, большой долг "
                    "и недостаток осадков."
                ),
                "value": {
                    "farm_id": "FARM-003",
                    "region": "Stavropol",
                    "crop_type": "corn",
                    "area_ha": 1200,
                    "temperature_avg": 26,
                    "precipitation_mm": 70,
                    "payment_delay_days": 60,
                    "previous_defaults": 1,
                    "debt": 7_000_000,
                },
            },
        },
    ),
) -> PredictionResponse:

    # Бизнес-валидация региона
    if request.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown region",
        )

    score = calculate_risk(request)

    level = risk_level(score)

    request_id = str(uuid.uuid4())

    result = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": "1.0",
    }

    save_prediction(request_id, result)

    return result


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получение прогноза по ID",
    description=(
        "Возвращает ранее сохранённый результат "
        "по уникальному request_id."
    ),
    responses={
        404: {
            "description": "Прогноз с указанным ID не найден"
        },
        422: {
            "description": "Некорректный request_id"
        },
    },
)
def get_prediction_by_id(request_id: str) -> PredictionResponse:
    result = get_prediction(request_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    return result


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="Получение списка прогнозов",
    description=(
        "Возвращает сохранённые прогнозы. "
        "Поддерживается ограничение количества результатов "
        "и фильтрация по уровню риска."
    ),
    responses={
        400: {
            "description": "Некорректное значение risk_level"
        },
        422: {
            "description": "Некорректное значение limit"
        },
    },
)
def get_predictions_list(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Максимальное количество результатов",
        examples=[10],
    ),
    risk_level_filter: str | None = Query(
        default=None,
        alias="risk_level",
        description="Фильтр: low, medium или high",
        examples=["high"],
    ),
) -> list[PredictionResponse]:

    values = get_predictions()

    # Фильтрация по уровню риска
    if risk_level_filter is not None:

        if risk_level_filter not in {
            "low",
            "medium",
            "high",
        }:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid risk_level. "
                    "Use low, medium or high."
                ),
            )

        values = [
            item
            for item in values
            if item["risk_level"] == risk_level_filter
        ]

    return values[:limit]