from typing import Any

from pydantic import BaseModel, Field

from config import MAX_BATCH_SIZE, LLM_MODEL


class FarmRequest(BaseModel):
    farm_id: str = Field(..., description="Идентификатор хозяйства", examples=["FARM-001"])
    region: str = Field(..., description="Регион хозяйства", examples=["Krasnodar"])
    crop_type: str = Field(..., description="Тип культуры", examples=["wheat"])
    area_ha: float = Field(..., gt=0, description="Площадь, га", examples=[500])
    temperature_avg: float = Field(..., description="Средняя температура", examples=[22])
    precipitation_mm: float = Field(
        ..., ge=0, description="Осадки, мм", examples=[300]
    )
    payment_delay_days: int = Field(
        ..., ge=0, description="Задержка платежей, дни", examples=[0]
    )
    previous_defaults: int = Field(
        ..., ge=0, description="Количество предыдущих дефолтов", examples=[0]
    )
    debt: float = Field(..., ge=0, description="Текущий долг", examples=[1_000_000])


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"От 1 до {MAX_BATCH_SIZE} хозяйств",
    )


class BatchPredictionResponse(BaseModel):
    request_id: str
    items: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class WeekdayMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: int | None
    prev_page: int | None


class WeekdayPageResponse(BaseModel):
    data: list[PredictionResponse]
    meta: WeekdayMeta


class LLMChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Текст запроса к LLM")
    model: str = Field(default=LLM_MODEL, description="Модель, по умолчанию auto")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, ge=1, le=4000)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str
    answer: str
    finish_reason: str | None
    latency_ms: float


class PredictWithLLMResponse(PredictionResponse):
    explanation: str
    requested_model: str
    returned_model: str
    latency_ms: float


class ErrorResponse(BaseModel):
    detail: Any
    request_id: str
