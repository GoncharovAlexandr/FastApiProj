from pydantic import BaseModel, Field


class FarmRequest(BaseModel):
    farm_id: str = Field(
        ...,
        description="Уникальный идентификатор хозяйства",
        examples=["FARM-001"],
    )

    region: str = Field(
        ...,
        description="Регион хозяйства",
        examples=["Krasnodar"],
    )

    crop_type: str = Field(
        ...,
        description="Тип сельскохозяйственной культуры",
        examples=["wheat"],
    )

    area_ha: float = Field(
        ...,
        gt=0,
        description="Площадь хозяйства в гектарах",
        examples=[500],
    )

    temperature_avg: float = Field(
        ...,
        description="Средняя температура",
        examples=[22],
    )

    precipitation_mm: float = Field(
        ...,
        ge=0,
        description="Количество осадков в миллиметрах",
        examples=[300],
    )

    payment_delay_days: int = Field(
        ...,
        ge=0,
        description="Количество дней задержки платежей",
        examples=[0],
    )

    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Количество предыдущих дефолтов",
        examples=[0],
    )

    debt: float = Field(
        ...,
        ge=0,
        description="Текущая задолженность",
        examples=[1000000],
    )


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str