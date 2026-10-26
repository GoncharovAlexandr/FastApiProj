"""Заглушка модели агроскоринга и имитация асинхронного инференса."""
import asyncio

from config import MODEL_SIMULATED_LATENCY_SECONDS
from schemas import FarmRequest


def calculate_risk(data: FarmRequest) -> float:
    """Детерминированный скоринг из практической работы №1."""
    score = 0.1
    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return min(score, 1.0)


async def model_predict_batch(items: list[FarmRequest]) -> list[float]:
    """Один асинхронный вызов модели для всего пакета.

    Задержка имитирует внешний/дорогой инференс, чтобы можно было проверить
    batching и timeout без настоящей ML-модели.
    """
    await asyncio.sleep(MODEL_SIMULATED_LATENCY_SECONDS)
    return [calculate_risk(item) for item in items]
