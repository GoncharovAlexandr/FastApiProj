from schemas import FarmRequest

ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}


def risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def recommendation(level: str) -> str:
    if level == "low":
        return "Стандартное рассмотрение"
    if level == "medium":
        return "Требуется дополнительная проверка"
    return "Высокий риск. Требуется ручное рассмотрение"


def build_prediction(request_id: str, request: FarmRequest, score: float) -> dict:
    level = risk_level(score)
    return {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": round(float(score), 4),
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": "1.0",
    }
