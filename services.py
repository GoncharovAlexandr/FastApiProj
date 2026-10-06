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