predictions: dict[str, dict] = {}


def save_prediction(request_id: str, result: dict) -> None:
    predictions[request_id] = result


def get_prediction(request_id: str) -> dict | None:
    return predictions.get(request_id)


def get_predictions() -> list[dict]:
    return list(predictions.values())