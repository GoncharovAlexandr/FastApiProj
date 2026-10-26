"""Настройки сервиса. Значения можно переопределить переменными окружения."""
import os

DB_PATH = os.getenv("AGRO_SCORING_DB", "agro_scoring.db")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://vireonix.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "auto")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "2.0"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_BATCH_WAIT_MS = int(os.getenv("MAX_BATCH_WAIT_MS", "50"))
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "100"))
# Реалистичная задержка-заглушка для демонстрации timeout: вместо неё в реальной
# системе вызывается модель инференса.
MODEL_SIMULATED_LATENCY_SECONDS = float(
    os.getenv("MODEL_SIMULATED_LATENCY_SECONDS", "0.02")
)
