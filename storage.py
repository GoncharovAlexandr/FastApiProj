"""Асинхронное хранилище SQLite на aiosqlite."""
from datetime import datetime, timezone
import logging
import math

import aiosqlite

logger = logging.getLogger("agro_scoring.storage")


class StorageUnavailable(Exception):
    """База данных недоступна или вернула ошибку."""


class Database:
    def __init__(self, path: str):
        self.path = path

    async def initialize(self) -> None:
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        farm_id TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        recommendation TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        weekday TEXT NOT NULL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_request_id "
                    "ON predictions(request_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_predictions_weekday "
                    "ON predictions(weekday, id DESC)"
                )
                await db.commit()
        except aiosqlite.Error as exc:
            logger.exception("Failed to initialize SQLite")
            raise StorageUnavailable("Prediction storage is unavailable") from exc

    @staticmethod
    def _prepare_record(result: dict, source: str) -> tuple:
        now = datetime.now(timezone.utc)
        return (
            result["request_id"],
            result["farm_id"],
            result["risk_score"],
            result["risk_level"],
            result["recommendation"],
            result["model_version"],
            source,
            now.isoformat(timespec="milliseconds"),
            now.strftime("%A").lower(),
        )

    async def save_many(self, results: list[dict], source: str) -> None:
        if not results:
            return
        records = [self._prepare_record(item, source) for item in results]
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                await db.executemany(
                    """
                    INSERT INTO predictions
                    (request_id, farm_id, risk_score, risk_level, recommendation,
                     model_version, source, created_at, weekday)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                # Одна транзакция для всех результатов пакета.
                await db.commit()
        except aiosqlite.Error as exc:
            logger.exception("Failed to write predictions to SQLite")
            raise StorageUnavailable("Prediction storage is unavailable") from exc

    @staticmethod
    def _public_row(row) -> dict:
        return {
            "request_id": row["request_id"],
            "farm_id": row["farm_id"],
            "risk_score": row["risk_score"],
            "risk_level": row["risk_level"],
            "recommendation": row["recommendation"],
            "model_version": row["model_version"],
        }

    async def get_by_request_id(self, request_id: str) -> list[dict]:
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM predictions WHERE request_id = ? ORDER BY id ASC",
                    (request_id,),
                )
                rows = await cursor.fetchall()
                return [self._public_row(row) for row in rows]
        except aiosqlite.Error as exc:
            logger.exception("Failed to read prediction by request_id")
            raise StorageUnavailable("Prediction storage is unavailable") from exc

    async def list_predictions(
        self, limit: int = 10, offset: int = 0, risk_level: str | None = None
    ) -> list[dict]:
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                db.row_factory = aiosqlite.Row
                if risk_level is None:
                    cursor = await db.execute(
                        "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                else:
                    cursor = await db.execute(
                        "SELECT * FROM predictions WHERE risk_level = ? "
                        "ORDER BY id DESC LIMIT ? OFFSET ?",
                        (risk_level, limit, offset),
                    )
                rows = await cursor.fetchall()
                return [self._public_row(row) for row in rows]
        except aiosqlite.Error as exc:
            logger.exception("Failed to list predictions")
            raise StorageUnavailable("Prediction storage is unavailable") from exc

    async def list_by_weekday(
        self, weekday: str, page: int, per_page: int
    ) -> dict:
        offset = (page - 1) * per_page
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                db.row_factory = aiosqlite.Row
                count_cursor = await db.execute(
                    "SELECT COUNT(*) AS total FROM predictions WHERE weekday = ?",
                    (weekday,),
                )
                count_row = await count_cursor.fetchone()
                total_items = int(count_row["total"])
                cursor = await db.execute(
                    "SELECT * FROM predictions WHERE weekday = ? "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (weekday, per_page, offset),
                )
                rows = await cursor.fetchall()
                total_pages = math.ceil(total_items / per_page) if total_items else 0
                next_page = page + 1 if page < total_pages else None
                prev_page = page - 1 if 1 < page <= total_pages else None
                return {
                    "data": [self._public_row(row) for row in rows],
                    "meta": {
                        "weekday": weekday,
                        "current_page": page,
                        "per_page": per_page,
                        "total_pages": total_pages,
                        "total_items": total_items,
                        "next_page": next_page,
                        "prev_page": prev_page,
                    },
                }
        except aiosqlite.Error as exc:
            logger.exception("Failed to paginate predictions")
            raise StorageUnavailable("Prediction storage is unavailable") from exc
