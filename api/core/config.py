"""Конфигурация приложения (этап 9)."""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Настройки из окружения; без секретов в коде."""

    def __init__(self) -> None:
        self.database_url: str = os.environ.get(
            "DATABASE_URL",
            "sqlite+pysqlite:///:memory:",
        )
        self.jwt_secret: str = os.environ.get("JWT_SECRET", "change-me-in-production-use-long-random")
        self.jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
        self.cors_allow_origins: list[str] = [
            o.strip()
            for o in (os.environ.get("CORS_ALLOW_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
