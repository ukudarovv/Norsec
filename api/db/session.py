"""Сессия БД (lazy engine для тестов с подменой env)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.core.config import get_settings
from api.db.base import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    try:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass


def get_engine() -> Engine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            if ":memory:" in url:
                _engine = create_engine(
                    url,
                    connect_args=connect_args,
                    poolclass=StaticPool,
                )
            else:
                _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        else:
            _engine = create_engine(url, pool_pre_ping=True)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _ensure_sqlite_foreign_keys)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal  # noqa: PLW0603
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def reset_engine() -> None:
    """Сброс кэша движка (тесты)."""
    global _engine, _SessionLocal  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_db() -> Generator[Session, None, None]:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
