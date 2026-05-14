"""Общая конфигурация тестов API (SQLite in-memory, изоляция на тест)."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-value-min-32-chars-long-abc")

from api.core import config

config.get_settings.cache_clear()

from api.db.session import init_db, reset_engine  # noqa: E402

reset_engine()
init_db()

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def _reset():
    try:
        from camera.camera_manager import reset_camera_manager

        reset_camera_manager()
    except Exception:
        pass
    try:
        from configs.load import clear_phase1_config_cache

        clear_phase1_config_cache()
    except Exception:
        pass
    config.get_settings.cache_clear()
    reset_engine()
    init_db()


import pytest


@pytest.fixture(autouse=True)
def _per_test_db() -> None:
    _reset()
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def db_session():
    from api.db.session import get_session_factory

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def admin_headers(client: TestClient) -> dict[str, str]:
    r = client.post(
        "/api/auth/register",
        json={"email": "admin@example.com", "password": "secret12345", "full_name": "Admin"},
    )
    assert r.status_code == 201, r.text
    t = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "secret12345"})
    assert t.status_code == 200, t.text
    tok = t.json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture()
def viewer_headers(client: TestClient, admin_headers: dict[str, str]) -> dict[str, str]:
    r = client.post(
        "/api/users",
        headers=admin_headers,
        json={"email": "viewer@example.com", "password": "secret12345", "role": "viewer"},
    )
    assert r.status_code == 201, r.text
    t = client.post("/api/auth/login", json={"email": "viewer@example.com", "password": "secret12345"})
    assert t.status_code == 200
    tok = t.json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture()
def reviewer_headers(client: TestClient, admin_headers: dict[str, str]) -> dict[str, str]:
    r = client.post(
        "/api/users",
        headers=admin_headers,
        json={"email": "rev@example.com", "password": "secret12345", "role": "reviewer"},
    )
    assert r.status_code == 201, r.text
    t = client.post("/api/auth/login", json={"email": "rev@example.com", "password": "secret12345"})
    assert t.status_code == 200
    tok = t.json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture()
def operator_headers(client: TestClient, admin_headers: dict[str, str]) -> dict[str, str]:
    r = client.post(
        "/api/users",
        headers=admin_headers,
        json={"email": "operator@example.com", "password": "secret12345", "role": "operator"},
    )
    assert r.status_code == 201, r.text
    t = client.post("/api/auth/login", json={"email": "operator@example.com", "password": "secret12345"})
    assert t.status_code == 200
    tok = t.json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}
