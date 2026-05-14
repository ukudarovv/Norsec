from __future__ import annotations

from fastapi.testclient import TestClient


def test_register_first_admin(client: TestClient) -> None:
    r = client.post(
        "/api/auth/register",
        json={"email": "first@example.com", "password": "secret12345", "full_name": "A"},
    )
    assert r.status_code == 201
    assert r.json()["role"] == "admin"
    assert r.json()["email"] == "first@example.com"


def test_login_and_me(client: TestClient, admin_headers: dict[str, str]) -> None:
    me = client.get("/api/auth/me", headers=admin_headers)
    assert me.status_code == 200
    assert me.json()["email"] == "admin@example.com"


def test_register_closed_after_first(client: TestClient, admin_headers: dict[str, str]) -> None:
    r = client.post(
        "/api/auth/register",
        json={"email": "x@example.com", "password": "secret12345"},
    )
    assert r.status_code == 403
