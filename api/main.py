"""
FastAPI: PostgreSQL, JWT, RBAC (этап 9).

Запуск:
  docker compose up -d postgres
  set DATABASE_URL=postgresql+psycopg2://bullying_ai:bullying_ai@127.0.0.1:5432/bullying_ai
  alembic upgrade head
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analytics, auth, cameras, dashboard, incidents, reviews, reviews_queue, users, ws_overlay


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("AUTO_CREATE_TABLES", "").lower() in ("1", "true", "yes"):
        from api.db.session import init_db

        init_db()
    yield


_origins = (os.environ.get("CORS_ALLOW_ORIGINS") or "http://localhost:5173,http://127.0.0.1:5173").split(",")
_origins = [o.strip() for o in _origins if o.strip()]

app = FastAPI(
    title="Bullying detection platform",
    description="AI risk candidates; human review. Stage 10: RTSP live + MJPEG + WebSocket overlay.",
    version="0.10.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["cameras"])
app.include_router(incidents.router, prefix="/api/incidents", tags=["incidents"])
app.include_router(reviews.router, prefix="/api/incidents", tags=["reviews"])
app.include_router(reviews_queue.router, prefix="/api/reviews", tags=["reviews"])
app.include_router(ws_overlay.router, tags=["ws"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
