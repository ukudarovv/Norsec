"""Бизнес-логика review workflow (Phase 3)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from api.db.models import Incident, Review, User
from api.schemas.review import REVIEW_STATUSES, ReviewRequest

STATUSES_REQUIRING_COMMENT = frozenset({"confirmed", "false_positive", "training_candidate"})


def validate_review_request(body: ReviewRequest) -> None:
    if body.status not in REVIEW_STATUSES:
        raise ValueError("invalid status")
    if body.status in STATUSES_REQUIRING_COMMENT:
        if not (body.comment and str(body.comment).strip()):
            raise ValueError("comment is required for this status")


def apply_review(db: Session, incident_id: uuid.UUID, reviewer: User, body: ReviewRequest) -> Incident | None:
    validate_review_request(body)
    inc = db.get(Incident, incident_id)
    if inc is None:
        return None
    tags = [str(t).strip() for t in (body.tags or []) if str(t).strip()]
    rev = Review(
        incident_id=inc.id,
        reviewer_id=reviewer.id,
        status=body.status,
        comment=body.comment,
        tags=tags,
    )
    inc.review_status = body.status
    db.add(rev)
    db.commit()
    db.refresh(inc)
    if body.status == "training_candidate":
        from mlops.training_candidates import record_candidate

        ev = dict(inc.evidence or {})
        try:
            record_candidate(
                {
                    "incident_id": str(inc.id),
                    "status": body.status,
                    "labels": [],
                    "tags": tags,
                    "clip_path": inc.clip_path or "",
                    "snapshot_path": str(ev.get("snapshot_path") or ""),
                }
            )
        except OSError:
            pass
    return db.scalar(select(Incident).options(joinedload(Incident.camera)).where(Incident.id == inc.id))


def list_review_queue(db: Session, *, limit: int = 100) -> list[Incident]:
    q = (
        select(Incident)
        .options(joinedload(Incident.camera))
        .where(Incident.review_status.in_(("new", "needs_review")))
        .order_by(Incident.created_at.asc())
        .limit(max(1, min(limit, 500)))
    )
    return list(db.scalars(q).all())


def latest_reviewer_emails(db: Session, incident_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
    if not incident_ids:
        return {}
    revs = list(
        db.scalars(
            select(Review).where(Review.incident_id.in_(incident_ids)).order_by(Review.created_at.desc())
        ).all()
    )
    last_uid: dict[uuid.UUID, uuid.UUID] = {}
    for r in revs:
        if r.incident_id not in last_uid:
            last_uid[r.incident_id] = r.reviewer_id
    uids = list(set(last_uid.values()))
    if not uids:
        return {}
    users = {u.id: u.email for u in db.scalars(select(User).where(User.id.in_(uids))).all()}
    return {iid: users.get(uid, "") or "" for iid, uid in last_uid.items()}
