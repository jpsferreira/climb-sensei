#!/usr/bin/env python
"""Migrate existing analyses to the route-centric model.

For each user:
1. Create an "Uncategorized" route
2. For each existing Analysis, create an Attempt linking:
   - Attempt.video_id = Analysis.video_id
   - Attempt.analysis_id = Analysis.id
   - Attempt.session_id = Analysis.session_id
   - Attempt.route_id = uncategorized_route.id
   - Attempt.date = Analysis.created_at
"""

from climb_sensei.database.config import SessionLocal
from climb_sensei.database.models import User, Analysis, Route, Attempt, Video


def migrate():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            existing = (
                db.query(Route)
                .filter(Route.user_id == user.id, Route.name == "Uncategorized")
                .first()
            )
            if existing:
                print(f"User {user.id}: already has Uncategorized route, skipping")
                continue

            analyses = (
                db.query(Analysis).join(Video).filter(Video.user_id == user.id).all()
            )

            if not analyses:
                print(f"User {user.id}: no analyses, skipping")
                continue

            route = Route(
                user_id=user.id,
                name="Uncategorized",
                grade="?",
                grade_system="hueco",
                type="boulder",
                status="projecting",
            )
            db.add(route)
            db.flush()

            for analysis in analyses:
                attempt = Attempt(
                    route_id=route.id,
                    video_id=analysis.video_id,
                    session_id=analysis.session_id,
                    analysis_id=analysis.id,
                    date=analysis.created_at,
                )
                db.add(attempt)

            db.commit()
            print(
                f"User {user.id}: migrated {len(analyses)} analyses "
                f"to Uncategorized route"
            )

    finally:
        db.close()


if __name__ == "__main__":
    migrate()
