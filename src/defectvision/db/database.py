"""DefectVision Database Package.

Provides SQLAlchemy declarative base, session management, and engine configuration.
"""

from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Default to PostgreSQL, with environment override
DEFAULT_DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/defectvision"


def get_database_url() -> str:
    """Retrieve database URL from environment or default to local PostgreSQL."""
    return os.getenv("DEFECTVISION_DATABASE_URL", os.getenv("DATABASE_URL", DEFAULT_DB_URL))


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in DefectVision."""
    pass


def get_engine(db_url: str | None = None, **engine_kwargs: Any):
    """Create a SQLAlchemy engine configured for PostgreSQL or SQLite."""
    url = db_url or get_database_url()
    kwargs: dict[str, Any] = dict(engine_kwargs)

    if url.startswith("sqlite"):
        kwargs.setdefault("connect_args", {"check_same_thread": False})
    else:
        # Standard PostgreSQL pool settings
        kwargs.setdefault("pool_pre_ping", True)
        kwargs.setdefault("pool_size", 5)
        kwargs.setdefault("max_overflow", 10)

    return create_engine(url, **kwargs)


# Application-level engine and session factory
_default_engine = None
_DefaultSessionLocal = None


def get_session_factory(engine=None) -> sessionmaker[Session]:
    """Return a session factory bound to the specified or default engine."""
    global _default_engine, _DefaultSessionLocal
    if engine is not None:
        return sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if _DefaultSessionLocal is None:
        if _default_engine is None:
            _default_engine = get_engine()
        _DefaultSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=_default_engine
        )
    return _DefaultSessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db(engine=None) -> None:
    """Create all database tables defined on Base.metadata."""
    eng = engine or _default_engine or get_engine()
    Base.metadata.create_all(bind=eng)
