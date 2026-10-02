from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import create_engine
from backend.app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# Sync engine for scripts/migrations/GDAL hooks
sync_engine = create_engine(
    settings.database_url.replace("postgresql+psycopg://", "postgresql+psycopg://"),
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


def get_db():
    """Dependency helper for synchronous DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
