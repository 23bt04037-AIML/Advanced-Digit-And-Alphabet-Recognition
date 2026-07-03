"""Database engine, session factory, and FastAPI dependency."""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost:3306/digit_recognition_db?charset=utf8mb4",
)

engine_args = {
    "pool_pre_ping": True,
    "echo": False,
    "pool_size": 10,
    "max_overflow": 20
}

engine = create_engine(
    DATABASE_URL,
    **engine_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    from backend.database.models import (  # noqa – import triggers table registration
        User, OTPRecord, PredictionLog, AuditLog, RuntimeLog, ModelMetrics,
        UniversalPredictionLog,
    )
    Base.metadata.create_all(bind=engine)
    logger.info("All database tables created / verified.")



def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK.")
        return True
    except Exception as exc:
        logger.error(f"Database connection FAILED: {exc}")
        return False
