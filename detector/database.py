import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
default_database = f"sqlite:///{PROJECT_ROOT / 'data' / 'feedback.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", default_database)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feedback: Mapped[str] = mapped_column(String(20), index=True)
    label: Mapped[str] = mapped_column(String(50))
    prediction: Mapped[str | None] = mapped_column(String(50), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    image: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    verdict: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    fake_probability: Mapped[float] = mapped_column(Float)
    storage_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


def init_database():
    if DATABASE_URL.startswith("sqlite"):
        (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    Base.metadata.create_all(engine)


def save_feedback(data: dict):
    created_at = data.get("createdAt")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    with SessionLocal() as session:
        record = Feedback(
            feedback=data["feedback"], label=data["label"], prediction=data.get("prediction"),
            confidence=data.get("confidence"), image=data.get("image"), page=data.get("page"),
            created_at=created_at or datetime.now(timezone.utc),
        )
        session.add(record)
        session.commit()


def list_feedback():
    with SessionLocal() as session:
        return list(session.scalars(select(Feedback).order_by(Feedback.id)))


def save_prediction(data: dict):
    with SessionLocal() as session:
        session.add(Prediction(verdict=data["verdict"], confidence=data["confidence"], fake_probability=data["fake_probability"], storage_uri=data.get("storage_uri")))
        session.commit()
