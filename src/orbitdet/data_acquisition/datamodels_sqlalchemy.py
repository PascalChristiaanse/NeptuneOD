"""SQLAlchemy 2.0 data models for the data-acquisition database.

This is a side-by-side alternative to ``datamodels.py`` (Peewee) so the two can be
compared directly. The schema is identical:

    DataSource ||--o{ SourceCollection : contains
    SourceCollection ||--o{ Dataset : partitioned_into
    AstroObject ||--o{ Dataset : identifies_target
    Observatory ||--o{ Dataset : observed_at
    Reference ||--o{ Dataset : documented_by
    Dataset ||--o{ Observation : contains
    Observation ||--o| MeasurementEquatorial : has
    Observation ||--o| MeasurementCartesian : has
    Observation ||--o| MeasurementScan : has
    Observation ||--o{ DerivedMeasurement : may_have

The models are bound to a fully in-memory SQLite database by default. Call
:func:`init_db` once at start-up to create the tables, and use :func:`session`
to obtain a unit-of-work session.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "DataSource",
    "SourceCollection",
    "AstroObject",
    "Observatory",
    "Reference",
    "Dataset",
    "Observation",
    "MeasurementEquatorial",
    "MeasurementCartesian",
    "MeasurementScan",
    "DerivedMeasurement",
    "ALL_MODELS",
    "init_db",
    "close_db",
    "session",
    "bind",
    "STANDARD_DATAFRAME_COLUMNS",
    "StandardDataFrameVerifier",
]


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

#: Fully in-memory SQLite engine. ``StaticPool`` keeps a single shared connection
#: so every session sees the same database (a plain pool would give each
#: connection its own empty ``:memory:`` database).
engine = create_engine(
    "sqlite:///:memory:",
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for all models."""


# ---------------------------------------------------------------------------
# Reference / metadata tables
# ---------------------------------------------------------------------------


class DataSource(Base):
    """A provider of observations (e.g. NSDB, Gaia, JPL)."""

    __tablename__ = "data_source"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)

    collections: Mapped[list[SourceCollection]] = relationship(back_populates="source")


class SourceCollection(Base):
    """A collection published by a :class:`DataSource`."""

    __tablename__ = "source_collection"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_source.id"))
    external_id: Mapped[str | None] = mapped_column(String)
    title: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)

    source: Mapped[DataSource] = relationship(back_populates="collections")
    datasets: Mapped[list[Dataset]] = relationship(back_populates="collection")


class AstroObject(Base):
    """The observed target (named ``AstroObject`` to avoid shadowing ``object``)."""

    __tablename__ = "object"

    id: Mapped[int] = mapped_column(primary_key=True)
    designation: Mapped[str] = mapped_column(String, unique=True)
    object_type: Mapped[str | None] = mapped_column(String)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="object")


class Observatory(Base):
    """An observing station identified by its MPC observatory code."""

    __tablename__ = "observatory"

    id: Mapped[int] = mapped_column(primary_key=True)
    observatory_code: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str | None] = mapped_column(String)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="observatory")


class Reference(Base):
    """A bibliographic reference documenting a dataset."""

    __tablename__ = "reference"

    id: Mapped[int] = mapped_column(primary_key=True)
    citation: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="reference")


class Dataset(Base):
    """A homogeneous set of observations sharing object/observatory/reference."""

    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[int] = mapped_column(ForeignKey("source_collection.id"))
    object_id: Mapped[int] = mapped_column(ForeignKey("object.id"))
    observatory_id: Mapped[int] = mapped_column(ForeignKey("observatory.id"))
    reference_id: Mapped[int] = mapped_column(ForeignKey("reference.id"))
    coordinate_system: Mapped[str | None] = mapped_column(String)
    observation_type: Mapped[str | None] = mapped_column(String)
    frame_center: Mapped[str | None] = mapped_column(String)
    original_equinox: Mapped[str | None] = mapped_column(String)
    original_reduction: Mapped[str | None] = mapped_column(String)
    receptor_type: Mapped[str | None] = mapped_column(String)
    time_scale: Mapped[str | None] = mapped_column(String)

    collection: Mapped[SourceCollection] = relationship(back_populates="datasets")
    object: Mapped[AstroObject] = relationship(back_populates="datasets")
    observatory: Mapped[Observatory] = relationship(back_populates="datasets")
    reference: Mapped[Reference] = relationship(back_populates="datasets")
    observations: Mapped[list[Observation]] = relationship(back_populates="dataset")


class Observation(Base):
    """A single observation epoch belonging to a :class:`Dataset`."""

    __tablename__ = "observation"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    observation_time: Mapped[datetime] = mapped_column(DateTime)

    dataset: Mapped[Dataset] = relationship(back_populates="observations")

    # ``uselist=False`` makes these scalar (one-to-one) instead of a collection.
    equatorial: Mapped[MeasurementEquatorial | None] = relationship(
        back_populates="observation", uselist=False
    )
    cartesian: Mapped[MeasurementCartesian | None] = relationship(
        back_populates="observation", uselist=False
    )
    scan: Mapped[MeasurementScan | None] = relationship(
        back_populates="observation", uselist=False
    )
    derived_measurements: Mapped[list[DerivedMeasurement]] = relationship(
        back_populates="observation"
    )


# ---------------------------------------------------------------------------
# Measurements (one-to-one with Observation, except DerivedMeasurement)
# ---------------------------------------------------------------------------


class MeasurementEquatorial(Base):
    """Equatorial (RA/Dec) measurement for an observation."""

    __tablename__ = "measurement_equatorial"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observation.id"), unique=True
    )
    ra: Mapped[float] = mapped_column(Float)
    dec: Mapped[float] = mapped_column(Float)
    original_unit: Mapped[str | None] = mapped_column(String)

    observation: Mapped[Observation] = relationship(back_populates="equatorial")


class MeasurementCartesian(Base):
    """Cartesian (x/y) measurement for an observation."""

    __tablename__ = "measurement_cartesian"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observation.id"), unique=True
    )
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    original_unit: Mapped[str | None] = mapped_column(String)

    observation: Mapped[Observation] = relationship(back_populates="cartesian")


class MeasurementScan(Base):
    """Scan-based measurement for an observation."""

    __tablename__ = "measurement_scan"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observation.id"), unique=True
    )
    along_scan: Mapped[float] = mapped_column(Float)
    across_scan: Mapped[float] = mapped_column(Float)
    scan_angle: Mapped[float] = mapped_column(Float)
    original_unit: Mapped[str | None] = mapped_column(String)

    observation: Mapped[Observation] = relationship(back_populates="scan")


class DerivedMeasurement(Base):
    """A derived measurement (e.g. a converted coordinate pair) for an observation."""

    __tablename__ = "derived_measurement"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey("observation.id"))
    coordinate_system: Mapped[str] = mapped_column(String)
    coordinate_1: Mapped[float] = mapped_column(Float)
    coordinate_2: Mapped[float] = mapped_column(Float)
    method: Mapped[str | None] = mapped_column(String)

    observation: Mapped[Observation] = relationship(
        back_populates="derived_measurements"
    )


#: All models in dependency order (parents before children).
ALL_MODELS: tuple[type[Base], ...] = (
    DataSource,
    SourceCollection,
    AstroObject,
    Observatory,
    Reference,
    Dataset,
    Observation,
    MeasurementEquatorial,
    MeasurementCartesian,
    MeasurementScan,
    DerivedMeasurement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_db(*, drop: bool = False) -> None:
    """Create all tables in the in-memory database.

    Args:
        drop: When ``True``, drop any existing tables first.
    """
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def close_db() -> None:
    """Dispose of the engine's connection pool."""
    engine.dispose()


@contextmanager
def session() -> Iterator[Session]:
    """Provide a transactional session scope.

    Commits on success, rolls back on exception, and always closes.

    Example:
        >>> with session() as s:
        ...     s.add(DataSource(name="NSDB"))
    """
    sess = SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


@contextmanager
def bind(other_engine) -> Iterator[Session]:
    """Temporarily bind a session to ``other_engine`` (useful for tests).

    Example:
        >>> from sqlalchemy import create_engine
        >>> test_engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
        >>> with bind(test_engine) as s:
        ...     Base.metadata.create_all(test_engine)
    """
    maker = sessionmaker(bind=other_engine, expire_on_commit=False)
    sess = maker()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# Dataframe verification helpers
# ---------------------------------------------------------------------------


@dataclass
class STANDARD_DATAFRAME_COLUMNS:
    """Defines the required columns and their types for a standard dataframe."""

    observed_object: str = field()
    observatory_code: str = field()
    data_source: str = field()
    reference: str = field()
    type: str = field()
    coordinates: str = field()
    center_of_frame: str = field()
    original_epoch_of_equinox: str = field()
    original_reduction: str = field()
    receptor_type: str = field()
    observation_epoch: str = field()
    first_observation_value: str = field()
    second_observation_value: str = field()


class StandardDataFrameVerifier:
    """Class to check dataframes for required columns and types."""

    def __init__(self, required_columns: dict[str, type]):
        self.required_columns = required_columns
