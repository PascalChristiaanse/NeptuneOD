"""Peewee data models for the data-acquisition database.

The schema mirrors ``data_module_database_design.md``:

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
:func:`init_db` once at start-up to connect and create the tables.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from peewee import (
    CharField,
    DateTimeField,
    DoubleField,
    ForeignKeyField,
    Model,
    SqliteDatabase,
    TextField,
)

__all__ = [
    "db",
    "BaseModel",
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
    "bind",
    "STANDARD_DATAFRAME_COLUMNS",
    "StandardDataFrameVerifier",
]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

#: Fully in-memory SQLite engine. Each connection gets its own private database;
#: use :func:`bind` with a shared-cache URI if you need cross-thread access.
db = SqliteDatabase(":memory:")


class BaseModel(Model):
    """Base model bound to the in-memory SQLite database."""

    class Meta:
        database = db


# ---------------------------------------------------------------------------
# Reference / metadata tables
# ---------------------------------------------------------------------------


class DataSource(BaseModel):
    """A provider of observations (e.g. NSDB, Gaia, JPL)."""

    name = CharField(unique=True)
    description = TextField(null=True)
    url = TextField(null=True)

    class Meta:
        table_name = "data_source"


class SourceCollection(BaseModel):
    """A collection published by a :class:`DataSource`."""

    source = ForeignKeyField(DataSource, backref="collections", column_name="source_id")
    external_id = CharField(null=True)
    title = TextField(null=True)
    url = TextField(null=True)

    class Meta:
        table_name = "source_collection"


class AstroObject(BaseModel):
    """The observed target (named ``AstroObject`` to avoid shadowing ``object``)."""

    designation = CharField(unique=True)
    object_type = CharField(null=True)

    class Meta:
        table_name = "object"


class Observatory(BaseModel):
    """An observing station identified by its MPC observatory code."""

    observatory_code = CharField(unique=True)
    name = CharField(null=True)

    class Meta:
        table_name = "observatory"


class Reference(BaseModel):
    """A bibliographic reference documenting a dataset."""

    citation = TextField(null=True)
    doi = CharField(null=True)

    class Meta:
        table_name = "reference"


class Dataset(BaseModel):
    """A homogeneous set of observations sharing object/observatory/reference."""

    collection = ForeignKeyField(
        SourceCollection, backref="datasets", column_name="collection_id"
    )
    object = ForeignKeyField(AstroObject, backref="datasets", column_name="object_id")
    observatory = ForeignKeyField(
        Observatory, backref="datasets", column_name="observatory_id"
    )
    reference = ForeignKeyField(Reference, backref="datasets", column_name="reference_id")
    coordinate_system = CharField(null=True)
    observation_type = CharField(null=True)
    frame_center = CharField(null=True)
    original_equinox = CharField(null=True)
    original_reduction = CharField(null=True)
    receptor_type = CharField(null=True)
    time_scale = CharField(null=True)

    class Meta:
        table_name = "dataset"


class Observation(BaseModel):
    """A single observation epoch belonging to a :class:`Dataset`."""

    dataset = ForeignKeyField(Dataset, backref="observations", column_name="dataset_id")
    observation_time = DateTimeField()

    class Meta:
        table_name = "observation"

    # Peewee backrefs always return a ``ModelSelect``; these properties expose the
    # one-to-one measurements as a single instance (or ``None``).
    @property
    def equatorial_measurement(self) -> MeasurementEquatorial | None:
        return self.equatorial.get_or_none()

    @property
    def cartesian_measurement(self) -> MeasurementCartesian | None:
        return self.cartesian.get_or_none()

    @property
    def scan_measurement(self) -> MeasurementScan | None:
        return self.scan.get_or_none()


# ---------------------------------------------------------------------------
# Measurements (one-to-one with Observation, except DerivedMeasurement)
# ---------------------------------------------------------------------------


class MeasurementEquatorial(BaseModel):
    """Equatorial (RA/Dec) measurement for an observation."""

    observation = ForeignKeyField(
        Observation, backref="equatorial", unique=True, column_name="observation_id"
    )
    ra = DoubleField()
    dec = DoubleField()
    original_unit = CharField(null=True)

    class Meta:
        table_name = "measurement_equatorial"


class MeasurementCartesian(BaseModel):
    """Cartesian (x/y) measurement for an observation."""

    observation = ForeignKeyField(
        Observation, backref="cartesian", unique=True, column_name="observation_id"
    )
    x = DoubleField()
    y = DoubleField()
    original_unit = CharField(null=True)

    class Meta:
        table_name = "measurement_cartesian"


class MeasurementScan(BaseModel):
    """Scan-based measurement for an observation."""

    observation = ForeignKeyField(
        Observation, backref="scan", unique=True, column_name="observation_id"
    )
    along_scan = DoubleField()
    across_scan = DoubleField()
    scan_angle = DoubleField(null=False)
    original_unit = CharField(null=True)

    class Meta:
        table_name = "measurement_scan"


class DerivedMeasurement(BaseModel):
    """A derived measurement (e.g. a converted coordinate pair) for an observation."""

    observation = ForeignKeyField(
        Observation, backref="derived_measurements", column_name="observation_id"
    )
    coordinate_system = CharField()
    coordinate_1 = DoubleField()
    coordinate_2 = DoubleField()
    method = CharField(null=True)

    class Meta:
        table_name = "derived_measurement"


#: All models in dependency order (parents before children).
ALL_MODELS: tuple[type[BaseModel], ...] = (
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


def init_db(*, drop: bool = False) -> SqliteDatabase:
    """Connect to the in-memory database and create all tables.

    Args:
        drop: When ``True``, drop any existing tables first.

    Returns:
        The bound :class:`SqliteDatabase` instance.
    """
    db.connect(reuse_if_open=True)
    if drop:
        db.drop_tables(ALL_MODELS, safe=True)
    db.create_tables(ALL_MODELS, safe=True)
    return db


def close_db() -> None:
    """Close the database connection if it is open."""
    if not db.is_closed():
        db.close()


@contextmanager
def bind(database: SqliteDatabase) -> Iterator[SqliteDatabase]:
    """Temporarily bind all models to ``database`` (useful for tests).

    Example:
        >>> with bind(SqliteDatabase(":memory:")) as test_db:
        ...     test_db.create_tables(ALL_MODELS)
    """
    with database.bind_ctx(ALL_MODELS):
        yield database


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
