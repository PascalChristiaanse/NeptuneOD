"""Structured configuration dataclasses for observation datasets."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationDatasetConfig:
    """Base configuration for all observation datasets.

    Attributes:
        type: Dataset modality (e.g., 'ground_ccd', 'space_ccd', 'gaia_ccd', etc.)
        file: Path to the dataset file.
        weight: Weight for this dataset in combined observations (1.0 = full weight).
        observatory_code: Observatory identifier code if applicable.
        name: Human-readable name for this dataset.
        metadata: Additional dataset-specific metadata.
    """

    type: str
    file: str
    weight: float = 1.0
    observatory_code: str | None = None
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0.0, 1.0], got {self.weight}")


@dataclass(frozen=True, kw_only=True)
class SimulatedObservationConfig(ObservationDatasetConfig):
    """Configuration for simulated observations.

    Represents synthetic observations used in testing or hybrid scenarios.
    """

    start_date_observation_period: str
    end_date_observation_period: str
    cadence: float
    observable_types: str
    noise_sigma: float

    def __post_init__(self):
        """Validate simulated config."""
        super().__post_init__()
