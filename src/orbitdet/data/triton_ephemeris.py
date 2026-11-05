"""Loader for the TritonEph2020 tabulated Triton ephemeris.

The ``TritonEph2020`` file shipped under ``data/kernels`` is an ASCII table with
one row per hour.  Each row has the form::

    Date(UTC),X(KM),Y(KM),Z(KM),Vx(KM/S),Vy(KM/S),Vz(KM/S)

The positions and velocities are the Cartesian state of Triton relative to the
Neptune system barycenter, expressed in the (barycentric) J2000 equatorial frame,
with the epoch column in UTC (the file header is literally ``Date(UTC)``).

This module converts that table into the ``dict[float, numpy.ndarray]`` state
history that :func:`tudatpy.dynamics.environment_setup.ephemeris.tabulated`
expects (Tudat units: metres, metres per second, seconds since J2000 TDB).
"""

import logging
from pathlib import Path

import numpy as np
from tudatpy.interface import spice

logger = logging.getLogger(__name__)

#: Frame in which the TritonEph2020 data is expressed.  The file is barycentric
#: equatorial, which is Tudat's ``J2000`` frame, so no rotation is required.
TRITON_EPH_2020_FRAME = "J2000"

#: Time system of the epoch column in TritonEph2020.
TRITON_EPH_2020_TIME_SYSTEM = "UTC"


def build_triton_eph_2020_state_history(
    file_path: str | Path,
    frame_orientation: str = TRITON_EPH_2020_FRAME,
    decimation: int = 1,
) -> dict[float, np.ndarray]:
    """Build a Tudat tabulated-ephemeris state history from ``TritonEph2020``.

    Parameters
    ----------
    file_path:
        Path to the ``TritonEph2020`` ASCII file.
    frame_orientation:
        Orientation of the frame the data is expressed in.  The file is defined
        in the barycentric J2000 frame, so only ``"J2000"`` is accepted (no frame
        rotation is applied).
    decimation:
        Keep only every ``decimation``-th sample.  ``1`` (default) preserves the
        native 1-hour cadence.  Larger values trade interpolation accuracy for
        memory and build time.

    Returns
    -------
    dict[float, np.ndarray]
        Mapping from epoch (seconds since J2000 TDB) to the Cartesian state of
        Triton relative to the Neptune system barycenter, as a ``(6, 1)`` array
        in metres and metres per second.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If ``frame_orientation`` is not ``"J2000"`` or ``decimation`` is below 1.
    """
    if frame_orientation != TRITON_EPH_2020_FRAME:
        raise ValueError(
            f"TritonEph2020 is defined in the '{TRITON_EPH_2020_FRAME}' frame, "
            f"but '{frame_orientation}' was requested; no rotation is applied."
        )
    if decimation < 1:
        raise ValueError(f"decimation must be at least 1, got {decimation}.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TritonEph2020 file not found: {path}")

    logger.info("Reading TritonEph2020 tabulated ephemeris from %s ...", path)
    raw_dates: list[str] = []
    raw_states: list[tuple[float, float, float, float, float, float]] = []
    kept = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            # Skip blank lines and the ASCII header/separator rows.
            if not line or not line[0].isdigit():
                continue
            fields = line.split(",")
            if len(fields) != 7:
                continue
            # Subsample on the fly: the SPICE string-to-epoch conversion and the
            # intermediate Python objects dominate the cost, so dropped rows
            # should not be retained at all.
            if kept % decimation != 0:
                kept += 1
                continue
            kept += 1
            raw_dates.append(fields[0])
            raw_states.append(tuple(float(value) for value in fields[1:7]))

    if not raw_dates:
        raise ValueError(f"No valid rows parsed from TritonEph2020 file: {path}")

    time_system_suffix = f" {TRITON_EPH_2020_TIME_SYSTEM}"
    epochs = [
        spice.convert_date_string_to_ephemeris_time(date + time_system_suffix)
        for date in raw_dates
    ]

    state_array = np.asarray(raw_states, dtype=float)
    state_array[:, :3] *= 1000.0  # km -> m
    state_array[:, 3:] *= 1000.0  # km/s -> m/s

    state_history: dict[float, np.ndarray] = {
        float(epoch): state.reshape(6, 1) for epoch, state in zip(epochs, state_array)
    }

    logger.info(
        "Built TritonEph2020 tabulated ephemeris with %d samples "
        "(decimation=%d) covering [%.1f, %.1f] s since J2000 TDB.",
        len(state_history),
        decimation,
        min(state_history),
        max(state_history),
    )
    return state_history
