"""Tests for the TritonEph2020 tabulated ephemeris loader."""

from pathlib import Path

import numpy as np
import pytest
from tudatpy.interface import spice

from orbitdet.data.triton_ephemeris import (
    TRITON_EPH_2020_FRAME,
    build_triton_eph_2020_state_history,
)

NAIF_LSK = Path("data/kernels/naif0012.tls")


@pytest.fixture(scope="module", autouse=True)
def _load_leap_second_kernel():
    """Ensure the leap-second kernel is loaded so UTC strings can be parsed."""
    if NAIF_LSK.exists():
        spice.load_kernel(str(NAIF_LSK))


def _write_sample(path: Path, rows: list[str]) -> Path:
    content = (
        "\n----Date(UTC)------ -----X(KM)------- ------Y(KM)-------\n"
        "---------------------------------------------------------------------\n"
    )
    content += "\n".join(rows) + "\n"
    path.write_text(content)
    return path


class TestBuildTritonEph2020StateHistory:
    def test_parses_rows_and_converts_units(self, tmp_path):
        sample = _write_sample(
            tmp_path / "TritonEph2020",
            [
                "2000-01-01 00:00:00,1.0,2.0,3.0,0.001,0.002,0.003",
                "2000-01-01 01:00:00,4.0,5.0,6.0,0.004,0.005,0.006",
            ],
        )
        history = build_triton_eph_2020_state_history(sample)

        assert len(history) == 2
        epochs = sorted(history)
        # UTC 2000-01-01 00:00:00 is -43135.816 s from J2000 TDB (J2000 is noon).
        assert epochs[0] == pytest.approx(-43135.816, abs=1e-3)
        # Each row's date is parsed independently, so allow sub-ms jitter.
        assert epochs[1] - epochs[0] == pytest.approx(3600.0, abs=1e-3)

        first = history[epochs[0]]
        assert first.shape == (6, 1)
        # Positions km -> m, velocities km/s -> m/s.
        np.testing.assert_allclose(first[:3, 0], [1000.0, 2000.0, 3000.0])
        np.testing.assert_allclose(first[3:, 0], [1.0, 2.0, 3.0])

    def test_decimation_subsamples(self, tmp_path):
        rows = [f"2000-01-01 0{h}:00:00,{h}.0,0.0,0.0,0.0,0.0,0.0" for h in range(6)]
        sample = _write_sample(tmp_path / "TritonEph2020", rows)

        full = build_triton_eph_2020_state_history(sample, decimation=1)
        coarse = build_triton_eph_2020_state_history(sample, decimation=2)

        assert len(full) == 6
        assert len(coarse) == 3
        assert sorted(coarse) == sorted(full)[::2]

    def test_rejects_non_j2000_frame(self, tmp_path):
        sample = _write_sample(
            tmp_path / "TritonEph2020",
            ["2000-01-01 00:00:00,1.0,2.0,3.0,0.0,0.0,0.0"],
        )
        with pytest.raises(ValueError, match="J2000"):
            build_triton_eph_2020_state_history(sample, frame_orientation="ECLIPJ2000")

    def test_rejects_invalid_decimation(self, tmp_path):
        sample = _write_sample(
            tmp_path / "TritonEph2020",
            ["2000-01-01 00:00:00,1.0,2.0,3.0,0.0,0.0,0.0"],
        )
        with pytest.raises(ValueError, match="decimation"):
            build_triton_eph_2020_state_history(sample, decimation=0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_triton_eph_2020_state_history(tmp_path / "does-not-exist")

    def test_empty_file_raises(self, tmp_path):
        sample = _write_sample(tmp_path / "TritonEph2020", [])
        with pytest.raises(ValueError):
            build_triton_eph_2020_state_history(sample)

    def test_frame_constant(self):
        assert TRITON_EPH_2020_FRAME == "J2000"


class TestRealFile:
    """Lightweight checks against the shipped TritonEph2020 file when present."""

    def test_real_file_head(self):
        real = Path("data/kernels/TritonEph2020")
        if not real.exists():
            pytest.skip("TritonEph2020 not present in this checkout")

        # Only parse a small head by decimating aggressively.
        history = build_triton_eph_2020_state_history(real, decimation=1_000_000)
        assert len(history) >= 1
        state = next(iter(history.values()))
        assert state.shape == (6, 1)
        # Triton orbits Neptune near 354,760 km.
        assert np.linalg.norm(state[:3]) / 1000.0 == pytest.approx(354_760, rel=1e-2)
