"""Tests for the HST astrometry normalisation helpers (no network required)."""

import numpy as np
import pandas as pd
import pytest

from orbitdet.data.hst_data import (
    OUTPUT_COLUMNS,
    _utc_to_tdb_seconds,
    normalise_mast_table,
    normalise_springer_table,
)


def _mast_table() -> pd.DataFrame:
    """Minimal MAST-shaped table (as returned by ``query_criteria``)."""
    return pd.DataFrame(
        {
            "sci_data_set_name": ["Y0S80103T", "Y0S80104T", "OTHER"],
            "sci_targname": ["TRITON", "TRITON", "NEPTUNE"],
            "sci_instrume": ["FOS", "FOS", "FOS"],
            "sci_spec_1234": ["G270H", "G190H", "MIRROR"],
            "sci_start_time": [
                "1991-10-29T23:40:19.827000",
                "1991-10-30T01:16:54.807000",
                "1991-10-30T02:00:00.000000",
            ],
            "sci_stop_time": [
                "1991-10-30T00:08:24.230000",
                "1991-10-30T01:44:22.350000",
                "1991-10-30T02:00:10.000000",
            ],
            "sci_actual_duration": [1560.0, 1499.985, 10.0],
            "sci_ra": [285.557327, 285.558878, 200.0],
            "sci_dec": [-21.904898, -21.904657, 10.0],
            "ang_sep": [0.0, 0.0, 0.0],
            "sci_pi_last_name": ["STERN", "STERN", "OTHER"],
        }
    )


def _springer_table() -> pd.DataFrame:
    """Minimal Showalter-shaped table (units row already skipped)."""
    return pd.DataFrame(
        {
            "Moon": ["Triton", "Triton", "Naiad"],
            "File name": ["a_flt.fits", "b_flt.fits", "c_flt.fits"],
            "Exposure": [222, 275, 100],
            "TDB": [155762390.183, 303946455.683, 100000000.0],
            "ΔRA": [10.0, -5.0, 1.0],
            "Δdec": [-2.0, 3.0, 0.5],
            "σ(RA,dec)": [0.00651, 0.00736, 0.01],
            "Weight": [1, 0, 1],
            "Instrument": ["ACS", "WFC3", "WFC3"],
            "Filter": ["CLEAR", "F606W", "F606W"],
            "Neptune RA": [315.592062, 327.317837, 100.0],
            "Neptune dec": [-16.968260, -13.618847, 5.0],
        }
    )


class TestNormaliseMast:
    def test_filters_to_target(self):
        table = normalise_mast_table(_mast_table(), target="Triton")

        assert len(table) == 2
        assert (table["moon"] == "Triton").all()

    def test_schema_and_source(self):
        table = normalise_mast_table(_mast_table(), target="Triton")

        assert list(table.columns) == OUTPUT_COLUMNS
        assert (table["source"] == "MAST").all()

    def test_epoch_is_mid_exposure_tdb(self):
        table = normalise_mast_table(_mast_table(), target="Triton")
        # 1991-10-29T23:40:19.827 UTC + 780 s (half of 1560 s) converted to TDB.
        assert table["epoch_TDB"].iloc[0] == pytest.approx(
            _utc_to_tdb_seconds("1991-10-29T23:40:19.827000", 1560.0), abs=1e-6
        )

    def test_ra_dec_passthrough(self):
        table = normalise_mast_table(_mast_table(), target="Triton")

        assert table["ra"].iloc[0] == pytest.approx(285.557327)
        assert table["dec"].iloc[0] == pytest.approx(-21.904898)


class TestNormaliseSpringer:
    def test_filters_to_moon(self):
        table = normalise_springer_table(_springer_table(), moon="Triton")

        assert len(table) == 2
        assert (table["moon"] == "Triton").all()

    def test_schema_and_source(self):
        table = normalise_springer_table(_springer_table(), moon="Triton")

        assert list(table.columns) == OUTPUT_COLUMNS
        assert (table["source"] == "Springer").all()

    def test_absolute_ra_dec_reconstruction(self):
        table = normalise_springer_table(_springer_table(), moon="Triton")

        neptune_ra, neptune_dec = 315.592062, -16.968260
        expected_dec = neptune_dec + (-2.0) / 3600.0
        expected_ra = neptune_ra + 10.0 / 3600.0 / np.cos(np.deg2rad(neptune_dec))

        assert table["dec"].iloc[0] == pytest.approx(expected_dec, abs=1e-9)
        assert table["ra"].iloc[0] == pytest.approx(expected_ra, abs=1e-9)

    def test_epoch_and_weight_passthrough(self):
        table = normalise_springer_table(_springer_table(), moon="Triton")

        assert table["epoch_TDB"].iloc[0] == pytest.approx(155762390.183)
        assert table["weight"].tolist() == [1, 0]

    def test_sigma_applied_to_both_axes(self):
        table = normalise_springer_table(_springer_table(), moon="Triton")

        assert table["ra_error_arcsec"].iloc[0] == pytest.approx(0.00651)
        assert table["dec_error_arcsec"].iloc[0] == pytest.approx(0.00651)

    def test_unknown_moon_raises(self):
        with pytest.raises(LookupError, match="Despina"):
            normalise_springer_table(_springer_table(), moon="Despina")

    def test_all_moons_kept_when_none(self):
        table = normalise_springer_table(_springer_table(), moon=None)

        assert len(table) == 3
        assert set(table["moon"]) == {"Triton", "Naiad"}