"""Tests for the analysis frame builder."""

from __future__ import annotations

import numpy as np
import tudatpy.estimation.observable_models_setup as settings
import tudatpy.estimation.observations as obs
from tudatpy.estimation.observable_models_setup import links

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame


def _record(epoch: float) -> dict:
    return {
        "observed_object": "Triton",
        "observatory_code": "673",
        "data_source": "nsdb",
        "source_set_id": "nm0001",
        "reference": "ref",
        "observation_type": "absolute",
        "coordinate_kind": "ra_dec",
        "frame_center": "topocentric",
        "original_equinox": "J2000",
        "original_reference_frame": "astrometric",
        "original_reduction": "none",
        "receptor_type": "ccd",
        "epoch_tdb": epoch,
        "first_value": 2.0,
        "second_value": -0.3,
    }


def _make_set(epochs, residuals, weights):
    link_ends = {
        links.transmitter: links.body_origin_link_end_id("Triton"),
        links.receiver: links.body_reference_point_link_end_id("Earth", "673"),
    }
    single_set = obs.create_single_observation_set(
        settings.model_settings.ObservableType.angular_position_type,
        link_ends,
        [np.array([[0.5], [0.25]])] * len(epochs),
        list(epochs),
        links.LinkEndType.receiver,
    )
    single_set.set_residuals(np.asarray(residuals, dtype=float).ravel())
    single_set.set_tabulated_weights(np.asarray(weights, dtype=float).ravel())
    return single_set


def _access():
    epochs = [1.0, 2.0]
    residuals = np.array([[1e-6, -2e-6], [3e-6, -4e-6]])
    weights = np.array([[4.0, 9.0], [16.0, 25.0]])
    single_set = _make_set(epochs, residuals, weights)
    frame = build_frame([_record(e) for e in epochs])
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 0.0]
    collection = obs.ObservationCollection([single_set])
    return ObservationAccess.build(collection, frame)


class TestAnalysisFrame:
    def test_adds_unit_and_normalized_columns(self):
        frame = analysis_frame(_access())
        assert "residual_first_rad_arcsec" in frame.columns
        assert "z_first" in frame.columns
        # z = r / sigma, sigma = 1/sqrt(w).
        expected = frame["residual_first_rad"] * np.sqrt(frame["weight_first"])
        np.testing.assert_allclose(frame["z_first"].to_numpy(), expected)

    def test_no_unit_columns_when_unit_none(self):
        frame = analysis_frame(_access(), unit=None)
        assert "residual_first_rad_arcsec" not in frame.columns
        assert "z_first" in frame.columns

    def test_mas_unit(self):
        access = _access()
        arcsec = analysis_frame(access, unit="arcsec")
        mas = analysis_frame(access, unit="mas")
        assert "residual_first_rad_mas" in mas.columns
        np.testing.assert_allclose(
            mas["residual_first_rad_mas"].to_numpy(),
            arcsec["residual_first_rad_arcsec"].to_numpy() * 1e3,
        )

    def test_scan_skipped_without_registered_angles(self):
        # No scan angles registered -> scan request is advisory, no columns added.
        frame = analysis_frame(_access(), scan=True)
        assert "residual_along_scan_rad" not in frame.columns

    def test_scan_columns_added_when_angles_registered(self, monkeypatch):
        import orbitdet.data.gaia as gaia

        monkeypatch.setattr(
            gaia, "get_scan_angles_for_epochs", lambda epochs: np.full(len(epochs), 0.3)
        )
        access = _access()
        frame = analysis_frame(access, scan=True)
        assert "scan_angle" in frame.columns
        assert "residual_along_scan_rad" in frame.columns
        assert "residual_across_scan_rad" in frame.columns

    def test_does_not_mutate_access_frame(self):
        access = _access()
        before = set(access.frame.columns)
        analysis_frame(access, scan=False)
        assert set(access.frame.columns) == before

    def test_unknown_unit_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown unit"):
            analysis_frame(_access(), unit="furlong")
