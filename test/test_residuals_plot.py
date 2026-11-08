"""Tests for the refactored ``Residuals`` plot (access + clustering)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tudatpy.estimation.observable_models_setup as settings  # noqa: E402
import tudatpy.estimation.observations as obs  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tudatpy.estimation.observable_models_setup import links  # noqa: E402

from orbitdet.analysis import ObservationAccess  # noqa: E402
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame  # noqa: E402
from orbitdet.visualization import Residuals  # noqa: E402


def _record(station: str, epoch: float) -> dict:
    return {
        "observed_object": "Triton",
        "observatory_code": station,
        "data_source": "nsdb",
        "source_set_id": f"src_{station}",
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


def _set(epochs, residuals, weights, reference_point):
    link_ends = {
        links.transmitter: links.body_origin_link_end_id("Triton"),
        links.receiver: links.body_reference_point_link_end_id("Earth", reference_point),
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
    """Two sets at different observatories, so clustering has something to split."""
    set_a = _set(
        [1.0, 2.0],
        np.array([[1e-6, -2e-6], [3e-6, -4e-6]]),
        np.ones((2, 2)),
        reference_point="673",
    )
    set_b = _set([3.0], np.array([[5e-6, 6e-6]]), np.ones((1, 2)), reference_point="689")
    frame = build_frame([_record("673", 1.0), _record("673", 2.0), _record("689", 3.0)])
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 0.0, 1.0]
    collection = obs.ObservationCollection([set_a, set_b])
    return ObservationAccess.build(collection, frame)


def _cfg(**residuals_cfg):
    return OmegaConf.create(residuals_cfg)


class TestResidualsGrouping:
    def test_default_groups_by_observatory(self):
        plot = Residuals(_cfg(), _access())
        fig, axs = plot._make_figure()
        # Two observatories -> two scatter series on each axis.
        assert len(axs[0].collections) == 2
        assert len(axs[1].collections) == 2
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert any("673" in label for label in labels)
        assert any("689" in label for label in labels)
        plt.close(fig)

    def test_group_by_set_id(self):
        plot = Residuals(_cfg(group_by="set_id"), _access())
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 2
        plt.close(fig)

    def test_group_by_source_set_id(self):
        plot = Residuals(_cfg(group_by="source_set_id"), _access())
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 2
        plt.close(fig)

    def test_group_by_single_value_is_one_series(self):
        plot = Residuals(_cfg(group_by="coordinate_kind"), _access())
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 1
        plt.close(fig)

    def test_config_group_by_used(self):
        plot = Residuals(_cfg(group_by="source_set_id"), _access())
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 2
        plt.close(fig)

    def test_unknown_group_by_raises(self):
        plot = Residuals(_cfg(group_by="not_a_column"), _access())
        with pytest.raises(ValueError, match="not_a_column"):
            plot._make_figure()

    def test_points_are_in_arcsec_by_default(self):
        plot = Residuals(_cfg(), _access())
        fig, axs = plot._make_figure()
        offsets = axs[0].collections[0].get_offsets()
        # First residual 1e-6 rad -> ~0.206 arcsec.
        assert offsets[0, 1] == pytest.approx(np.rad2deg(1e-6) * 3600.0, rel=1e-3)
        plt.close(fig)

    def test_mas_unit(self):
        plot = Residuals(_cfg(axes={"unit": "mas"}), _access())
        fig, axs = plot._make_figure()
        offsets = axs[0].collections[0].get_offsets()
        assert offsets[0, 1] == pytest.approx(np.rad2deg(1e-6) * 3600.0 * 1e3, rel=1e-3)
        plt.close(fig)

    def test_rejected_rows_excluded_by_default(self):
        access = _access()
        # Mark the 689 row rejected and refresh the frame.
        access.frame.loc[access.frame["set_id"] == "689", "rejected"] = True
        plot = Residuals(_cfg(), access)
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 1  # only 673 remains
        plt.close(fig)

    def test_include_rejected(self):
        access = _access()
        access.frame.loc[access.frame["set_id"] == "689", "rejected"] = True
        plot = Residuals(_cfg(include_rejected=True), access)
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 2
        plt.close(fig)

    def test_legend_reports_rms(self):
        plot = Residuals(_cfg(), _access())
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert any("RMS:" in label and "arcsec" in label for label in labels)
        plt.close(fig)
