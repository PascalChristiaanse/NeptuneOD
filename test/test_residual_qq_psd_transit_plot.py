"""Tests for the refactored ResidualQQ, ResidualsPSD and GaiaTransitErrorComponents."""

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
from orbitdet.visualization import (  # noqa: E402
    GaiaTransitErrorComponents,
    ResidualQQ,
    ResidualsPSD,
)


def _record(epoch, station="673", scan_angle=np.nan):
    return {
        "observed_object": "Triton",
        "observatory_code": station,
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
        "scan_angle": scan_angle,
    }


def _set(epochs, residuals, reference_point):
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
    single_set.set_tabulated_weights(np.ones(2 * len(epochs)))
    return single_set


def _access(scan_angle=np.nan, n=12, spacing=1.0):
    """Two observatories with enough points for Q-Q and PSD."""
    rng = np.random.default_rng(0)
    epochs_a = [float(i) * spacing for i in range(n)]
    epochs_b = [float(n) * spacing + float(i) * spacing for i in range(n)]
    res_a = rng.normal(0.0, 1e-6, (n, 2))
    res_b = rng.normal(0.0, 2e-6, (n, 2))
    set_a = _set(epochs_a, res_a, "673")
    set_b = _set(epochs_b, res_b, "689")
    frame = build_frame(
        [_record(e, station="673", scan_angle=scan_angle) for e in epochs_a]
        + [_record(e, station="689", scan_angle=scan_angle) for e in epochs_b]
    )
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0] * n + [1.0] * n
    return ObservationAccess.build(obs.ObservationCollection([set_a, set_b]), frame)


def _cfg(name, **inner):
    return OmegaConf.create(inner)


class TestResidualQQ:
    def test_two_panels_with_series(self):
        access = _access()
        plot = ResidualQQ(_cfg("residual_qq"), access)
        fig, axs = plot._make_figure()
        assert len(axs) == 2
        assert len(axs[0].collections) == 2  # two observatories
        assert len(axs[1].collections) == 2
        plt.close(fig)

    def test_reference_line_present(self):
        access = _access()
        plot = ResidualQQ(_cfg("residual_qq"), access)
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert "Normal reference" in labels
        plt.close(fig)

    def test_groups_by_observatory_default(self):
        access = _access()
        plot = ResidualQQ(_cfg("residual_qq"), access)
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert "673" in labels and "689" in labels
        plt.close(fig)

    def test_group_by_argument_overrides_default(self):
        access = _access()
        plot = ResidualQQ(_cfg("residual_qq", group_by="source_set_id"), access)
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert "nm0001" in labels
        plt.close(fig)

    def test_unknown_grouping_column_raises(self):
        access = _access()
        plot = ResidualQQ(_cfg("residual_qq", group_by="not_a_column"), access)
        with pytest.raises(ValueError, match="not in the standard frame"):
            plot._make_figure()


class TestResidualsPSD:
    def test_two_panels_with_spectra(self):
        access = _access()
        plot = ResidualsPSD(_cfg("residuals_psd", method="lombscargle"), access)
        fig, axs = plot._make_figure()
        assert len(axs) == 2
        assert len(axs[0].lines) > 0
        plt.close(fig)

    def test_welch_method_runs(self):
        access = _access()
        cfg = _cfg(
            "residuals_psd",
            method="welch",
            window={"type": "hamming"},
            window_length_days=5.0,
        )
        plot = ResidualsPSD(cfg, access)
        fig, axs = plot._make_figure()
        assert len(axs[0].lines) > 0
        plt.close(fig)

    def test_groups_by_observatory_default(self):
        access = _access()
        cfg = _cfg("residuals_psd", method="lombscargle")
        plot = ResidualsPSD(cfg, access)
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert "673" in labels and "689" in labels
        plt.close(fig)

    def test_unknown_grouping_column_raises(self):
        access = _access()
        cfg = _cfg("residuals_psd", method="lombscargle", group_by="not_a_column")
        plot = ResidualsPSD(cfg, access)
        with pytest.raises(ValueError, match="not in the standard frame"):
            plot._make_figure()


class TestGaiaTransitErrorComponents:
    def test_two_panels(self):
        access = _access(scan_angle=0.3)
        plot = GaiaTransitErrorComponents(_cfg("gaia_transit_error_components"), access)
        fig, axs = plot._make_figure()
        assert len(axs) == 2
        plt.close(fig)

    def test_components_computed(self):
        access = _access(scan_angle=0.3)
        plot = GaiaTransitErrorComponents(_cfg("gaia_transit_error_components"), access)
        components = plot.components()
        assert components.epochs.size == 24
        assert np.isfinite(components.random_sigma_al)
        plt.close("all")

    def test_raises_without_scan_angle(self):
        access = _access(scan_angle=np.nan)
        plot = GaiaTransitErrorComponents(_cfg("gaia_transit_error_components"), access)
        with pytest.raises(ValueError, match="scan angle"):
            plot.components()
