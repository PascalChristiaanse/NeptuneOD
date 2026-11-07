"""Tests for the refactored ``ResidualsScan`` plot (access + scan frame)."""

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
from orbitdet.visualization import ResidualsScan  # noqa: E402

ARCSEC = np.rad2deg(1.0) * 3600.0


def _record(epoch: float) -> dict:
    return {
        "observed_object": "Triton",
        "observatory_code": "Gaia",
        "data_source": "gaia",
        "source_set_id": "gaia_dr3",
        "reference": "ref",
        "observation_type": "absolute",
        "coordinate_kind": "ra_dec",
        "frame_center": "barycentric",
        "original_equinox": "J2000",
        "original_reference_frame": "astrometric",
        "original_reduction": "none",
        "receptor_type": "ccd",
        "epoch_tdb": epoch,
        "first_value": 2.0,
        "second_value": -0.3,
    }


def _gaia_set(epochs, residuals, first_value=2.0, second_value=-0.3):
    link_ends = {
        links.transmitter: links.body_origin_link_end_id("Triton"),
        links.receiver: links.body_reference_point_link_end_id("Earth", "Gaia"),
    }
    single_set = obs.create_single_observation_set(
        settings.model_settings.ObservableType.angular_position_type,
        link_ends,
        [np.array([[first_value], [second_value]])] * len(epochs),
        list(epochs),
        links.LinkEndType.receiver,
    )
    single_set.set_residuals(np.asarray(residuals, dtype=float).ravel())
    single_set.set_tabulated_weights(np.ones(2 * len(epochs)))
    return single_set


def _access(epochs, residuals, psi):
    single_set = _gaia_set(epochs, residuals)
    frame = build_frame([_record(e) for e in epochs])
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0] * len(epochs)
    frame["scan_angle"] = np.asarray(psi, dtype=float)
    collection = obs.ObservationCollection([single_set])
    return ObservationAccess.build(collection, frame)


def _cfg(**residuals_cfg):
    return OmegaConf.create(residuals_cfg)


class TestResidualsScan:
    def test_plots_along_and_across_scan(self):
        epochs = [1.0, 2.0]
        # residual [dRA, dDec]; psi=0 -> AL = dDec, AC = dRA*cos(dec).
        residuals = np.array([[1e-6, 2e-6], [3e-6, -4e-6]])
        access = _access(epochs, residuals, psi=[0.0, 0.0])
        plot = ResidualsScan(_cfg(), access)
        fig, axs = plot._make_figure()
        assert len(axs) == 2
        al = axs[0].collections[0].get_offsets()
        ac = axs[1].collections[0].get_offsets()
        # AL at psi=0 == dDec in arcsec.
        assert al[0, 1] == pytest.approx(2e-6 * ARCSEC, rel=1e-6)
        # AC at psi=0 == dRA * cos(dec) in arcsec (dec = -0.3).
        assert ac[0, 1] == pytest.approx(1e-6 * np.cos(-0.3) * ARCSEC, rel=1e-6)
        plt.close(fig)

    def test_scan_angles_pulled_from_frame(self):
        # psi = pi/2 -> AL = dRA*cos(dec), AC = -dDec.
        epochs = [1.0]
        residuals = np.array([[1e-6, 2e-6]])
        access = _access(epochs, residuals, psi=[np.pi / 2])
        plot = ResidualsScan(_cfg(), access)
        fig, axs = plot._make_figure()
        al = axs[0].collections[0].get_offsets()
        ac = axs[1].collections[0].get_offsets()
        assert al[0, 1] == pytest.approx(1e-6 * np.cos(-0.3) * ARCSEC, rel=1e-6)
        assert ac[0, 1] == pytest.approx(-2e-6 * ARCSEC, rel=1e-6)
        plt.close(fig)

    def test_groups_by_observatory_default(self):
        epochs = [1.0, 2.0]
        residuals = np.array([[1e-6, 2e-6], [3e-6, -4e-6]])
        access = _access(epochs, residuals, psi=[0.0, 0.0])
        plot = ResidualsScan(_cfg(), access)
        fig, axs = plot._make_figure()
        # All rows share observatory_code "Gaia" -> a single series.
        assert len(axs[0].collections) == 1
        plt.close(fig)

    def test_raises_without_scan_angle(self):
        # No scan_angle in the frame (non-scanning source) -> cannot plot scan frame.
        epochs = [1.0]
        residuals = np.array([[1e-6, 2e-6]])
        single_set = _gaia_set(epochs, residuals)
        frame = build_frame([_record(1.0)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        collection = obs.ObservationCollection([single_set])
        access = ObservationAccess.build(collection, frame)
        plot = ResidualsScan(_cfg(), access)
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()

    def test_legend_reports_rms(self):
        epochs = [1.0, 2.0]
        residuals = np.array([[1e-6, 2e-6], [3e-6, -4e-6]])
        access = _access(epochs, residuals, psi=[0.0, 0.0])
        plot = ResidualsScan(_cfg(), access)
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert any("RMS:" in label and "arcsec" in label for label in labels)
        plt.close(fig)

    def test_mas_unit(self):
        epochs = [1.0]
        residuals = np.array([[1e-6, 2e-6]])
        access = _access(epochs, residuals, psi=[0.0])
        plot = ResidualsScan(_cfg(axes={"unit": "mas"}), access)
        fig, axs = plot._make_figure()
        al = axs[0].collections[0].get_offsets()
        assert al[0, 1] == pytest.approx(2e-6 * ARCSEC * 1e3, rel=1e-6)
        plt.close(fig)
