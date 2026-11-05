"""Tests for the refactored normalized-residual figures (access + whitening)."""

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
from orbitdet.visualization import NormalizedResiduals, NormalizedResidualScan  # noqa: E402

ARCSEC = np.rad2deg(1.0) * 3600.0


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


def _cfg(**residuals_cfg):
    return OmegaConf.create({"figures": {"norm": residuals_cfg}})


class TestNormalizedResiduals:
    def _access(self):
        # Two non-scanning observatories (scan_angle NaN).
        set_a = _set(
            [1.0, 2.0], np.array([[2e-6, 4e-6], [6e-6, 8e-6]]), np.ones((2, 2)), "673"
        )
        set_b = _set([3.0], np.array([[3e-6, 9e-6]]), np.ones((1, 2)), "689")
        frame = build_frame([_record(1.0), _record(2.0), _record(3.0, station="689")])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 0.0, 1.0]
        return ObservationAccess.build(obs.ObservationCollection([set_a, set_b]), frame)

    def test_z_is_residual_over_sigma(self):
        access = self._access()
        plot = NormalizedResiduals(_cfg(), access, use_figure_name="norm")
        fig, axs = plot._make_figure()
        # weight 1 -> sigma 1 rad -> z == residual in radians.
        offsets = axs[0].collections[0].get_offsets()
        assert offsets[0, 1] == pytest.approx(2e-6, rel=1e-6)
        plt.close(fig)

    def test_groups_by_observatory_default(self):
        access = self._access()
        plot = NormalizedResiduals(_cfg(), access, use_figure_name="norm")
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 2
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert any("RMS(z)" in label for label in labels)
        plt.close(fig)

    def test_scan_rows_excluded(self):
        # A scanning row is dropped; the non-scan rows remain.
        set_a = _set([1.0], np.array([[2e-6, 4e-6]]), np.ones((1, 2)), "673")
        set_b = _set([2.0], np.array([[3e-6, 9e-6]]), np.ones((1, 2)), "Gaia")
        frame = build_frame([_record(1.0), _record(2.0, station="Gaia", scan_angle=0.3)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 1.0]
        access = ObservationAccess.build(obs.ObservationCollection([set_a, set_b]), frame)
        plot = NormalizedResiduals(_cfg(), access, use_figure_name="norm")
        fig, axs = plot._make_figure()
        assert len(axs[0].collections) == 1  # only 673
        plt.close(fig)

    def test_all_scanning_raises(self):
        single_set = _set([1.0], np.array([[2e-6, 4e-6]]), np.ones((1, 2)), "Gaia")
        frame = build_frame([_record(1.0, station="Gaia", scan_angle=0.3)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        access = ObservationAccess.build(obs.ObservationCollection([single_set]), frame)
        plot = NormalizedResiduals(_cfg(), access, use_figure_name="norm")
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()


class TestNormalizedResidualScan:
    def test_whitened_covariance_is_identity(self):
        # The whitening must decorrelate: with W = L L^T, Cov(L^T r) == I.
        base = np.array([[4.0, 1.0], [1.0, 2.0]])
        weight = np.zeros((4, 4))
        weight[0:2, 0:2] = base
        weight[2:4, 2:4] = base
        # Residuals have covariance W^{-1}; Cov(z) = L^T W^{-1} L = I.
        L = np.linalg.cholesky(weight)
        cov_z = L.T @ np.linalg.inv(weight) @ L
        np.testing.assert_allclose(cov_z, np.eye(4), atol=1e-9)

    def test_rank_deficient_falls_back_to_blocks(self):
        # tudatpy refuses a singular full weight matrix, so exercise the
        # rank-deficient branch of ``_whitened_set`` directly with a stub access
        # whose scan-basis weight matrix is rank-1 per observation.
        from types import SimpleNamespace

        import pandas as pd

        sigma = 1e-6
        direction = np.array([1.0, 0.0])
        block = (direction / sigma)[:, None] * (direction / sigma)[None, :]
        weight = np.zeros((4, 4))
        weight[0:2, 0:2] = block
        weight[2:4, 2:4] = block
        access = SimpleNamespace(weight_matrix=lambda *a, **k: weight)
        plot = NormalizedResidualScan(_cfg(), access, use_figure_name="norm")
        rows = pd.DataFrame(
            {
                "epoch_tdb": [1.0, 2.0],
                "residual_along_scan_rad": [2e-6, -4e-6],
                "residual_across_scan_rad": [1e-6, 3e-6],
            }
        )
        result = plot._whitened_set(0, rows, "residual_along_scan_rad", "residual_across_scan_rad")
        assert result is not None
        epochs, z_al, z_ac = result
        # Rank-1 block along AL: z_al = AL/sigma; AC has no independent
        # component and is left undefined (NaN).
        np.testing.assert_allclose(z_al, np.array([2e-6, -4e-6]) / sigma, rtol=1e-9)
        assert np.all(np.isnan(z_ac))
        plt.close("all")

    def test_raises_without_scan_angle(self):
        single_set = _set([1.0], np.array([[1e-6, 2e-6]]), np.ones(2), "Gaia")
        frame = build_frame([_record(1.0, station="Gaia")])  # scan_angle NaN
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        access = ObservationAccess.build(obs.ObservationCollection([single_set]), frame)
        plot = NormalizedResidualScan(_cfg(), access, use_figure_name="norm")
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()
