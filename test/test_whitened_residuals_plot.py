"""Tests for the whitened-residual plots and symmetric whitening."""

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

from orbitdet.analysis import ObservationAccess, Whitening, analysis_frame  # noqa: E402
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame  # noqa: E402
from orbitdet.visualization import (  # noqa: E402
    MahalanobisResiduals,
    SymmetricWhitenedResiduals,
    WhitenedResidualScatter,
)


def _record(epoch, station="Gaia", scan_angle=0.0):
    return {
        "observed_object": "Triton",
        "observatory_code": station,
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
        "scan_angle": scan_angle,
    }


def _set(epochs, residuals, weight_matrix):
    link_ends = {
        links.transmitter: links.body_origin_link_end_id("Triton"),
        links.receiver: links.body_reference_point_link_end_id("Earth", "Gaia"),
    }
    single_set = obs.create_single_observation_set(
        settings.model_settings.ObservableType.angular_position_type,
        link_ends,
        [np.array([[0.5], [0.25]])] * len(epochs),
        list(epochs),
        links.LinkEndType.receiver,
    )
    single_set.set_residuals(np.asarray(residuals, dtype=float).ravel())
    single_set.set_full_weight_matrix(np.asarray(weight_matrix, dtype=float))
    return single_set


def _access(weight_matrix, residuals=None, scan_angle=0.0):
    epochs = [1.0, 2.0]
    if residuals is None:
        residuals = np.array([[1e-6, 2e-6], [3e-6, -4e-6]])
    single_set = _set(epochs, residuals, weight_matrix)
    frame = build_frame([_record(e, scan_angle=scan_angle) for e in epochs])
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0, 0.0]
    return ObservationAccess.build(obs.ObservationCollection([single_set]), frame)


def _cfg(**residuals_cfg):
    return OmegaConf.create({"figures": {"whitened": residuals_cfg}})


def _correlated_weight(sigma_al=1e-6, sigma_ac=2e-6, rho=0.8):
    """A block-diagonal weight matrix with a correlated 2x2 block per observation."""
    cov = np.array(
        [
            [sigma_al**2, rho * sigma_al * sigma_ac],
            [rho * sigma_al * sigma_ac, sigma_ac**2],
        ]
    )
    block = np.linalg.inv(cov)
    weight = np.zeros((4, 4))
    weight[0:2, 0:2] = block
    weight[2:4, 2:4] = block
    return weight


# ===========================================================================
# Symmetric whitening
# ===========================================================================


class TestSymmetricWhitening:
    def test_factor_is_symmetric(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="symmetric")
        assert whitening.kind == "symmetric"
        np.testing.assert_allclose(whitening.factor, whitening.factor.T, atol=1e-12)

    def test_cholesky_factor_is_triangular(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="cholesky")
        assert whitening.kind == "cholesky"
        np.testing.assert_allclose(whitening.factor, np.tril(whitening.factor), atol=1e-12)

    def test_both_methods_give_unit_covariance(self):
        weight = _correlated_weight()
        covariance = np.linalg.inv(weight)
        for method in ("cholesky", "symmetric"):
            whitening = Whitening.from_weight_matrix(weight, method=method)
            cov_z = whitening.factor.T @ covariance @ whitening.factor
            np.testing.assert_allclose(cov_z, np.eye(4), atol=1e-9)

    def test_both_methods_give_same_chi2(self):
        weight = _correlated_weight()
        residual = np.array([1e-6, 2e-6, 3e-6, -4e-6])
        cholesky = Whitening.from_weight_matrix(weight, method="cholesky")
        symmetric = Whitening.from_weight_matrix(weight, method="symmetric")
        assert cholesky.chi2(residual) == pytest.approx(symmetric.chi2(residual), rel=1e-9)
        assert cholesky.chi2(residual) == pytest.approx(residual @ weight @ residual, rel=1e-9)

    def test_symmetric_spreads_correlation(self):
        # Cholesky dumps the correlation onto component 1 (lower-triangular);
        # the symmetric root mixes both components.
        weight = _correlated_weight()
        cholesky = Whitening.from_weight_matrix(weight, method="cholesky")
        symmetric = Whitening.from_weight_matrix(weight, method="symmetric")
        # Cholesky's first row has no dependence on the second component.
        assert cholesky.factor[0, 1] == pytest.approx(0.0, abs=1e-15)
        # The symmetric root's off-diagonal entries are non-zero.
        assert abs(symmetric.factor[0, 1]) > 1e-3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown whitening method"):
            Whitening.from_weight_matrix(_correlated_weight(), method="bogus")

    def test_symmetric_block_diagonal_detected(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="symmetric")
        assert whitening.is_block_diagonal


# ===========================================================================
# WhitenedResidualScatter
# ===========================================================================


class TestWhitenedResidualScatter:
    def test_plots_one_point_per_observation(self):
        access = _access(_correlated_weight())
        plot = WhitenedResidualScatter(_cfg(), access, use_figure_name="whitened")
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 2
        plt.close(fig)

    def test_matches_independent_whitening(self):
        weight = _correlated_weight()
        access = _access(weight)
        plot = WhitenedResidualScatter(_cfg(), access, use_figure_name="whitened")
        fig, ax = plot._make_figure()

        # The plot whitens with the scan-basis weight matrix, not the raw one.
        scan_weight = access.weight_matrix(0, basis="scan")
        frame = analysis_frame(access, unit=None, scan=True, normalized=False)
        residual = frame[
            ["residual_along_scan_rad", "residual_across_scan_rad"]
        ].to_numpy("float64").ravel()
        z = np.linalg.cholesky(scan_weight).T @ residual
        expected = z.reshape(-1, 2)

        offsets = ax.collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], expected[:, 0], rtol=1e-9)
        np.testing.assert_allclose(offsets[:, 1], expected[:, 1], rtol=1e-9)
        plt.close(fig)

    def test_symmetric_method_differs_from_cholesky(self):
        weight = _correlated_weight()
        access = _access(weight)
        fig_c, ax_c = WhitenedResidualScatter(
            _cfg(), access, use_figure_name="whitened", method="cholesky"
        )._make_figure()
        fig_s, ax_s = WhitenedResidualScatter(
            _cfg(), access, use_figure_name="whitened", method="symmetric"
        )._make_figure()
        cholesky_offsets = ax_c.collections[0].get_offsets()
        symmetric_offsets = ax_s.collections[0].get_offsets()
        assert not np.allclose(cholesky_offsets, symmetric_offsets)
        plt.close(fig_c)
        plt.close(fig_s)

    def test_raises_without_scan_angle(self):
        access = _access(_correlated_weight(), scan_angle=np.nan)
        plot = WhitenedResidualScatter(_cfg(), access, use_figure_name="whitened")
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()


# ===========================================================================
# SymmetricWhitenedResiduals
# ===========================================================================


class TestSymmetricWhitenedResiduals:
    def test_two_panels(self):
        access = _access(_correlated_weight())
        plot = SymmetricWhitenedResiduals(_cfg(), access, use_figure_name="whitened")
        fig, axs = plot._make_figure()
        assert len(axs) == 2
        assert len(axs[0].collections) == 1
        assert len(axs[1].collections) == 1
        plt.close(fig)

    def test_uses_symmetric_whitening(self):
        weight = _correlated_weight()
        access = _access(weight)
        plot = SymmetricWhitenedResiduals(_cfg(), access, use_figure_name="whitened")
        fig, axs = plot._make_figure()

        scan_weight = access.weight_matrix(0, basis="scan")
        frame = analysis_frame(access, unit=None, scan=True, normalized=False)
        residual = frame[
            ["residual_along_scan_rad", "residual_across_scan_rad"]
        ].to_numpy("float64").ravel()
        whitening = Whitening.from_weight_matrix(scan_weight, method="symmetric")
        expected = whitening.apply(residual).reshape(-1, 2)

        np.testing.assert_allclose(
            axs[0].collections[0].get_offsets()[:, 1], expected[:, 0], rtol=1e-9
        )
        np.testing.assert_allclose(
            axs[1].collections[0].get_offsets()[:, 1], expected[:, 1], rtol=1e-9
        )
        plt.close(fig)

    def test_legend_reports_rms(self):
        access = _access(_correlated_weight())
        plot = SymmetricWhitenedResiduals(_cfg(), access, use_figure_name="whitened")
        fig, axs = plot._make_figure()
        labels = [t.get_text() for t in axs[0].get_legend().get_texts()]
        assert any("RMS(z)" in label for label in labels)
        plt.close(fig)

    def test_raises_without_scan_angle(self):
        access = _access(_correlated_weight(), scan_angle=np.nan)
        plot = SymmetricWhitenedResiduals(_cfg(), access, use_figure_name="whitened")
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()


# ===========================================================================
# MahalanobisResiduals
# ===========================================================================


class TestMahalanobisResiduals:
    def test_one_point_per_observation(self):
        access = _access(_correlated_weight())
        plot = MahalanobisResiduals(_cfg(), access, use_figure_name="whitened")
        fig, ax = plot._make_figure()
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 2
        plt.close(fig)

    def test_distance_is_invariant_to_whitening_method(self):
        weight = _correlated_weight()
        access = _access(weight)
        fig_c, ax_c = MahalanobisResiduals(
            _cfg(), access, use_figure_name="whitened", method="cholesky"
        )._make_figure()
        fig_s, ax_s = MahalanobisResiduals(
            _cfg(), access, use_figure_name="whitened", method="symmetric"
        )._make_figure()
        np.testing.assert_allclose(
            ax_c.collections[0].get_offsets()[:, 1],
            ax_s.collections[0].get_offsets()[:, 1],
            rtol=1e-9,
        )
        plt.close(fig_c)
        plt.close(fig_s)

    def test_distance_matches_quadratic_form(self):
        weight = _correlated_weight()
        access = _access(weight)
        plot = MahalanobisResiduals(_cfg(), access, use_figure_name="whitened")
        fig, ax = plot._make_figure()

        scan_weight = access.weight_matrix(0, basis="scan")
        frame = analysis_frame(access, unit=None, scan=True, normalized=False)
        residual = frame[
            ["residual_along_scan_rad", "residual_across_scan_rad"]
        ].to_numpy("float64")
        expected = np.array(
            [
                np.sqrt(
                    residual[i]
                    @ scan_weight[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                    @ residual[i]
                )
                for i in range(2)
            ]
        )
        np.testing.assert_allclose(
            ax.collections[0].get_offsets()[:, 1], expected, rtol=1e-9
        )
        plt.close(fig)

    def test_raises_without_scan_angle(self):
        access = _access(_correlated_weight(), scan_angle=np.nan)
        plot = MahalanobisResiduals(_cfg(), access, use_figure_name="whitened")
        with pytest.raises(ValueError, match="scan_angle"):
            plot._make_figure()
