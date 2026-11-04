"""Tests for the analysis access layer and its views, statistics and tables."""

from __future__ import annotations

import numpy as np
import pytest
import tudatpy.estimation.observable_models_setup as settings
import tudatpy.estimation.observations as obs
from tudatpy.estimation.observable_models_setup import links

from orbitdet.analysis import (
    ObservationAccess,
    rotate,
    to_arcsec,
    to_mas,
    with_scan_frame,
)
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def make_record(station: str, epoch: float, **overrides) -> dict:
    """One standard-frame record."""
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
        **overrides,
    }


def make_set(epochs, reference_point="673", residuals=None, weights=None):
    """Build a real SingleObservationSet with optional residuals and weights."""
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
    if residuals is not None:
        single_set.set_residuals(np.asarray(residuals, dtype=float).ravel())
    if weights is not None:
        single_set.set_tabulated_weights(np.asarray(weights, dtype=float).ravel())
    return single_set


def make_access(sets, epochs_per_set, observations=None):
    """Build an ObservationAccess over synthetic sets and a matching frame.

    ``epochs_per_set[i]`` are the epochs of set ``i``; the frame rows are the
    concatenation of all epochs and ``observation_set_index`` marks which set
    built each row.
    """
    records = []
    set_index_column = []
    for index, epochs in enumerate(epochs_per_set):
        for epoch in epochs:
            records.append(make_record("673", epoch))
            set_index_column.append(float(index))
    frame = build_frame(records)
    frame[OBSERVATION_SET_INDEX_COLUMN] = set_index_column
    collection = obs.ObservationCollection(list(sets))
    return ObservationAccess.build(collection, frame)


# ---------------------------------------------------------------------------
# Access layer
# ---------------------------------------------------------------------------


class TestObservationAccess:
    def test_joins_residuals_and_weights(self):
        residuals = np.array([[1e-6, -2e-6], [3e-6, -4e-6]])
        weights = np.array([[4.0, 9.0], [16.0, 25.0]])
        single_set = make_set([1.0, 2.0], residuals=residuals, weights=weights)
        access = make_access([single_set], [[1.0, 2.0]])

        frame = access.frame
        np.testing.assert_allclose(frame["residual_first_rad"], residuals[:, 0])
        np.testing.assert_allclose(frame["residual_second_rad"], residuals[:, 1])
        np.testing.assert_allclose(frame["weight_first"], weights[:, 0])
        # Diagonal weights -> sigma = 1/sqrt(w).
        np.testing.assert_allclose(frame["sigma_first_rad"], 1.0 / np.sqrt(weights[:, 0]))
        assert (frame["set_id"] == "673").all()
        assert (frame["observable_type"] == "absolute").all()
        assert not frame["has_full_weight_matrix"].any()

    def test_record_and_residual_getters(self):
        residuals = np.array([[1e-6, -2e-6], [3e-6, -4e-6]])
        single_set = make_set([1.0, 2.0], residuals=residuals)
        access = make_access([single_set], [[1.0, 2.0]])

        assert access.residual(1) == pytest.approx((3e-6, -4e-6))
        assert access.record(1)["set_id"] == "673"
        assert len(access) == 2
        assert access.set_ids == ["673"]

    def test_set_frame_and_by_set_id(self):
        set_a = make_set([1.0, 2.0], reference_point="673")
        set_b = make_set([3.0], reference_point="689")
        access = make_access([set_a, set_b], [[1.0, 2.0], [3.0]])

        assert len(access.set_frame(0)) == 2
        assert len(access.set_frame(1)) == 1
        assert len(access.by_set_id("689")) == 1

    def test_matches_rows_by_epoch_not_row_order(self):
        """A set whose epochs are not in frame order still joins to the right rows."""
        # Frame rows in epoch order 1,2,3 but the set lists them 3,1,2.
        single_set = make_set([3.0, 1.0, 2.0], residuals=np.zeros((3, 2)))
        access = make_access([single_set], [[1.0, 2.0, 3.0]])
        # Every row still joins (the set covers all three epochs).
        assert access.frame["residual_first_rad"].notna().all()

    def test_refresh_reads_new_residuals_without_recomputing_weights(self):
        single_set = make_set(
            [1.0], residuals=np.array([[1e-6, 0.0]]), weights=np.array([[4.0, 4.0]])
        )
        access = make_access([single_set], [[1.0]])
        assert access.residual(0)[0] == pytest.approx(1e-6)

        # Estimation changes residuals; weights must be preserved by refresh.
        single_set.set_residuals(np.array([5e-6, 0.0]))
        access.refresh()
        assert access.residual(0)[0] == pytest.approx(5e-6)
        assert access.frame["weight_first"].iloc[0] == pytest.approx(4.0)

    def test_full_matrix_marginal_sigma_uses_inverse(self):
        block = np.array([[2.0, 0.5], [0.5, 1.0]])
        full = np.zeros((4, 4))
        full[0:2, 0:2] = block
        full[2:4, 2:4] = block

        single_set = make_set([1.0, 2.0], weights=np.zeros((2, 2)))
        single_set.set_full_weight_matrix(full)
        access = make_access([single_set], [[1.0, 2.0]])

        marginal = np.sqrt(np.diag(np.linalg.inv(full)))
        # Columns are per component: first -> [d0, d2], second -> [d1, d3].
        np.testing.assert_allclose(access.frame["sigma_first_rad"], [marginal[0], marginal[2]])
        np.testing.assert_allclose(access.frame["sigma_second_rad"], [marginal[1], marginal[3]])
        assert access.frame["has_full_weight_matrix"].all()

    def test_frame_without_set_index_is_all_null(self):
        frame = build_frame([make_record("673", 1.0)])
        collection = obs.ObservationCollection([make_set([1.0])])
        access = ObservationAccess.build(collection, frame)

        # No row carries a set index, so nothing joins.
        assert access.frame["residual_first_rad"].isna().all()
        assert access.frame["weight_first"].isna().all()


class TestWeightMatrices:
    def test_diagonal_weight_matrix_and_diagonal(self):
        weights = np.array([[4.0, 9.0], [16.0, 25.0]])
        single_set = make_set([1.0, 2.0], weights=weights)
        access = make_access([single_set], [[1.0, 2.0]])

        matrix = access.weight_matrix(0)
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(np.diag(matrix), weights.ravel())
        np.testing.assert_allclose(access.weight_diagonal(0), weights.ravel())
        assert not access.has_full_weight_matrix(0)

    def test_covariance_is_inverse_of_weight(self):
        weights = np.array([[4.0, 9.0]])
        single_set = make_set([1.0], weights=weights)
        access = make_access([single_set], [[1.0]])

        covariance = access.covariance_matrix(0)
        np.testing.assert_allclose(covariance, np.linalg.inv(access.weight_matrix(0)))
        np.testing.assert_allclose(np.diag(covariance), 1.0 / weights.ravel())

    def test_full_weight_matrix_is_off_diagonal_and_invertible(self):
        block = np.array([[2.0, 0.5], [0.5, 1.0]])
        full = np.zeros((4, 4))
        full[0:2, 0:2] = block
        full[2:4, 2:4] = block

        single_set = make_set([1.0, 2.0], weights=np.zeros((2, 2)))
        single_set.set_full_weight_matrix(full)
        access = make_access([single_set], [[1.0, 2.0]])

        assert access.has_full_weight_matrix(0)
        np.testing.assert_allclose(access.weight_matrix(0), full)
        np.testing.assert_allclose(access.covariance_matrix(0), np.linalg.inv(full))

    def test_observation_set_and_index_of(self):
        set_a = make_set([1.0], reference_point="673")
        set_b = make_set([2.0], reference_point="689")
        access = make_access([set_a, set_b], [[1.0], [2.0]])

        assert access.observation_set(1) is set_b
        assert access.set_index_of(set_b) == 1
        with pytest.raises(ValueError):
            access.set_index_of(make_set([3.0]))

    def test_weight_matrices_keyed_by_set_id(self):
        set_a = make_set([1.0], reference_point="673", weights=np.array([[4.0, 9.0]]))
        set_b = make_set([2.0], reference_point="689", weights=np.array([[1.0, 1.0]]))
        access = make_access([set_a, set_b], [[1.0], [2.0]])

        matrices = access.weight_matrices()
        assert set(matrices) == {"673", "689"}
        np.testing.assert_allclose(np.diag(matrices["673"]), [4.0, 9.0])

    def test_set_level_accessors_accept_id_or_object(self):
        weights = np.array([[4.0, 9.0]])
        single_set = make_set([1.0], reference_point="Gaia", weights=weights)
        access = make_access([single_set], [[1.0]])

        # By identifier string, by set object, and by index all agree.
        by_id = access.weight_matrix("Gaia")
        by_obj = access.weight_matrix(single_set)
        by_index = access.weight_matrix(0)
        np.testing.assert_allclose(by_id, by_index)
        np.testing.assert_allclose(by_obj, by_index)
        assert access.set_index_for("Gaia") == 0
        assert access.set_indices("Gaia") == [0]
        assert access.observation_set("Gaia") is single_set

    def test_ambiguous_set_id_raises(self):
        set_a = make_set([1.0], reference_point="673")
        set_b = make_set([2.0], reference_point="673")
        access = make_access([set_a, set_b], [[1.0], [2.0]])

        assert access.set_indices("673") == [0, 1]
        with pytest.raises(ValueError, match="ambiguous"):
            access.weight_matrix("673")
        with pytest.raises(KeyError):
            access.weight_matrix("nope")

    def test_covariance_and_weight_in_scan_basis(self, monkeypatch):
        """basis='scan' applies T C T^T and its dual, using the registered angles."""
        import orbitdet.data.gaia as gaia

        psi = np.array([0.7, -0.3])
        monkeypatch.setattr(gaia, "get_scan_angles_for_epochs", lambda epochs: psi[: len(epochs)])

        weights = np.array([[4.0, 9.0], [16.0, 25.0]])
        single_set = make_set([1.0, 2.0], reference_point="Gaia", weights=weights)
        access = make_access([single_set], [[1.0, 2.0]])

        # Canonical is unchanged.
        W_canonical = access.weight_matrix("Gaia")
        np.testing.assert_allclose(np.diag(W_canonical), weights.ravel())

        # The scan basis is block-diagonal with the registered psi per observation.
        basis = access.basis_transform("Gaia")
        assert basis.transform.shape == (4, 4)
        np.testing.assert_allclose(
            access.weight_matrix("Gaia", basis="scan"), basis.weight(W_canonical)
        )
        np.testing.assert_allclose(
            access.covariance_matrix("Gaia", basis="scan"),
            basis.covariance(access.covariance_matrix("Gaia")),
        )

    def test_scan_basis_without_angles_raises(self, monkeypatch):
        import orbitdet.data.gaia as gaia

        monkeypatch.setattr(gaia, "get_scan_angles_for_epochs", lambda epochs: None)
        single_set = make_set([1.0], weights=np.array([[1.0, 1.0]]))
        access = make_access([single_set], [[1.0]])

        with pytest.raises(ValueError, match="no registered scan angles"):
            access.basis_transform(0, kind="scan")

    def test_scan_basis_defaults_to_raw_ra(self, monkeypatch):
        """The default basis applies cos(dec): the pipeline's RA is raw (α, not α cos δ)."""
        import orbitdet.data.gaia as gaia

        psi = np.array([0.7])
        monkeypatch.setattr(gaia, "get_scan_angles_for_epochs", lambda epochs: psi[: len(epochs)])
        # make_set uses observed [[0.5],[0.25]] -> dec = 0.25 rad.
        single_set = make_set([1.0], reference_point="Gaia", weights=np.array([[1.0, 1.0]]))
        access = make_access([single_set], [[1.0]])

        default = access.basis_transform("Gaia")  # de_projected defaults to False
        assert not default.de_projected
        assert not default.is_orthogonal
        cos_dec = np.cos(0.25)
        np.testing.assert_allclose(
            default.transform,
            [[np.sin(0.7) * cos_dec, np.cos(0.7)], [np.cos(0.7) * cos_dec, -np.sin(0.7)]],
        )

        # Explicit de-projection drops the factor and is orthogonal.
        de_projected = access.basis_transform("Gaia", de_projected=True)
        assert de_projected.is_orthogonal
        assert de_projected.de_projected


class TestRejection:
    def test_rejected_column_marks_rows_dropped_from_a_set(self):
        """A set narrowed to a subset of its rows flags the rest as rejected."""
        residuals = np.array([[1e-6, 0.0]])
        # Built from two epochs, but only the first survives.
        single_set = make_set([1.0], residuals=residuals)
        access = make_access([single_set], [[1.0, 2.0]])

        frame = access.frame
        assert frame["rejected"].tolist() == [False, True]
        assert len(access.accepted()) == 1
        assert len(access.rejected()) == 1
        # The rejected row no longer joins a residual.
        assert np.isnan(frame["residual_first_rad"].iloc[1])

    def test_empty_set_marks_all_rows_rejected(self):
        single_set = make_set([])  # no surviving observations
        access = make_access([single_set], [[1.0, 2.0]])
        assert access.frame["rejected"].all()

    def test_refresh_resets_rejection(self):
        single_set = make_set([1.0], residuals=np.array([[1e-6, 0.0]]))
        access = make_access([single_set], [[1.0, 2.0]])
        assert access.frame["rejected"].iloc[1]

        # A set that now holds every epoch marks nothing rejected on refresh.
        full = make_set([1.0, 2.0])
        access.collection = obs.ObservationCollection([full])
        access.refresh()
        assert not access.frame["rejected"].any()


class TestViews:
    def test_to_arcsec_adds_columns_without_mutating(self):
        residuals = np.array([[1e-6, -2e-6]])
        single_set = make_set([1.0], residuals=residuals, weights=np.ones((1, 2)))
        access = make_access([single_set], [[1.0]])

        viewed = to_arcsec(access.frame)
        expected = 1e-6 * np.rad2deg(1.0) * 3600.0
        assert viewed["residual_first_rad_arcsec"].iloc[0] == pytest.approx(expected)
        # Original frame is not mutated.
        assert "residual_first_rad_arcsec" not in access.frame.columns

    def test_to_mas_scales_by_1000(self):
        residuals = np.array([[1e-6, 0.0]])
        single_set = make_set([1.0], residuals=residuals)
        access = make_access([single_set], [[1.0]])
        mas = to_mas(access.frame)
        arcsec = to_arcsec(access.frame)
        np.testing.assert_allclose(
            mas["residual_first_rad_mas"].iloc[0],
            arcsec["residual_first_rad_arcsec"].iloc[0] * 1e3,
        )

    def test_rotate_matches_reference_convention(self):
        first = np.array([1.0, 0.0])
        second = np.array([0.0, 1.0])
        dec = np.array([0.0, 0.0])
        psi = np.array([np.pi / 2, 0.0])
        along, across = rotate(first, second, dec, psi)
        # At psi=pi/2, along = first, across = -second... check against formula.
        east = first * np.cos(dec)
        expected_along = east * np.sin(psi) + second * np.cos(psi)
        expected_across = east * np.cos(psi) - second * np.sin(psi)
        np.testing.assert_allclose(along, expected_along)
        np.testing.assert_allclose(across, expected_across)

    def test_with_scan_frame(self):
        residuals = np.array([[1e-6, 2e-6]])
        single_set = make_set([1.0], residuals=residuals, weights=np.ones((1, 2)))
        access = make_access([single_set], [[1.0]])
        frame = access.frame.copy()
        frame["scan_angle"] = np.pi / 4

        scanned = with_scan_frame(frame)
        assert "residual_along_scan_rad" in scanned.columns
        assert "residual_across_scan_rad" in scanned.columns
        # Input unchanged.
        assert "residual_along_scan_rad" not in frame.columns

    def test_with_scan_frame_requires_psi(self):
        frame = build_frame([make_record("673", 1.0)]).drop(columns=["scan_angle"])
        frame["residual_first_rad"] = 0.0
        frame["residual_second_rad"] = 0.0
        with pytest.raises(ValueError, match="scan_angle"):
            with_scan_frame(frame)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


