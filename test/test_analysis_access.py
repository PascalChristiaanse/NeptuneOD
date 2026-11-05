"""Tests for the analysis access layer and its views, statistics and tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tudatpy.estimation.observable_models_setup as settings
import tudatpy.estimation.observations as obs
from tudatpy.estimation.observable_models_setup import links

from orbitdet.analysis import (
    ObservationAccess,
    ResidualSummary,
    SourceResidualSummary,
    aggregate,
    chi2,
    normalized_residuals,
    reduced_chi2,
    rms,
    rotate,
    to_arcsec,
    to_mas,
    weighted_rms,
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

    def test_frame_without_set_index_is_all_null(self):
        frame = build_frame([make_record("673", 1.0)])
        collection = obs.ObservationCollection([make_set([1.0])])
        access = ObservationAccess.build(collection, frame)

        # No row carries a set index, so nothing joins.
        assert access.frame["residual_first_rad"].isna().all()
        assert access.frame["weight_first"].isna().all()


class TestWeightMatrices:
    def test_observation_set_and_index_of(self):
        set_a = make_set([1.0], reference_point="673")
        set_b = make_set([2.0], reference_point="689")
        access = make_access([set_a, set_b], [[1.0], [2.0]])

        assert access.observation_set(1) is set_b
        assert access.set_index_of(set_b) == 1
        with pytest.raises(ValueError):
            access.set_index_of(make_set([3.0]))

    def test_ambiguous_set_id_raises(self):
        set_a = make_set([1.0], reference_point="673")
        set_b = make_set([2.0], reference_point="673")
        access = make_access([set_a, set_b], [[1.0], [2.0]])

        assert access.set_indices("673") == [0, 1]
        with pytest.raises(ValueError, match="ambiguous"):
            access.weight_matrix("673")
        with pytest.raises(KeyError):
            access.weight_matrix("nope")

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


class TestChi2:
    def test_residuals_returns_raw_vector(self):
        residuals = np.array([[1e-6, -2e-6], [3e-6, -4e-6]])
        single_set = make_set([1.0, 2.0], residuals=residuals)
        access = make_access([single_set], [[1.0, 2.0]])
        np.testing.assert_allclose(access.residuals(0), residuals.ravel())


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


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


class TestStatistics:
    def test_rms_ignores_non_finite(self):
        assert rms([3.0, 4.0]) == pytest.approx(np.sqrt((9 + 16) / 2))
        assert rms([]) != rms([])  # NaN
        assert rms([np.nan, 3.0]) == pytest.approx(3.0)

    def test_weighted_rms(self):
        # Equal weights reduce to the plain RMS.
        assert weighted_rms([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_normalized_residuals_and_chi2(self):
        residuals = np.array([[2e-6, 0.0], [0.0, 3e-6]])
        sigmas = np.array([[1e-6, 1e-6], [1e-6, 1e-6]])
        single_set = make_set([1.0, 2.0], residuals=residuals)
        access = make_access([single_set], [[1.0, 2.0]])
        access.frame["sigma_first_rad"] = sigmas[:, 0]
        access.frame["sigma_second_rad"] = sigmas[:, 1]

        z = normalized_residuals(access.frame)
        assert z["z_first"].iloc[0] == pytest.approx(2.0)
        assert z["z_second"].iloc[1] == pytest.approx(3.0)
        assert chi2(access.frame) == pytest.approx(4.0 + 9.0)

    def test_reduced_chi2(self):
        assert reduced_chi2(10, 2, 8.0) == pytest.approx(1.0)
        assert reduced_chi2(2, 2, 8.0) != reduced_chi2(2, 2, 8.0)  # NaN

    def test_chi2_from_matrix_reduces_to_componentwise(self):
        from orbitdet.analysis import chi2_from_matrix

        r = np.array([2.0, 3.0])
        W = np.diag([4.0, 9.0])  # sigma = [0.5, 1/3]
        # r^T W r == sum (r/sigma)^2 == (2*2)^2 + (3*3)^2
        assert chi2_from_matrix(r, W) == pytest.approx(16.0 + 81.0)

    def test_chi2_from_matrix_full_matrix_differs_from_componentwise(self):
        from orbitdet.analysis import chi2_from_matrix

        r = np.array([1.0, 1.0])
        W = np.array([[2.0, 1.0], [1.0, 2.0]])
        # Full matrix carries the correlation the componentwise sum ignores.
        assert chi2_from_matrix(r, W) == pytest.approx(6.0)
        assert chi2_from_matrix(r, np.diag(np.diag(W))) == pytest.approx(4.0)

    def test_chi2_from_matrix_drops_non_finite(self):
        from orbitdet.analysis import chi2_from_matrix

        r = np.array([2.0, np.nan])
        W = np.diag([4.0, 9.0])
        assert chi2_from_matrix(r, W) == pytest.approx(16.0)

    def test_chi2_from_matrix_empty_is_nan(self):
        from orbitdet.analysis import chi2_from_matrix

        assert chi2_from_matrix(np.array([np.nan]), np.diag([1.0])) != 0.0  # NaN

    def test_mahalanobis_from_weight_matrix(self):
        from orbitdet.analysis import mahalanobis

        r = np.array([2.0, 3.0])
        W = np.diag([4.0, 9.0])
        assert mahalanobis(r, weight_matrix=W) == pytest.approx(np.sqrt(16.0 + 81.0))

    def test_mahalanobis_from_sigmas_matches_diagonal_matrix(self):
        from orbitdet.analysis import mahalanobis

        r = np.array([2.0, 3.0])
        sigmas = np.array([0.5, 1.0 / 3.0])
        assert mahalanobis(r, sigmas=sigmas) == pytest.approx(np.sqrt(16.0 + 81.0))

    def test_mahalanobis_requires_exactly_one_source(self):
        from orbitdet.analysis import mahalanobis

        r = np.array([1.0])
        with pytest.raises(ValueError, match="exactly one"):
            mahalanobis(r)
        with pytest.raises(ValueError, match="exactly one"):
            mahalanobis(r, weight_matrix=np.eye(1), sigmas=np.array([1.0]))

    def test_mahalanobis_sigmas_ignores_zero_and_non_finite(self):
        from orbitdet.analysis import mahalanobis

        r = np.array([2.0, 3.0, 5.0])
        sigmas = np.array([0.5, 0.0, np.nan])
        assert mahalanobis(r, sigmas=sigmas) == pytest.approx(4.0)

    def test_mahalanobis_sigmas_length_mismatch_raises(self):
        from orbitdet.analysis import mahalanobis

        with pytest.raises(ValueError, match="does not match"):
            mahalanobis(np.array([1.0, 2.0]), sigmas=np.array([1.0]))

    def test_aggregate_groups_and_units(self):
        residuals = np.array([[1e-6, 1e-6], [3e-6, -1e-6]])
        set_a = make_set([1.0, 2.0], reference_point="673", residuals=residuals)
        set_b = make_set([3.0], reference_point="689", residuals=np.array([[0.0, 0.0]]))
        access = make_access([set_a, set_b], [[1.0, 2.0], [3.0]])

        table = aggregate(access.frame, by="set_id", to_unit="arcsec")
        assert set(table["set_id"]) == {"673", "689"}
        row_673 = table.loc[table["set_id"] == "673"].iloc[0]
        assert row_673["n_obs"] == 2
        assert row_673["n_components"] == 4

    def test_aggregate_rejects_unknown_unit(self):
        frame = build_frame([make_record("673", 1.0)])
        with pytest.raises(ValueError, match="Unknown unit"):
            aggregate(frame, by=None, to_unit="furlong")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTables:
    def test_source_summary_builds_frame(self):
        single_set = make_set([1.0, 2.0], residuals=np.ones((2, 2)) * 1e-6)
        access = make_access([single_set], [[1.0, 2.0]])

        table = SourceResidualSummary(access)._build()
        assert isinstance(table, pd.DataFrame)
        assert table["n_obs"].iloc[0] == 2

    def test_residual_summary_produces_multiple_groupings(self):
        single_set = make_set([1.0, 2.0], residuals=np.ones((2, 2)) * 1e-6)
        access = make_access([single_set], [[1.0, 2.0]])

        tables = ResidualSummary(access)._build()
        assert isinstance(tables, dict)
        # Every grouping key is prefixed with the table name.
        assert all(name.startswith("residual_summary_") for name in tables)

    def test_markdown_fallback_renders(self):
        from orbitdet.analysis.tables.base import _to_markdown_fallback

        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        text = _to_markdown_fallback(frame)
        assert "| a | b |" in text
        assert "| 1 | x |" in text
