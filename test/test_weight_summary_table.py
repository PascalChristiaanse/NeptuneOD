"""Tests for the per-group weight summary table.

``WeightSummary`` replaces the old ``visualization.WeightSummaryTable``: it reads
the access layer's derived frame for provenance and residuals, and joins the
weights report only for the cluster assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tudatpy.estimation.observable_models_setup as settings
import tudatpy.estimation.observations as obs
from tudatpy.estimation.observable_models_setup import links

from orbitdet.analysis import ObservationAccess
from orbitdet.analysis.tables import WeightSummary, coordinate_axis_labels
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _record(station: str, epoch: float, **overrides) -> dict:
    return {
        "observed_object": "Triton",
        "observatory_code": station,
        "data_source": "nsdb",
        "source_set_id": "nm0001",
        "reference": "Veiga et al. (1996)",
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


def _make_set(epochs, reference_point="673", residuals=None):
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
    return single_set


def _make_access(rows, residuals):
    """Build an access layer over one set whose frame rows are ``rows``.

    ``rows`` is a list of ``(record_overrides, epoch)`` pairs; ``residuals`` is
    the matching ``(n, 2)`` array in radians.
    """
    epochs = [epoch for _, epoch in rows]
    records = [_record("673", epoch, **overrides) for overrides, epoch in rows]
    frame = build_frame(records)
    frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0] * len(rows)
    single_set = _make_set(epochs, residuals=residuals)
    return ObservationAccess.build(obs.ObservationCollection([single_set]), frame)


def _weights_report(epochs, cluster_ids, set_id="673"):
    """A minimal weights report, as the engine emits it."""
    return pd.DataFrame(
        {
            "set_id": set_id,
            "time": epochs,
            "cluster_id": cluster_ids,
            "parent_id": "",
            "parent_level": "",
            "cluster_level": "timeframe",
            "strategy": "id_v2",
        }
    )


#: 1 arcsec in radians.
ARCSEC = np.deg2rad(1.0) / 3600.0


@pytest.fixture
def simple_access():
    """Two sources, two observations each, with known residuals."""
    rows = [
        ({"source_set_id": "nm0001", "reference": "Source A"}, 100.0),
        ({"source_set_id": "nm0001", "reference": "Source A"}, 200.0),
        ({"source_set_id": "nm0002", "reference": "Source B"}, 300.0),
        ({"source_set_id": "nm0002", "reference": "Source B"}, 400.0),
    ]
    # Source A: RA residuals 3, 4 arcsec -> RMS sqrt((9+16)/2) = 3.5355
    # Source B: RA residuals 6, 8 arcsec -> RMS sqrt((36+64)/2) = 7.0711
    residuals = np.array(
        [
            [3 * ARCSEC, 1 * ARCSEC],
            [4 * ARCSEC, 2 * ARCSEC],
            [6 * ARCSEC, 3 * ARCSEC],
            [8 * ARCSEC, 4 * ARCSEC],
        ]
    )
    return _make_access(rows, residuals)


# ---------------------------------------------------------------------------
# Coordinate labels
# ---------------------------------------------------------------------------


class TestCoordinateAxisLabels:
    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("ra_dec", (r"$\alpha$", r"$\delta$")),
            ("position_angle_separation", (r"$\rho$", r"$\theta$")),
            ("cartesian_xy", ("$x$", "$y$")),
            ("", ("", "")),
            ("something_else", ("", "")),
        ],
    )
    def test_labels(self, kind, expected):
        assert coordinate_axis_labels(kind) == expected


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


class TestBuild:
    def test_default_groups_by_source(self, simple_access):
        table = WeightSummary(simple_access)._build()
        assert list(table["source_set_id"]) == ["nm0001", "nm0002"]
        assert list(table["n_obs"]) == [2, 2]

    def test_rms_values(self, simple_access):
        table = WeightSummary(simple_access)._build()
        np.testing.assert_allclose(table["ra_rms_arcsec"], [np.sqrt(12.5), np.sqrt(50.0)])
        np.testing.assert_allclose(table["dec_rms_arcsec"], [np.sqrt(2.5), np.sqrt(12.5)])

    def test_axis_labels_from_coordinate_kind(self, simple_access):
        table = WeightSummary(simple_access)._build()
        assert table["ra_type"].iloc[0] == r"$\alpha$"
        assert table["dec_type"].iloc[0] == r"$\delta$"

    def test_group_by_observatory(self, simple_access):
        table = WeightSummary(simple_access, by=("observatory_code",))._build()
        assert list(table["observatory_code"]) == ["673"]
        assert table["n_obs"].iloc[0] == 4

    def test_group_by_multiple_columns(self, simple_access):
        table = WeightSummary(simple_access, by=("source_set_id", "observatory_code"))._build()
        assert list(table.columns[:2]) == ["source_set_id", "observatory_code"]
        assert len(table) == 2

    def test_extra_columns_carried_through(self, simple_access):
        table = WeightSummary(simple_access, columns=("reference", "original_reduction"))._build()
        assert list(table["reference"]) == ["Source A", "Source B"]
        assert list(table["original_reduction"]) == ["none", "none"]

    def test_missing_extra_column_is_skipped(self, simple_access):
        table = WeightSummary(simple_access, columns=("nope",))._build()
        assert "nope" not in table.columns

    def test_unknown_grouping_column_raises(self, simple_access):
        with pytest.raises(ValueError, match="not in the frame"):
            WeightSummary(simple_access, by=("nope",))._build()

    def test_unknown_unit_raises(self, simple_access):
        with pytest.raises(ValueError, match="Unknown unit"):
            WeightSummary(simple_access, to_unit="furlongs")

    def test_mas_unit(self, simple_access):
        table = WeightSummary(simple_access, to_unit="mas")._build()
        np.testing.assert_allclose(
            table["ra_rms_arcsec"], [np.sqrt(12.5) * 1e3, np.sqrt(50.0) * 1e3]
        )

    def test_empty_frame(self):
        frame = build_frame([_record("673", 100.0)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        access = ObservationAccess.build(obs.ObservationCollection([]), frame)
        table = WeightSummary(access)._build()
        assert table.empty

    def test_rejected_rows_excluded(self, simple_access):
        """A rejected row must not contribute to the RMS."""
        simple_access.frame.loc[0, "rejected"] = True
        table = WeightSummary(simple_access)._build()
        # Source A now has only the 4 arcsec residual.
        np.testing.assert_allclose(table["ra_rms_arcsec"].iloc[0], 4.0)


# ---------------------------------------------------------------------------
# Weights-report join
# ---------------------------------------------------------------------------


class TestWeightsJoin:
    def test_cluster_id_available_after_join(self, simple_access):
        report = _weights_report([100.0, 200.0, 300.0, 400.0], ["c0", "c0", "c1", "c1"])
        table = WeightSummary(simple_access, weights=report, by=("cluster_id",))._build()
        assert list(table["cluster_id"]) == ["c0", "c1"]
        assert list(table["n_obs"]) == [2, 2]

    def test_group_by_source_and_cluster(self, simple_access):
        report = _weights_report([100.0, 200.0, 300.0, 400.0], ["c0", "c0", "c1", "c1"])
        table = WeightSummary(
            simple_access, weights=report, by=("source_set_id", "cluster_id")
        )._build()
        assert list(table.columns[:2]) == ["source_set_id", "cluster_id"]
        assert len(table) == 2

    def test_join_does_not_duplicate_rows(self, simple_access):
        report = _weights_report([100.0, 200.0, 300.0, 400.0], ["c0", "c0", "c1", "c1"])
        table = WeightSummary(simple_access, weights=report)._build()
        assert table["n_obs"].sum() == 4

    def test_partial_report_leaves_nulls(self, simple_access):
        """A report covering only some epochs must not drop the other rows."""
        report = _weights_report([100.0, 200.0], ["c0", "c0"])
        table = WeightSummary(simple_access, weights=report)._build()
        assert table["n_obs"].sum() == 4

    def test_no_report_means_no_cluster_column(self, simple_access):
        table = WeightSummary(simple_access)._build()
        assert "cluster_id" not in table.columns


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_to_string_has_headers_and_total(self, simple_access):
        table = WeightSummary(simple_access)._build()
        text = WeightSummary(simple_access).to_string(table)
        assert "source_set_id" in text
        assert "RA RMS [arcsec]" in text
        assert "Total" in text
        assert "4" in text

    def test_to_string_empty(self):
        frame = build_frame([_record("673", 100.0)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        access = ObservationAccess.build(obs.ObservationCollection([]), frame)
        assert WeightSummary(access).to_string() == "No weight data."

    def test_to_latex_structure(self, simple_access):
        table = WeightSummary(simple_access)._build()
        latex = WeightSummary(simple_access).to_latex(table)
        assert r"\begin{tabular}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert "Total" in latex

    def test_to_latex_escapes_underscores(self, simple_access):
        table = WeightSummary(simple_access)._build()
        latex = WeightSummary(simple_access).to_latex(table)
        assert r"nm0001" in latex
        assert "nm0001_" not in latex

    def test_to_latex_empty(self):
        frame = build_frame([_record("673", 100.0)])
        frame[OBSERVATION_SET_INDEX_COLUMN] = [0.0]
        access = ObservationAccess.build(obs.ObservationCollection([]), frame)
        assert WeightSummary(access).to_latex() == "% No weight data."

    def test_column_kinds_classified(self, simple_access):
        summary = WeightSummary(simple_access)
        summary._build()
        assert summary.column_kinds["n_obs"] == "count"
        assert summary.column_kinds["ra_rms_arcsec"] == "angle"
        assert summary.column_kinds["source_set_id"] == "text"


# ---------------------------------------------------------------------------
# Varying extra columns
# ---------------------------------------------------------------------------


class TestVaryingColumns:
    def test_varying_column_is_joined_with_warning(self, simple_access, caplog):
        simple_access.frame.loc[0, "original_reduction"] = "Gaia DR2"
        table = WeightSummary(simple_access, columns=("original_reduction",))._build()
        assert "Gaia DR2" in table["original_reduction"].iloc[0]
        assert "none" in table["original_reduction"].iloc[0]
        assert "varies within a group" in caplog.text


# ---------------------------------------------------------------------------
# Report-only mode (no access layer)
# ---------------------------------------------------------------------------


def _report(rows):
    """A weights report shaped like the engine's, with residuals in arcsec."""
    return pd.DataFrame(rows)


class TestFromWeights:
    def _report(self):
        return _report(
            [
                {
                    "set_id": "673",
                    "time": 100.0,
                    "cluster_id": "c0",
                    "source_name": "Source A",
                    "ra_residual_arcsec": 3.0,
                    "dec_residual_arcsec": 1.0,
                    "ra_type": r"$\alpha$",
                    "dec_type": r"$\delta$",
                },
                {
                    "set_id": "673",
                    "time": 200.0,
                    "cluster_id": "c0",
                    "source_name": "Source A",
                    "ra_residual_arcsec": 4.0,
                    "dec_residual_arcsec": 2.0,
                    "ra_type": r"$\alpha$",
                    "dec_type": r"$\delta$",
                },
                {
                    "set_id": "673",
                    "time": 300.0,
                    "cluster_id": "c1",
                    "source_name": "Source B",
                    "ra_residual_arcsec": 6.0,
                    "dec_residual_arcsec": 3.0,
                    "ra_type": r"$\alpha$",
                    "dec_type": r"$\delta$",
                },
            ]
        )

    def test_groups_by_source_name(self):
        summary = WeightSummary.from_weights(self._report())
        table = summary._build()
        assert list(table["source_name"]) == ["Source A", "Source B"]
        assert list(table["n_obs"]) == [2, 1]

    def test_rms_matches_arcsec_residuals(self):
        summary = WeightSummary.from_weights(self._report())
        table = summary._build()
        np.testing.assert_allclose(table["ra_rms_arcsec"].iloc[0], np.sqrt(12.5))
        np.testing.assert_allclose(table["ra_rms_arcsec"].iloc[1], 6.0)

    def test_axis_labels_come_from_report(self):
        summary = WeightSummary.from_weights(self._report())
        table = summary._build()
        assert table["ra_type"].iloc[0] == r"$\alpha$"

    def test_group_by_cluster(self):
        summary = WeightSummary.from_weights(self._report(), by=("source_name", "cluster_id"))
        table = summary._build()
        assert list(table["cluster_id"]) == ["c0", "c1"]

    def test_empty_report(self):
        summary = WeightSummary.from_weights(pd.DataFrame())
        assert summary._build().empty

    def test_report_without_residuals_warns(self, caplog):
        report = pd.DataFrame([{"source_name": "A", "cluster_id": "c0"}])
        summary = WeightSummary.from_weights(report)
        table = summary._build()
        assert "no residual columns" in caplog.text
        assert table["n_obs"].iloc[0] == 1
