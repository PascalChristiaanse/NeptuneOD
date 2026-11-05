"""Tests for the ``column`` clustering level and :func:`partition_by_column`.

The ``column`` level is the "cluster by whatever the frame already knows" level:
it partitions on a standard-frame column (data source, instrument, reduction
type, ...) instead of on time.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbitdet.observations.clustering import (
    build_cluster_list,
    partition_by_column,
)


def _view(**columns):
    """A stand-in for SetView exposing only ``column(name)``."""

    def column(name):
        if name not in columns:
            raise KeyError(name)
        return np.asarray(columns[name])

    return SimpleNamespace(column=column)


# ===========================================================================
# partition_by_column
# ===========================================================================


class TestPartitionByColumn:
    def test_one_cluster_per_distinct_value(self):
        clusters = partition_by_column(
            np.array(["nsdb", "nsdb", "gaia_archive"]), "s1", column="data_source"
        )
        assert len(clusters) == 2
        assert clusters[0].indices.tolist() == [0, 1]
        assert clusters[1].indices.tolist() == [2]

    def test_first_appearance_order_is_preserved(self):
        clusters = partition_by_column(
            np.array(["b", "a", "b", "a"]), "s1", column="kind"
        )
        assert [c.metadata["value"] for c in clusters] == ["b", "a"]

    def test_cluster_ids_are_slugified(self):
        clusters = partition_by_column(
            np.array(["Voyager 2", "Gaia"]), "s1", column="observatory_code"
        )
        assert clusters[0].cluster_id == "s1_observatory_code_voyager_2"
        assert clusters[1].cluster_id == "s1_observatory_code_gaia"

    def test_missing_values_form_their_own_cluster(self):
        clusters = partition_by_column(
            np.array(["a", None, "a", np.nan], dtype=object), "s1", column="kind"
        )
        assert len(clusters) == 2
        missing = [c for c in clusters if c.metadata["value"] is None]
        assert len(missing) == 1
        assert missing[0].indices.tolist() == [1, 3]
        assert missing[0].cluster_id.endswith("_missing")

    def test_metadata_records_the_column_and_value(self):
        clusters = partition_by_column(np.array(["nsdb"]), "s1", column="data_source")
        assert clusters[0].metadata["column"] == "data_source"
        assert clusters[0].metadata["value"] == "nsdb"
        assert clusters[0].metadata["n_obs"] == 1
        assert clusters[0].level == "column"
        assert clusters[0].layer_name == "column"

    def test_empty_input_yields_no_clusters(self):
        assert partition_by_column(np.array([]), "s1", column="kind") == []

    def test_layer_is_recorded(self):
        clusters = partition_by_column(np.array(["a"]), "s1", layer=3, column="kind")
        assert clusters[0].layer == 3

    def test_numeric_values_are_supported(self):
        clusters = partition_by_column(np.array([1, 2, 1]), "s1", column="instrument")
        assert len(clusters) == 2
        assert clusters[0].indices.tolist() == [0, 2]

    def test_indices_cover_every_observation_exactly_once(self):
        values = np.array(["a", "b", "a", "c", "b"])
        clusters = partition_by_column(values, "s1", column="kind")
        seen = sorted(i for c in clusters for i in c.indices.tolist())
        assert seen == [0, 1, 2, 3, 4]


# ===========================================================================
# The `column` level in build_cluster_list
# ===========================================================================


class TestColumnLevel:
    def test_column_level_partitions_the_set(self):
        view = _view(data_source=["nsdb", "nsdb", "gaia_archive"])
        clusters = build_cluster_list(
            np.array([1.0, 2.0, 3.0]),
            "s1",
            {"levels": [{"type": "column", "column": "data_source"}]},
            view=view,
        )
        assert len(clusters) == 2
        assert {c.level for c in clusters} == {"column"}

    def test_column_level_requires_a_column_name(self):
        with pytest.raises(ValueError, match="requires a 'column' name"):
            build_cluster_list(
                np.array([1.0]), "s1", {"levels": [{"type": "column"}]}, view=_view()
            )

    def test_column_level_requires_a_view(self):
        with pytest.raises(ValueError, match="needs the set's frame"):
            build_cluster_list(
                np.array([1.0]),
                "s1",
                {"levels": [{"type": "column", "column": "data_source"}]},
            )

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError, match="not a standard-frame column"):
            build_cluster_list(
                np.array([1.0]),
                "s1",
                {"levels": [{"type": "column", "column": "nope"}]},
                view=_view(),
            )

    def test_column_level_composes_with_timeframe(self):
        # Coarse: data source.  Fine: observing night within each source.
        view = _view(data_source=["nsdb", "nsdb", "gaia_archive"])
        clusters = build_cluster_list(
            np.array([0.0, 86400.0, 172800.0]),
            "s1",
            {
                "levels": [
                    {"type": "column", "column": "data_source"},
                    {"type": "timeframe", "gap_threshold_hours": 1.0},
                ]
            },
            view=view,
        )
        levels = {c.level for c in clusters}
        assert levels == {"column", "timeframe"}
        # Every timeframe cluster is linked to a column parent.
        timeframes = [c for c in clusters if c.level == "timeframe"]
        assert all(c.parent_level == "column" for c in timeframes)

    def test_time_based_levels_still_work_without_a_view(self):
        clusters = build_cluster_list(
            np.array([0.0, 86400.0]),
            "s1",
            {"levels": [{"type": "timeframe", "gap_threshold_hours": 1.0}]},
        )
        assert len(clusters) == 2

    def test_unknown_level_type_still_raises(self):
        with pytest.raises(ValueError, match="Unknown clustering level type"):
            build_cluster_list(np.array([1.0]), "s1", {"levels": [{"type": "nope"}]})
