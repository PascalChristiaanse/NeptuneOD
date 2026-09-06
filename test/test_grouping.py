"""Tests for the grouping system: Group, GroupList, partitioners, and build_group_list."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from orbitdet.observations.weighting.grouping import (
    Group,
    GroupList,
    _split_by_gap,
    build_group_list,
    partition_by_set,
    partition_by_sections,
    partition_by_timeframes,
)


# ===========================================================================
# Group data structure
# ===========================================================================


class TestGroup:
    def test_create_group(self):
        g = Group(
            indices=np.array([0, 1, 2], dtype=int),
            group_id="test_group",
            level="timeframe",
        )
        assert g.group_id == "test_group"
        assert g.level == "timeframe"
        assert g.size == 3
        assert g.parent_level is None
        assert g.parent_id is None
        assert g.metadata == {}

    def test_group_with_all_fields(self):
        g = Group(
            indices=np.array([5, 6], dtype=int),
            group_id="full_group",
            level="section",
            parent_level="set",
            parent_id="my_set",
            metadata={"t_start": 0.0, "t_end": 100.0},
        )
        assert g.parent_level == "set"
        assert g.parent_id == "my_set"
        assert g.metadata["t_start"] == 0.0

    def test_group_size_property(self):
        g = Group(indices=np.array([10, 20, 30, 40], dtype=int), group_id="size_test")
        assert g.size == 4

    def test_group_empty_indices(self):
        g = Group(indices=np.array([], dtype=int), group_id="empty")
        assert g.size == 0


# ===========================================================================
# GroupList data structure
# ===========================================================================


class TestGroupList:
    def test_empty_group_list(self):
        gl = GroupList()
        assert len(gl) == 0
        assert list(gl) == []

    def test_group_list_from_list(self):
        g1 = Group(indices=np.array([0, 1], dtype=int), group_id="g1", level="set")
        g2 = Group(indices=np.array([2, 3], dtype=int), group_id="g2", level="timeframe")
        gl = GroupList([g1, g2])
        assert len(gl) == 2

    def test_add_group(self):
        gl = GroupList()
        g = Group(indices=np.array([0], dtype=int), group_id="g1")
        gl.add(g)
        assert len(gl) == 1
        assert gl[0] is g

    def test_extend_groups(self):
        gl = GroupList()
        g1 = Group(indices=np.array([0], dtype=int), group_id="g1")
        g2 = Group(indices=np.array([1], dtype=int), group_id="g2")
        gl.extend([g1, g2])
        assert len(gl) == 2

    def test_iteration(self):
        gl = GroupList([
            Group(indices=np.array([0], dtype=int), group_id="a"),
            Group(indices=np.array([1], dtype=int), group_id="b"),
        ])
        ids = [g.group_id for g in gl]
        assert ids == ["a", "b"]

    def test_getitem(self):
        g = Group(indices=np.array([0], dtype=int), group_id="first")
        gl = GroupList([g])
        assert gl[0] is g

    def test_by_level(self):
        g_set = Group(indices=np.array([0, 1], dtype=int), group_id="set", level="set")
        g_tf = Group(indices=np.array([2], dtype=int), group_id="tf", level="timeframe")
        gl = GroupList([g_set, g_tf])

        sets = gl.by_level("set")
        assert len(sets) == 1
        assert sets[0] is g_set

        tfs = gl.by_level("timeframe")
        assert len(tfs) == 1
        assert tfs[0] is g_tf

        nonexistent = gl.by_level("section")
        assert len(nonexistent) == 0

    def test_by_parent(self):
        child = Group(
            indices=np.array([0], dtype=int), group_id="child",
            parent_level="set", parent_id="parent_set",
        )
        other = Group(
            indices=np.array([1], dtype=int), group_id="other",
            parent_level="set", parent_id="other_set",
        )
        gl = GroupList([child, other])

        result = gl.by_parent("parent_set")
        assert len(result) == 1
        assert result[0] is child

    def test_groups_property_returns_copy(self):
        g = Group(indices=np.array([0], dtype=int), group_id="g")
        gl = GroupList([g])
        retrieved = gl.groups
        assert retrieved == [g]
        # Modifying the returned list should not affect the internal list
        retrieved.clear()
        assert len(gl) == 1


# ===========================================================================
# partition_by_set
# ===========================================================================


class TestPartitionBySet:
    def test_single_set_group(self):
        groups = partition_by_set(5, "obs_set_1")
        assert len(groups) == 1
        g = groups[0]
        assert g.level == "set"
        assert g.group_id == "obs_set_1"
        assert np.array_equal(g.indices, np.arange(5, dtype=int))

    def test_zero_observations(self):
        groups = partition_by_set(0, "empty_set")
        assert len(groups) == 1
        assert groups[0].size == 0

    def test_set_id_in_group_id(self):
        groups = partition_by_set(3, "my_observatory")
        assert groups[0].group_id == "my_observatory"


# ===========================================================================
# partition_by_timeframes
# ===========================================================================


class TestPartitionByTimeframes:
    def test_no_gaps_single_timeframe(self):
        times = np.array([0.0, 3600.0, 7200.0, 10800.0])  # 3-hour span, 1-hour gaps
        groups = partition_by_timeframes(times, gap_threshold_hours=4.0)
        assert len(groups) == 1
        assert groups[0].level == "timeframe"
        assert groups[0].size == 4

    def test_large_gap_splits_into_two(self):
        times = np.array([0.0, 3600.0, 7200.0, 86400.0, 90000.0])
        # gap of 79200s (22h) between index 2 and 3
        groups = partition_by_timeframes(times, gap_threshold_hours=4.0)
        assert len(groups) == 2
        assert groups[0].size == 3
        assert groups[1].size == 2

    def test_gap_below_threshold_no_split(self):
        times = np.array([0.0, 7200.0, 10800.0])  # max gap 2h
        groups = partition_by_timeframes(times, gap_threshold_hours=4.0)
        assert len(groups) == 1

    def test_min_obs_per_frame_prevents_split(self):
        # 1 obs before a large gap, min_obs_per_frame=2
        times = np.array([0.0, 100000.0, 100100.0])
        groups = partition_by_timeframes(times, gap_threshold_hours=1.0, min_obs_per_frame=2)
        assert len(groups) == 1  # first frame has only 1 obs, so no split

    def test_min_obs_per_frame_allows_split(self):
        # 2 obs before a large gap, min_obs_per_frame=2
        times = np.array([0.0, 3600.0, 100000.0, 100100.0])
        groups = partition_by_timeframes(times, gap_threshold_hours=1.0, min_obs_per_frame=2)
        assert len(groups) == 2
        assert groups[0].size == 2
        assert groups[1].size == 2

    def test_empty_times(self):
        groups = partition_by_timeframes(np.array([]))
        assert len(groups) == 0

    def test_single_observation(self):
        groups = partition_by_timeframes(np.array([42.0]))
        assert len(groups) == 1
        assert groups[0].size == 1

    def test_group_ids_sequential(self):
        times = np.array([0.0, 100000.0, 200000.0])
        groups = partition_by_timeframes(times, gap_threshold_hours=1.0, set_id="obs1")
        assert groups[0].group_id == "obs1_tf_0000"
        assert groups[1].group_id == "obs1_tf_0001"
        assert groups[2].group_id == "obs1_tf_0002"

    def test_group_ids_without_set_id(self):
        times = np.array([0.0, 100000.0])
        groups = partition_by_timeframes(times, gap_threshold_hours=1.0)
        assert groups[0].group_id == "tf_0000"
        assert groups[1].group_id == "tf_0001"

    def test_metadata_contains_time_bounds(self):
        times = np.array([100.0, 200.0, 100000.0])
        groups = partition_by_timeframes(times, gap_threshold_hours=1.0)
        assert groups[0].metadata["t_start"] == 100.0
        assert groups[0].metadata["t_end"] == 200.0
        assert groups[0].metadata["n_obs"] == 2
        assert groups[1].metadata["t_start"] == 100000.0
        assert groups[1].metadata["n_obs"] == 1


# ===========================================================================
# partition_by_sections
# ===========================================================================


class TestPartitionBySections:
    def test_two_sections(self):
        times = np.array([0.0, 100.0, 500.0, 600.0, 1000.0])
        sections = [("early", 0.0, 400.0), ("late", 400.0, 1000.0)]
        groups = partition_by_sections(times, sections)
        assert len(groups) == 2
        assert groups[0].group_id == "early"
        assert groups[0].size == 2  # times 0, 100
        assert groups[1].group_id == "late"
        assert groups[1].size == 3  # times 500, 600, 1000

    def test_first_matching_section_wins(self):
        times = np.array([100.0, 200.0])
        sections = [("first", 0.0, 500.0), ("second", 0.0, 500.0)]
        groups = partition_by_sections(times, sections)
        assert len(groups) == 1
        assert groups[0].group_id == "first"

    def test_observations_outside_all_sections(self):
        times = np.array([-1000.0, 5000.0])
        sections = [("middle", 0.0, 1000.0)]
        groups = partition_by_sections(times, sections)
        assert len(groups) == 0

    def test_empty_section_not_included(self):
        times = np.array([100.0, 200.0])
        sections = [("empty", 300.0, 400.0), ("populated", 0.0, 250.0)]
        groups = partition_by_sections(times, sections)
        assert len(groups) == 1
        assert groups[0].group_id == "populated"

    def test_extreme_bounds(self):
        times = np.array([0.0, 1e20])
        sections = [("all", -1e30, 1e30)]
        groups = partition_by_sections(times, sections)
        assert len(groups) == 1
        assert groups[0].size == 2

    def test_group_id_includes_set_id(self):
        times = np.array([100.0, 200.0])
        sections = [("night1", 0.0, 500.0)]
        groups = partition_by_sections(times, sections, set_id="obs1")
        assert groups[0].group_id == "obs1_night1"

    def test_metadata_completeness(self):
        times = np.array([100.0, 200.0, 300.0])
        sections = [("segment", 0.0, 500.0)]
        groups = partition_by_sections(times, sections)
        meta = groups[0].metadata
        assert meta["section_name"] == "segment"
        assert meta["t_start"] == 0.0
        assert meta["t_end"] == 500.0
        assert meta["n_obs"] == 3

    def test_level_is_section(self):
        times = np.array([100.0])
        sections = [("a", 0.0, 500.0)]
        groups = partition_by_sections(times, sections)
        assert groups[0].level == "section"


# ===========================================================================
# _split_by_gap (internal helper)
# ===========================================================================


class TestSplitByGap:
    def test_empty_array(self):
        result = _split_by_gap(np.array([]), 3600.0)
        assert result == []

    def test_single_element(self):
        result = _split_by_gap(np.array([42.0]), 3600.0)
        assert result == [[0]]

    def test_no_gap_below_threshold(self):
        times = np.array([0.0, 1800.0, 3599.0])  # gaps < 3600s
        result = _split_by_gap(times, 3600.0)
        assert result == [[0, 1, 2]]

    def test_gap_above_threshold_splits(self):
        times = np.array([0.0, 3601.0])
        result = _split_by_gap(times, 3600.0)
        assert result == [[0], [1]]

    def test_gap_above_threshold_but_min_obs_not_met(self):
        times = np.array([0.0, 100000.0, 100100.0])
        result = _split_by_gap(times, 3600.0, min_obs_per_frame=2)
        # first frame has 1 obs (< 2), so no split
        assert result == [[0, 1, 2]]

    def test_multiple_gaps(self):
        times = np.array([0.0, 100000.0, 100100.0, 200000.0, 200200.0])
        result = _split_by_gap(times, 3600.0)
        assert result == [[0], [1, 2], [3, 4]]

    def test_exact_threshold_no_split(self):
        times = np.array([0.0, 3600.0])  # gap == threshold
        result = _split_by_gap(times, 3600.0)
        assert result == [[0, 1]]

    def test_exact_threshold_splits_when_greater(self):
        times = np.array([0.0, 3600.001])  # gap > threshold
        result = _split_by_gap(times, 3600.0)
        assert result == [[0], [1]]


# ===========================================================================
# build_group_list (pipeline)
# ===========================================================================


class TestBuildGroupList:
    def test_no_levels_defaults_to_set(self):
        times = np.array([0.0, 100.0, 200.0])
        config = OmegaConf.create({"levels": None})
        gl = build_group_list(times, "obs1", config)
        assert len(gl) == 1
        assert gl[0].level == "set"
        assert gl[0].group_id == "obs1"

    def test_single_timeframe_level(self):
        times = np.array([0.0, 3600.0, 100000.0])
        config = OmegaConf.create({
            "levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}],
        })
        gl = build_group_list(times, "obs1", config)
        assert len(gl) == 2
        assert gl[0].level == "timeframe"
        assert gl[1].level == "timeframe"
        # Timeframes should have set as parent
        assert gl[0].parent_level == "set"
        assert gl[0].parent_id == "obs1"

    def test_single_section_level(self):
        times = np.array([0.0, 100.0, 500.0, 600.0])
        config = OmegaConf.create({
            "levels": [{
                "type": "section",
                "sections": [
                    {"name": "early", "start": 0.0, "end": 400.0},
                    {"name": "late", "start": 400.0, "end": 1000.0},
                ],
            }],
        })
        gl = build_group_list(times, "obs1", config)
        assert len(gl) == 2
        assert gl[0].level == "section"
        assert gl[0].group_id == "obs1_early"
        assert gl[1].group_id == "obs1_late"

    def test_section_then_timeframe_hierarchy(self):
        """Sections are coarser; timeframes should have section parent references."""
        times = np.array([0.0, 100.0, 500.0, 600.0, 100000.0, 100100.0])
        config = OmegaConf.create({
            "levels": [
                {
                    "type": "section",
                    "sections": [
                        {"name": "early", "start": 0.0, "end": 1000.0},
                        {"name": "late", "start": 100000.0, "end": 200000.0},
                    ],
                },
                {"type": "timeframe", "gap_threshold_hours": 4.0},
            ],
        })
        gl = build_group_list(times, "obs1", config)
        # Should have 2 sections + 2 timeframes = 4 groups
        assert len(gl) == 4

        sections = gl.by_level("section")
        tfs = gl.by_level("timeframe")
        assert len(sections) == 2
        assert len(tfs) == 2

        # Timeframes should have section parent references
        for tf in tfs:
            assert tf.parent_level == "section"
            assert tf.parent_id is not None

    def test_unassigned_observations_get_other_section(self):
        times = np.array([0.0, 100.0, 10000.0])
        config = OmegaConf.create({
            "levels": [{
                "type": "section",
                "sections": [
                    {"name": "early", "start": 0.0, "end": 200.0},
                ],
            }],
        })
        gl = build_group_list(times, "obs1", config)
        sections = gl.by_level("section")
        # Should have 'early' and 'other'
        assert len(sections) == 2
        section_names = [s.group_id for s in sections]
        assert "obs1_early" in section_names
        assert "obs1_other" in section_names

    def test_missing_type_field_raises_error(self):
        times = np.array([0.0, 100.0])
        config = OmegaConf.create({
            "levels": [{"gap_threshold_hours": 4.0}],
        })
        with pytest.raises(ValueError, match="must have a 'type' field"):
            build_group_list(times, "obs1", config)

    def test_unknown_level_type_raises_error(self):
        times = np.array([0.0, 100.0])
        config = OmegaConf.create({
            "levels": [{"type": "unknown_type"}],
        })
        with pytest.raises(ValueError, match="Unknown grouping level type"):
            build_group_list(times, "obs1", config)

    def test_section_without_sections_list_raises_error(self):
        times = np.array([0.0, 100.0])
        config = OmegaConf.create({
            "levels": [{"type": "section"}],
        })
        with pytest.raises(ValueError, match="requires a 'sections' list"):
            build_group_list(times, "obs1", config)

    def test_section_missing_fields_raises_error(self):
        times = np.array([0.0, 100.0])
        config = OmegaConf.create({
            "levels": [{
                "type": "section",
                "sections": [{"name": "bad", "start": 0.0}],  # missing 'end'
            }],
        })
        with pytest.raises(ValueError, match="must have 'name', 'start', and 'end'"):
            build_group_list(times, "obs1", config)

    def test_set_level_type_is_ignored(self):
        """An explicit 'set' level type should be a no-op."""
        times = np.array([0.0, 100.0, 200.0])
        config = OmegaConf.create({
            "levels": [
                {"type": "set"},
                {"type": "timeframe", "gap_threshold_hours": 4.0},
            ],
        })
        gl = build_group_list(times, "obs1", config)
        assert len(gl) == 1  # single timeframe (no gaps)
        assert gl[0].level == "timeframe"

    def test_empty_times(self):
        times = np.array([])
        config = OmegaConf.create({
            "levels": [{"type": "timeframe", "gap_threshold_hours": 4.0}],
        })
        gl = build_group_list(times, "obs1", config)
        assert len(gl) == 0

    def test_timeframe_with_section_parent(self):
        """Timeframes nested under sections should have correct parent refs."""
        times = np.array([0.0, 100.0, 500.0, 600.0])
        config = OmegaConf.create({
            "levels": [
                {
                    "type": "section",
                    "sections": [
                        {"name": "early", "start": 0.0, "end": 400.0},
                        {"name": "late", "start": 400.0, "end": 1000.0},
                    ],
                },
                {"type": "timeframe", "gap_threshold_hours": 4.0},
            ],
        })
        gl = build_group_list(times, "obs1", config)
        tfs = gl.by_level("timeframe")
        for tf in tfs:
            assert tf.parent_level == "section"
            # The first timeframe should be in 'early', the second in 'late'
            if np.array_equal(tf.indices, np.array([0, 1])):
                assert tf.parent_id == "obs1_early"
            elif np.array_equal(tf.indices, np.array([2, 3])):
                assert tf.parent_id == "obs1_late"