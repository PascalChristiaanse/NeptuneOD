"""Tests for :mod:`orbitdet.observations.plan`.

A plan is the default-plus-overrides config shape shared by the weighting and
outlier engines.  These tests cover compilation (including the legacy shapes it
must keep accepting) and routing.
"""

import pytest
from omegaconf import OmegaConf

from orbitdet.observations.plan import Plan, PlanEntry, compile_plan
from orbitdet.preprocessing.schema import empty_frame


def _frame(*sources, codes=None):
    """A frame with one row per data source, for routing tests."""
    frame = empty_frame()
    frame["data_source"] = list(sources)
    frame["observatory_code"] = list(codes or ["689"] * len(sources))
    frame["epoch_tdb"] = [float(i) for i in range(len(sources))]
    return frame


# ===========================================================================
# Compilation
# ===========================================================================


class TestCompilePlan:
    def test_default_only(self):
        plan = compile_plan({"default": {"strategy": {"type": "id_v2"}}})
        assert isinstance(plan, Plan)
        assert plan.default.strategy == {"type": "id_v2"}
        assert plan.overrides == []
        assert len(plan) == 1

    def test_default_plus_overrides(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {
                        "select": "data_source == 'gaia_archive'",
                        "strategy": {"type": "gaia_archive"},
                    }
                ],
            }
        )
        assert len(plan) == 2
        assert plan.overrides[0].selection.query == "data_source == 'gaia_archive'"

    def test_missing_default_raises(self):
        # There is no implicit default: a plan must say what happens to
        # observations no override matches.
        with pytest.raises(ValueError, match="no 'default' entry"):
            compile_plan({})

    def test_top_level_strategy_is_not_a_default(self):
        # The pre-plan shape is gone; `strategy:` alone is not a plan.
        with pytest.raises(ValueError, match="no 'default' entry"):
            compile_plan({"strategy": {"type": "id_v2"}})

    def test_omegaconf_input(self):
        cfg = OmegaConf.create(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {"select": "epoch_tdb > 1.0", "strategy": {"type": "id"}},
                ],
            }
        )
        plan = compile_plan(cfg)
        assert len(plan.overrides) == 1

    def test_override_without_select_raises(self):
        with pytest.raises(ValueError, match="has no 'select'"):
            compile_plan(
                {
                    "default": {"strategy": {"type": "id_v2"}},
                    "overrides": [{"strategy": {"type": "id"}}],
                }
            )

    def test_override_with_empty_select_raises(self):
        # An empty selection would match everything and shadow later entries.
        with pytest.raises(ValueError, match="empty 'select'"):
            compile_plan(
                {
                    "default": {"strategy": {"type": "id_v2"}},
                    "overrides": [{"select": None, "strategy": {"type": "id"}}],
                }
            )

    def test_entry_without_strategy_raises(self):
        with pytest.raises(ValueError, match="has no 'strategy'"):
            compile_plan({"default": {"clustering": {"levels": []}}})

    def test_unknown_entry_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            compile_plan({"default": {"strategy": {"type": "id_v2"}, "stratgy": {}}})

    def test_non_mapping_entry_raises(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            compile_plan({"default": "id_v2"})

    def test_invalid_select_is_rejected_at_compile_time(self):
        with pytest.raises(ValueError, match="unknown column"):
            compile_plan(
                {
                    "default": {"strategy": {"type": "id_v2"}},
                    "overrides": [{"select": "nope == 1", "strategy": {"type": "id"}}],
                }
            )


# ===========================================================================
# Clustering attachment
# ===========================================================================


class TestClustering:
    def test_top_level_clustering_is_the_plan_default(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "clustering": {"levels": [{"type": "timeframe"}]},
            }
        )
        assert plan.default.clustering == {"levels": [{"type": "timeframe"}]}

    def test_entry_clustering_overrides_the_plan_default(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "clustering": {"levels": [{"type": "timeframe"}]},
                "overrides": [
                    {
                        "select": "data_source == 'gaia_archive'",
                        "strategy": {"type": "gaia_archive"},
                        "clustering": {"levels": [{"type": "set"}]},
                    }
                ],
            }
        )
        assert plan.overrides[0].clustering == {"levels": [{"type": "set"}]}

    def test_entry_without_clustering_inherits_the_plan_default(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "clustering": {"levels": [{"type": "timeframe"}]},
                "overrides": [
                    {"select": "data_source == 'gaia_archive'", "strategy": {"type": "id"}}
                ],
            }
        )
        assert plan.overrides[0].clustering == {"levels": [{"type": "timeframe"}]}

    def test_default_clustering_argument_is_the_fallback(self):
        plan = compile_plan(
            {"default": {"strategy": {"type": "id_v2"}}},
            default_clustering={"levels": [{"type": "set"}]},
        )
        assert plan.default.clustering == {"levels": [{"type": "set"}]}


# ===========================================================================
# Routing
# ===========================================================================


class TestEntryFor:
    def _plan(self):
        return compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {
                        "select": "data_source == 'gaia_archive'",
                        "strategy": {"type": "gaia_archive"},
                    },
                    {
                        "select": "observatory_code == 'Voyager 2'",
                        "strategy": {"type": "layered"},
                    },
                ],
            }
        )

    def test_first_matching_override_wins(self):
        plan = self._plan()
        entry = plan.entry_for(_frame("gaia_archive", codes=["Gaia"]))
        assert entry.strategy == {"type": "gaia_archive"}

    def test_second_override_matches_when_the_first_does_not(self):
        plan = self._plan()
        entry = plan.entry_for(_frame("voyager", codes=["Voyager 2"]))
        assert entry.strategy == {"type": "layered"}

    def test_unmatched_falls_back_to_default(self):
        plan = self._plan()
        entry = plan.entry_for(_frame("nsdb", codes=["689"]))
        assert entry is plan.default

    def test_a_set_matches_when_any_row_does(self):
        # A set is routed as a whole, so one matching row is enough.
        plan = self._plan()
        entry = plan.entry_for(_frame("nsdb", "gaia_archive", codes=["689", "Gaia"]))
        assert entry.strategy == {"type": "gaia_archive"}

    def test_empty_frame_falls_back_to_default(self):
        plan = self._plan()
        assert plan.entry_for(empty_frame()) is plan.default

    def test_none_frame_falls_back_to_default(self):
        # Without a view there is no frame to match on.
        plan = self._plan()
        assert plan.entry_for(None) is plan.default

    def test_override_order_is_respected(self):
        # Two overrides that both match: the earlier one wins.
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {"select": "epoch_tdb >= 0", "strategy": {"type": "first"}},
                    {"select": "epoch_tdb >= 0", "strategy": {"type": "second"}},
                ],
            }
        )
        assert plan.entry_for(_frame("nsdb")).strategy == {"type": "first"}


# ===========================================================================
# Representation
# ===========================================================================


class TestRepr:
    def test_entry_label_names_the_query(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {"select": "data_source == 'gaia_archive'", "strategy": {"type": "id"}}
                ],
            }
        )
        assert "data_source" in plan.overrides[0].label

    def test_default_label(self):
        plan = compile_plan({"default": {"strategy": {"type": "id_v2"}}})
        assert plan.default.label == "default"

    def test_plan_repr_counts_overrides(self):
        plan = compile_plan(
            {
                "default": {"strategy": {"type": "id_v2"}},
                "overrides": [
                    {"select": "epoch_tdb > 1.0", "strategy": {"type": "id"}},
                    {"select": "epoch_tdb > 2.0", "strategy": {"type": "id"}},
                ],
            }
        )
        assert "overrides=2" in repr(plan)

    def test_plan_entry_is_a_dataclass(self):
        entry = PlanEntry(selection=None, strategy="id_v2")
        assert entry.label == "default"
