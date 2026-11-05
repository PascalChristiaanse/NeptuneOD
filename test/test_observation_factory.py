"""Tests for the observation dataset registry and factory system."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tudatpy.estimation.observable_models_setup as settings
import tudatpy.estimation.observations as obs
from omegaconf import OmegaConf
from tudatpy.estimation.observable_models_setup import links

from orbitdet.observations import (
    ObservationDatasetConfig,
    SimulatedObservationConfig,
    create_observation_collection,
    create_observation_dataset,
    get_factory,
    list_registered_types,
    register_dataset_factory,
)
from orbitdet.observations import collection as collection_module
from orbitdet.observations.registry import _FACTORY_REGISTRY, dispatch_key
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN, build_frame


def make_record(station: str, epoch: float, **overrides: object) -> dict[str, object]:
    """Build one standard-frame record, overriding fields as needed."""
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


@pytest.fixture(autouse=True)
def reset_registry_state():
    original_registry = dict(_FACTORY_REGISTRY)
    yield
    _FACTORY_REGISTRY.clear()
    _FACTORY_REGISTRY.update(original_registry)


class TestRegistry:
    def test_register_and_retrieve_factory(self):
        test_type = "test_type"

        @register_dataset_factory(test_type)
        def wrapped(cfg):
            return cfg, "settings"

        retrieved = get_factory(test_type)

        assert retrieved is wrapped
        assert test_type in list_registered_types()

    def test_duplicate_registration_raises_error(self):
        test_type = "duplicate_test"

        @register_dataset_factory(test_type)
        def factory1(cfg):
            return cfg, "result1"

        with pytest.raises(ValueError, match="already registered"):

            @register_dataset_factory(test_type)
            def factory2(cfg):
                return cfg, "result2"

        assert get_factory(test_type) is factory1

    def test_unknown_type_lookup_raises_error(self):
        with pytest.raises(ValueError, match="No factory registered"):
            get_factory("nonexistent_type_xyz")

    def test_list_registered_types_includes_simulated(self):
        types = list_registered_types()

        assert "simulated" in types
        assert types == sorted(types)


class TestConfigDataclasses:
    def test_observation_dataset_config_creation(self):
        cfg = ObservationDatasetConfig(type="ground_ccd", file="test.csv")

        assert cfg.type == "ground_ccd"
        assert cfg.file == "test.csv"
        assert cfg.weight == 1.0
        assert cfg.metadata == {}

    def test_simulated_observation_config_creation(self):
        cfg = SimulatedObservationConfig(
            type="simulated",
            file="test.csv",
            start_date_observation_period="2025-01-01T00:00:00",
            end_date_observation_period="2025-01-02T00:00:00",
            cadence=3600.0,
            observable_types="relative_cartesian_position",
            noise_sigma=1.0,
        )

        assert cfg.type == "simulated"
        assert cfg.cadence == 3600.0
        assert cfg.noise_sigma == 1.0

    def test_config_frozen(self):
        cfg = ObservationDatasetConfig(type="ground_ccd", file="test.csv")

        with pytest.raises(Exception):
            cfg.file = "modified.csv"

    def test_weight_validation(self):
        with pytest.raises(ValueError, match="weight must be in"):
            ObservationDatasetConfig(type="ground_ccd", file="test.csv", weight=1.5)

        with pytest.raises(ValueError, match="weight must be in"):
            ObservationDatasetConfig(type="ground_ccd", file="test.csv", weight=-0.1)


class TestCentralFactory:
    def test_dispatches_using_registered_factory(self):
        system_of_bodies = SimpleNamespace()
        factory_calls = []

        @register_dataset_factory("test_dispatch")
        def factory(cfg, dataset_cfg, system):
            factory_calls.append((dataset_cfg.type, system))
            return "dataset", "settings"

        dataset_cfg = OmegaConf.create({"type": "test_dispatch", "file": "test.csv", "weight": 1.0})
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})

        assert create_observation_dataset(cfg, dataset_cfg, system_of_bodies) == (
            "dataset",
            "settings",
        )
        assert factory_calls == [("test_dispatch", system_of_bodies)]

    def test_missing_type_field_raises_error(self):
        dataset_cfg = OmegaConf.create({"file": "test.csv", "weight": 1.0})
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})

        with pytest.raises(ValueError, match="must have either a 'type' field"):
            create_observation_dataset(cfg, dataset_cfg, SimpleNamespace())

    def test_dispatch_key_selects_the_factory(self):
        # A config carrying the standard frame's two normalised fields dispatches
        # on those, without consulting 'type' at all.  A synthetic key is used so
        # the test does not depend on which real factories happen to be
        # registered; test_standard_frame_factories covers the real ones.
        dataset_cfg = OmegaConf.create(
            {
                "observation_type": "test_kind",
                "coordinate_kind": "test_coords",
                "file": "test.csv",
            }
        )
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})
        factory_calls = []

        # A factory that declares `frame` receives it, which is how the dispatcher
        # reaches the standard-frame consumers.
        @register_dataset_factory(dispatch_key("test_kind", "test_coords"))
        def factory(cfg, dataset_cfg, system_of_bodies, *, frame):
            factory_calls.append((system_of_bodies, frame))
            return "dataset", "settings"

        assert create_observation_dataset(
            cfg, dataset_cfg, SimpleNamespace(), frame="THE_FRAME"
        ) == ("dataset", "settings")
        assert len(factory_calls) == 1
        assert factory_calls[0][1] == "THE_FRAME"

    def test_dispatch_key_is_preferred_over_type(self):
        # A config with both must dispatch on the normalised pair, so that adding
        # the fields to a config actually switches it over.
        dataset_cfg = OmegaConf.create(
            {
                "observation_type": "test_pref",
                "coordinate_kind": "test_coords",
                "type": "legacy_should_not_be_used",
                "file": "test.csv",
            }
        )
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})

        @register_dataset_factory(dispatch_key("test_pref", "test_coords"))
        def preferred(cfg, dataset_cfg, system_of_bodies):
            return "preferred", "settings"

        @register_dataset_factory("legacy_should_not_be_used")
        def legacy(cfg, dataset_cfg, system_of_bodies):
            return "legacy", "settings"

        result = create_observation_dataset(cfg, dataset_cfg, SimpleNamespace())

        assert result == ("preferred", "settings")

    def test_factory_declaring_a_frame_receives_it(self):
        # A factory that consumes a standard frame declares `frame`; the dispatcher
        # passes it.  One that does not (Gaia, Voyager, simulated) is called the
        # old way, so a single call site serves both.
        dataset_cfg = OmegaConf.create({"type": "frame_consumer", "file": "t.csv"})
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})
        received = {}

        @register_dataset_factory("frame_consumer")
        def factory(cfg, dataset_cfg, system_of_bodies, *, frame):
            received["frame"] = frame
            return "dataset", "settings"

        create_observation_dataset(cfg, dataset_cfg, SimpleNamespace(), frame="THE_FRAME")

        assert received["frame"] == "THE_FRAME"

    def test_factory_without_a_frame_parameter_is_called_positionally(self):
        dataset_cfg = OmegaConf.create({"type": "no_frame", "file": "t.csv"})
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})

        @register_dataset_factory("no_frame")
        def factory(cfg, dataset_cfg, system_of_bodies):
            return "dataset", "settings"

        assert create_observation_dataset(cfg, dataset_cfg, SimpleNamespace(), frame="UNUSED") == (
            "dataset",
            "settings",
        )

    def test_dispatch_key_falls_back_to_type_when_unregistered(self):
        # A standard-frame config whose key has no factory must still work via
        # its legacy type, so adding the two fields to a config cannot strand a
        # dataset whose factory has not been migrated yet.
        dataset_cfg = OmegaConf.create(
            {
                "observation_type": "unmigrated_kind",
                "coordinate_kind": "unmigrated_coords",
                "type": "legacy_name",
                "file": "test.csv",
            }
        )
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})

        @register_dataset_factory("legacy_name")
        def factory(cfg, dataset_cfg, system_of_bodies):
            return "dataset", "settings"

        assert create_observation_dataset(cfg, dataset_cfg, SimpleNamespace()) == (
            "dataset",
            "settings",
        )

    def test_aliases_resolve_to_the_same_factory(self):
        # The fused NSDB type strings stay registered as aliases so the 82
        # committed configs keep loading unchanged.
        @register_dataset_factory("canonical_name", aliases=["fused_typo_name"])
        def factory(cfg, dataset_cfg, system_of_bodies):
            return "dataset", "settings"

        assert get_factory("canonical_name") is factory
        assert get_factory("fused_typo_name") is factory

    def test_alias_clash_leaves_the_registry_untouched(self):
        @register_dataset_factory("taken_name")
        def first(cfg, dataset_cfg, system_of_bodies):
            return "first", "settings"

        with pytest.raises(ValueError, match="already registered"):

            @register_dataset_factory("fresh_name", aliases=["taken_name"])
            def second(cfg, dataset_cfg, system_of_bodies):
                return "second", "settings"

        # The canonical name must not have been registered by the failed call.
        assert get_factory("taken_name") is first
        with pytest.raises(ValueError, match="No factory registered"):
            get_factory("fresh_name")

    def test_unknown_type_raises_error(self):
        dataset_cfg = OmegaConf.create({"type": "unknown_modality_xyz", "file": "test.csv"})
        cfg = OmegaConf.create({"datasets": {"test": dataset_cfg}})
        with pytest.raises(ValueError, match="No factory registered"):
            create_observation_dataset(cfg, dataset_cfg, SimpleNamespace())


class TestCollectionBuilder:
    def test_collection_builds_and_merges_datasets(self, monkeypatch):
        system_of_bodies = SimpleNamespace()
        collection_cfg = OmegaConf.create(
            {
                "datasets": {
                    "first": {"type": "alpha", "file": "data1.csv", "weight": 1.0},
                    "second": {"type": "beta", "file": "data2.csv", "weight": 0.8},
                }
            }
        )

        created = []

        def fake_create_observation_dataset(cfg, dataset_cfg, system, *, frame=None):
            created.append((dataset_cfg.type, system))
            return f"dataset:{dataset_cfg.type}", f"settings:{dataset_cfg.type}"

        class FakeObservationCollection:
            def __init__(self, observation_sets):
                self.observation_sets = observation_sets

        class FakeMergedCollection:
            def get_single_observation_sets(self):
                return []

        merged_collection = FakeMergedCollection()
        merge_mock = MagicMock(return_value=merged_collection)

        monkeypatch.setattr(
            collection_module, "create_observation_dataset", fake_create_observation_dataset
        )
        monkeypatch.setattr(
            collection_module.obs, "ObservationCollection", FakeObservationCollection
        )
        monkeypatch.setattr(collection_module.obs, "merge_observation_collections", merge_mock)

        result = create_observation_collection(collection_cfg, system_of_bodies)

        assert result[0] is merged_collection
        assert result[1] == ["settings:alpha", "settings:beta"]
        assert result[2] == {}  # dataset_metadata
        assert result[3].empty  # standard frame: no NSDB dataset in this config
        assert created == [("alpha", system_of_bodies), ("beta", system_of_bodies)]
        merge_mock.assert_called_once()
        merged_collections = merge_mock.call_args.args[0]
        assert len(merged_collections) == 1
        assert isinstance(merged_collections[0], FakeObservationCollection)
        assert merged_collections[0].observation_sets == ["dataset:alpha", "dataset:beta"]

    def test_collection_missing_datasets_key_raises_error(self):
        collection_cfg = OmegaConf.create({"name": "test"})

        with pytest.raises(ValueError, match="must have a 'datasets' list"):
            create_observation_collection(collection_cfg, SimpleNamespace())

    def test_collection_keeps_dataset_and_settings_alignment(self, monkeypatch):
        system_of_bodies = SimpleNamespace()
        collection_cfg = OmegaConf.create(
            {
                "datasets": {
                    "first": {"type": "alpha", "file": "data1.csv", "weight": 1.0},
                    "second": {"type": "beta", "file": "data2.csv", "weight": 0.8},
                    "third": {"type": "gamma", "file": "data3.csv", "weight": 0.7},
                }
            }
        )

        created = [
            ("dataset:alpha", "settings:alpha"),
            (None, "settings:beta"),
            ("dataset:gamma", None),
        ]

        def fake_create_observation_dataset(cfg, dataset_cfg, system, *, frame=None):
            return created.pop(0)

        class FakeObservationCollection:
            def __init__(self, observation_sets):
                self.observation_sets = observation_sets

        class FakeMergedCollection:
            def get_single_observation_sets(self):
                return []

        merged_collection = FakeMergedCollection()
        merge_mock = MagicMock(return_value=merged_collection)

        monkeypatch.setattr(
            collection_module, "create_observation_dataset", fake_create_observation_dataset
        )
        monkeypatch.setattr(
            collection_module.obs, "ObservationCollection", FakeObservationCollection
        )
        monkeypatch.setattr(collection_module.obs, "merge_observation_collections", merge_mock)

        result = create_observation_collection(collection_cfg, system_of_bodies)

        assert result[0] is merged_collection
        assert result[1] == ["settings:alpha", "settings:beta", None]
        assert result[2] == {}  # dataset_metadata
        assert result[3].empty  # standard frame: no NSDB dataset in this config
        merge_mock.assert_called_once()
        merged_collections = merge_mock.call_args.args[0]
        assert len(merged_collections) == 1
        assert isinstance(merged_collections[0], FakeObservationCollection)
        assert merged_collections[0].observation_sets == ["dataset:alpha", None, "dataset:gamma"]

    def test_collection_not_dictconfig_raises_error(self):
        with pytest.raises(TypeError, match="Expected DictConfig"):
            create_observation_collection({"datasets": []}, SimpleNamespace())

    def test_collection_marks_each_row_with_the_set_it_built(self, monkeypatch):
        """Each frame row carries the final index of the set built from it."""
        system_of_bodies = SimpleNamespace()
        collection_cfg = OmegaConf.create(
            {"datasets": {"first": {"type": "alpha", "file": "data1.csv", "weight": 1.0}}}
        )
        frame = build_frame(
            [
                make_record("673", 1.0),
                make_record("673", 2.0),
                make_record("689", 3.0),
            ]
        )
        times = {
            "673": [1.0, 2.0],
            "689": [3.0],
        }

        def make_set(station_name, epochs):
            link_ends = {
                links.transmitter: links.body_origin_link_end_id("Triton"),
                links.receiver: links.body_reference_point_link_end_id("Earth", station_name),
            }
            return obs.create_single_observation_set(
                settings.model_settings.ObservableType.angular_position_type,
                link_ends,
                [np.array([[0.5], [0.25]])] * len(epochs),
                epochs,
                links.LinkEndType.receiver,
            )

        merged = obs.ObservationCollection(
            [make_set(name, epochs) for name, epochs in times.items()]
        )

        def fake_create_observation_dataset(cfg, dataset_cfg, system, *, frame=None):
            return merged, "settings:alpha"

        monkeypatch.setattr(
            collection_module, "create_observation_dataset", fake_create_observation_dataset
        )
        monkeypatch.setattr(collection_module, "standard_frame_for", lambda *a, **k: frame)

        result = create_observation_collection(collection_cfg, system_of_bodies)
        marked = result[3]

        assert set(marked[OBSERVATION_SET_INDEX_COLUMN].dropna()) == {0.0, 1.0}
        # Row order is preserved, so the frames' own epochs say which index is right.
        by_epoch = dict(zip(marked["epoch_tdb"], marked[OBSERVATION_SET_INDEX_COLUMN], strict=True))
        assert by_epoch[1.0] == 0.0
        assert by_epoch[2.0] == 0.0
        assert by_epoch[3.0] == 1.0
        # The same walk fills the metadata's own index, so the two agree.
        indices = {meta["index"] for entries in result[2].values() for meta in entries}
        assert indices == {0, 1}

    def test_collection_leaves_unbuildable_rows_unmarked(self, monkeypatch):
        """A row the factory dropped gets no index rather than a wrong one."""
        collection_cfg = OmegaConf.create(
            {"datasets": {"first": {"type": "alpha", "file": "data1.csv", "weight": 1.0}}}
        )
        frame = build_frame([make_record("673", 1.0), make_record("673", np.nan)])

        link_ends = {
            links.transmitter: links.body_origin_link_end_id("Triton"),
            links.receiver: links.body_reference_point_link_end_id("Earth", "673"),
        }
        built = obs.create_single_observation_set(
            settings.model_settings.ObservableType.angular_position_type,
            link_ends,
            [np.array([[0.5], [0.25]])],
            [1.0],
            links.LinkEndType.receiver,
        )
        merged = obs.ObservationCollection([built])

        def fake_create_observation_dataset(cfg, dataset_cfg, system, *, frame=None):
            return merged, "settings:alpha"

        monkeypatch.setattr(
            collection_module, "create_observation_dataset", fake_create_observation_dataset
        )
        monkeypatch.setattr(collection_module, "standard_frame_for", lambda *a, **k: frame)

        marked = create_observation_collection(collection_cfg, SimpleNamespace())[3]

        assert marked[OBSERVATION_SET_INDEX_COLUMN].tolist()[0] == 0.0
        assert pd.isna(marked[OBSERVATION_SET_INDEX_COLUMN].tolist()[1])
