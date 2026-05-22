"""Tests for the observation dataset registry and factory system."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

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
from orbitdet.observations.registry import _FACTORY_REGISTRY


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

        with pytest.raises(ValueError, match="must have a 'type' field"):
            create_observation_dataset(cfg, dataset_cfg, SimpleNamespace())

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

        def fake_create_observation_dataset(cfg, dataset_cfg, system):
            created.append((dataset_cfg.type, system))
            return f"dataset:{dataset_cfg.type}", f"settings:{dataset_cfg.type}"

        collection_mock = MagicMock(return_value="observation-collection")

        monkeypatch.setattr(
            collection_module, "create_observation_dataset", fake_create_observation_dataset
        )
        monkeypatch.setattr(collection_module.obs, "ObservationCollection", collection_mock)

        result = create_observation_collection(collection_cfg, system_of_bodies)

        assert result == ("observation-collection", ["settings:alpha", "settings:beta"])
        assert created == [("alpha", system_of_bodies), ("beta", system_of_bodies)]
        collection_mock.assert_called_once_with(["dataset:alpha", "dataset:beta"])

    def test_collection_missing_datasets_key_raises_error(self):
        collection_cfg = OmegaConf.create({"name": "test"})

        with pytest.raises(ValueError, match="must have a 'datasets' list"):
            create_observation_collection(collection_cfg, SimpleNamespace())

    def test_collection_not_dictconfig_raises_error(self):
        with pytest.raises(TypeError, match="Expected DictConfig"):
            create_observation_collection({"datasets": []}, SimpleNamespace())
