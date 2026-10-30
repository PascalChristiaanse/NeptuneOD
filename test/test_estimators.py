"""Tests for the estimatable-parameter construction helpers.

These tests exercise the config<->TudatPy mapping without building a real
:class:`~tudatpy.dynamics.environment.SystemOfBodies` (which would require SPICE
kernels). The parameter *settings* builders used by
:func:`get_additional_parameter_identifiers` need no bodies, so real identifiers
are produced; a lightweight :class:`FakeParameterSet` stands in for the
consolidated set.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf
from tudatpy.dynamics import parameters_setup as param_setup

from orbitdet.estimation.estimators import (
    ParameterBlock,
    apply_configured_parameter_values,
    get_additional_parameter_identifiers,
    get_parameter_block_layout,
    iter_parameter_entries,
)

_STATE_ID = (
    param_setup.EstimatableParameterTypes.initial_body_state_type,
    ("Triton", ""),
)


class FakeParameter:
    """Minimal stand-in for an ``EstimatableParameter`` block."""

    def __init__(self, size: int, value=None):
        self._size = int(size)
        if value is None:
            value = np.zeros(self._size)
        self._value = np.asarray(value, dtype=float).reshape(-1, 1)

    @property
    def parameter_size(self) -> int:
        return self._size

    def get_parameter_value(self) -> np.ndarray:
        return self._value

    def set_parameter_value(self, value) -> None:
        value = np.asarray(value, dtype=float).reshape(-1, 1)
        if value.size != self._size:
            raise ValueError(f"expected {self._size} values, got {value.size}")
        self._value = value


class FakeParameterSet:
    """Fake ``EstimatableParameterSet`` exposing the API the layout builder uses.

    Parameters
    ----------
    blocks : list[tuple[identifier, int]]
        ``(identifier, size)`` pairs in parameter-vector order. Start indices are
        assigned cumulatively.
    """

    def __init__(self, blocks):
        self._blocks = []
        start = 0
        for identifier, size in blocks:
            value = np.arange(start, start + size, dtype=float)
            self._blocks.append((identifier, start, size, FakeParameter(size, value)))
            start += size
        self.parameter_set_size = start

    def get_parameter_identifiers(self):
        return [identifier for identifier, _, _, _ in self._blocks]

    def indices_for_parameter_type(self, identifier):
        return [(start, size) for ident, start, size, _ in self._blocks if ident == identifier]

    def parameters_for_parameter_type(self, identifier):
        return [param for ident, _, _, param in self._blocks if ident == identifier]


def _cfg(parameters: dict, apriori: dict | None = None):
    estimation = {"parameters_to_estimate": parameters}
    if apriori is not None:
        estimation["apriori"] = apriori
    return OmegaConf.create({"bodies_to_propagate": {"Triton": {}}, "estimation": estimation})


def _cfg_with_state(**additional):
    parameters = {"initial_state": True}
    parameters.update(additional)
    return _cfg(parameters)


def _identifiers(cfg) -> dict[str, tuple]:
    return get_additional_parameter_identifiers(cfg)


class TestIterParameterEntries:
    def test_plain_and_mapping_forms(self):
        cfg = _cfg(
            {
                "initial_state": True,
                "neptune_GM": False,
                "triton_GM": {"value": 1.0},
                "iau_rotation_model_pole": {"apriori": [1e-6, 1e-6]},
                "neptune_j2_j4": {"enabled": False},
            }
        )
        entries = dict(iter_parameter_entries(cfg))
        assert entries == {
            "initial_state": True,
            "neptune_GM": False,
            "triton_GM": True,
            "iau_rotation_model_pole": True,
            "neptune_j2_j4": False,
        }


class TestGetAdditionalParameterIdentifiers:
    def test_only_enabled_non_state_keys(self):
        cfg = _cfg(
            {
                "initial_state": True,
                "neptune_GM": True,
                "triton_GM": False,
                "iau_rotation_model_pole": True,
            }
        )
        identifiers = _identifiers(cfg)
        assert set(identifiers) == {"neptune_GM", "iau_rotation_model_pole"}
        assert "initial_state" not in identifiers

    def test_returns_hashable_identifier_tuples(self):
        cfg = _cfg_with_state(neptune_GM=True)
        identifier = _identifiers(cfg)["neptune_GM"]
        assert isinstance(identifier, tuple)
        assert {identifier: "ok"}[identifier] == "ok"


class TestParameterBlockLayout:
    def test_reorders_to_match_parameter_set(self):
        # Tudat orders state first, then scalars, then vectors — regardless of
        # the order in the config (config lists 1/Q before GM here).
        cfg = _cfg_with_state(
            neptune_inverse_tidal_quality_factor=True,
            neptune_GM=True,
        )
        ids = _identifiers(cfg)
        fake = FakeParameterSet(
            [
                (_STATE_ID, 6),
                (ids["neptune_GM"], 1),
                (ids["neptune_inverse_tidal_quality_factor"], 1),
            ]
        )
        layout = get_parameter_block_layout(cfg, fake)
        assert layout == [
            ParameterBlock(key="initial_state", start=0, size=6),
            ParameterBlock(key="neptune_GM", start=6, size=1),
            ParameterBlock(key="neptune_inverse_tidal_quality_factor", start=7, size=1),
        ]

    def test_sizes_come_from_parameter_set_not_config(self):
        # The libration block is 4 scalars (2 amplitudes x 2 frequencies); the
        # layout must report whatever the set says, never a hard-coded size.
        cfg = _cfg_with_state(iau_rotation_model_pole_librations=True)
        ids = _identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["iau_rotation_model_pole_librations"], 4)])
        layout = get_parameter_block_layout(cfg, fake)
        assert layout[1].size == 4

    def test_unmapped_identifier_raises(self):
        cfg = _cfg_with_state(neptune_GM=True)
        ids = _identifiers(cfg)
        unknown_id = _identifiers(_cfg_with_state(triton_GM=True))["triton_GM"]
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_GM"], 1)])
        # Add a block whose identifier no configured key maps to.
        fake._blocks.append((unknown_id, 7, 7, FakeParameter(1)))
        with pytest.raises(ValueError, match="no configured parameter key maps to"):
            get_parameter_block_layout(cfg, fake)


class TestApplyConfiguredParameterValues:
    def test_sets_scalar_parameter(self):
        cfg = _cfg(
            {
                "initial_state": True,
                "neptune_GM": {"value": 6.9e6},
            }
        )
        ids = _identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_GM"], 1)])
        applied = apply_configured_parameter_values(cfg, fake)
        assert applied == ["neptune_GM"]
        gm = fake.parameters_for_parameter_type(ids["neptune_GM"])[0]
        np.testing.assert_allclose(gm.get_parameter_value(), [[6.9e6]])

    def test_sets_vector_parameter_with_list(self):
        cfg = _cfg(
            {
                "initial_state": True,
                "neptune_j2_j4": {"value": [1e-4, 2e-5]},
            }
        )
        ids = _identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_j2_j4"], 2)])
        apply_configured_parameter_values(cfg, fake)
        sh = fake.parameters_for_parameter_type(ids["neptune_j2_j4"])[0]
        np.testing.assert_allclose(sh.get_parameter_value().ravel(), [1e-4, 2e-5])

    def test_sets_initial_state(self):
        cfg = _cfg({"initial_state": {"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}})
        fake = FakeParameterSet([(_STATE_ID, 6)])
        applied = apply_configured_parameter_values(cfg, fake)
        assert applied == ["initial_state"]
        state = fake.parameters_for_parameter_type(_STATE_ID)[0]
        np.testing.assert_allclose(state.get_parameter_value().ravel(), [1, 2, 3, 4, 5, 6])

    def test_no_values_is_noop(self):
        cfg = _cfg_with_state(neptune_GM=True)
        fake = FakeParameterSet([(_STATE_ID, 6), (_identifiers(cfg)["neptune_GM"], 1)])
        assert apply_configured_parameter_values(cfg, fake) == []

    def test_wrong_size_raises(self):
        cfg = _cfg({"initial_state": True, "neptune_j2_j4": {"value": [1e-4, 2e-5, 3e-6]}})
        ids = _identifiers(cfg)
        fake = FakeParameterSet([(_STATE_ID, 6), (ids["neptune_j2_j4"], 2)])
        with pytest.raises(ValueError, match="has 3 entries but its parameter block has size 2"):
            apply_configured_parameter_values(cfg, fake)

    def test_value_for_unrequested_parameter_raises(self):
        cfg = _cfg({"initial_state": True, "triton_GM": {"value": 1.0}})
        # Only a state block present; triton_GM value has no matching enabled key.
        fake = FakeParameterSet([(_STATE_ID, 6)])
        with pytest.raises(ValueError, match="not an enabled additional parameter|Expected"):
            apply_configured_parameter_values(cfg, fake)
