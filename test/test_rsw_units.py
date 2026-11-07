"""Tests for the RSW length-unit support."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from orbitdet.visualization.RSW_distance import (
    DEFAULT_LENGTH_UNIT,
    LENGTH_UNITS,
    length_unit,
    to_length_unit,
)


def _cfg(unit: str | None = None):
    axes = {} if unit is None else {"unit": unit}
    return OmegaConf.create({"axes": axes})


# -- length_unit ------------------------------------------------------------


def test_default_unit_is_metres():
    assert DEFAULT_LENGTH_UNIT == "m"
    assert length_unit(_cfg()) == "m"


def test_unit_read_from_config():
    assert length_unit(_cfg("km")) == "km"


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown length unit 'cm'"):
        length_unit(_cfg("cm"))


def test_supported_units():
    assert set(LENGTH_UNITS) == {"m", "km"}


# -- to_length_unit ---------------------------------------------------------


def test_metres_are_unchanged():
    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(to_length_unit(values, "m"), values)


def test_metres_to_kilometres():
    values = np.array([1000.0, 2500.0, -500.0])
    np.testing.assert_allclose(to_length_unit(values, "km"), [1.0, 2.5, -0.5])


def test_conversion_preserves_shape():
    values = np.zeros((10, 3))
    assert to_length_unit(values, "km").shape == (10, 3)


def test_conversion_preserves_nan():
    values = np.array([np.nan, 1000.0])
    result = to_length_unit(values, "km")
    assert np.isnan(result[0])
    assert result[1] == 1.0


def test_conversion_accepts_a_list():
    np.testing.assert_allclose(to_length_unit([1000.0, 2000.0], "km"), [1.0, 2.0])
