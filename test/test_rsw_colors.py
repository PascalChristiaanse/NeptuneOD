"""Tests for the named colour palettes and the RSW colour collection."""

import pytest
from omegaconf import OmegaConf

from orbitdet.visualization.palettes import (
    DEFAULT_PALETTE,
    OKABE_ITO,
    PALETTES,
    palette,
)
from orbitdet.visualization.RSW_distance import (
    DEFAULT_COLORS,
    color_collection,
    color_for,
)


def _cfg(**styling):
    return OmegaConf.create({"styling": styling})


# -- palette ----------------------------------------------------------------


def test_default_palette_is_black():
    assert DEFAULT_PALETTE == "black"
    assert palette() == ["#000000"]


def test_okabe_ito_starts_with_the_canonical_pair():
    assert OKABE_ITO[0] == "#0072B2"
    assert OKABE_ITO[1] == "#D55E00"


def test_palette_returns_a_copy():
    """Mutating the result must not corrupt the module-level table."""
    colors = palette("okabe_ito")
    colors.append("#123456")
    assert "#123456" not in PALETTES["okabe_ito"]


def test_unknown_palette_raises():
    with pytest.raises(ValueError, match="Unknown palette 'viridis'"):
        palette("viridis")


def test_every_palette_has_at_least_one_colour():
    for name, colors in PALETTES.items():
        assert len(colors) >= 1, f"{name} is empty"


def test_multi_series_palettes_have_at_least_two_colours():
    """``black`` is deliberately a single colour; the rest must distinguish series."""
    for name, colors in PALETTES.items():
        if name == "black":
            continue
        assert len(colors) >= 2, f"{name} has fewer than two colours"


def test_palette_names_are_listed_in_the_error():
    with pytest.raises(ValueError, match="okabe_ito"):
        palette("nope")


# -- color_collection -------------------------------------------------------


def test_default_is_black():
    assert color_collection(_cfg()) == DEFAULT_COLORS


def test_palette_name_is_resolved():
    assert color_collection(_cfg(palette="okabe_ito")) == OKABE_ITO


def test_palette_takes_precedence_over_colors():
    cfg = _cfg(palette="okabe_ito", colors=["#123456"])
    assert color_collection(cfg) == OKABE_ITO


def test_explicit_colors_still_work():
    assert color_collection(_cfg(colors=["#123456", "#abcdef"])) == ["#123456", "#abcdef"]


def test_a_bare_string_is_a_one_colour_collection():
    assert color_collection(_cfg(colors="tab:blue")) == ["tab:blue"]


def test_empty_colors_raises():
    with pytest.raises(ValueError, match="at least one colour"):
        color_collection(_cfg(colors=[]))


def test_unknown_palette_in_config_raises():
    with pytest.raises(ValueError, match="Unknown palette"):
        color_collection(_cfg(palette="nope"))


# -- color_for --------------------------------------------------------------


def test_color_for_indexes_in_order():
    colors = ["a", "b", "c"]
    assert [color_for(colors, i) for i in range(3)] == ["a", "b", "c"]


def test_color_for_cycles_past_the_end():
    colors = ["a", "b"]
    assert color_for(colors, 2) == "a"
    assert color_for(colors, 3) == "b"


def test_single_colour_applies_to_every_series():
    assert {color_for(["black"], i) for i in range(5)} == {"black"}
