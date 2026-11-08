"""Tests for the journal figure styling."""

import matplotlib
import pytest

from orbitdet.visualization.style import (
    COLOR_CYCLE,
    DEFAULT_STYLE,
    apply_style,
    available_styles,
)


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Undo the style after each test so the suite is order-independent."""
    saved = dict(matplotlib.rcParams)
    yield
    matplotlib.rcParams.update(saved)


def test_default_style_is_a_and_a():
    assert DEFAULT_STYLE == "A&A"


def test_available_styles_includes_the_journals():
    styles = available_styles()
    assert "A&A" in styles
    assert "MNRAS" in styles


def test_apply_style_returns_the_name():
    assert apply_style("A&A") == "A&A"


def test_apply_style_defaults_to_the_default_style():
    assert apply_style() == DEFAULT_STYLE


def test_apply_style_sets_journal_fonts():
    apply_style("A&A")
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["font.size"] == 9.0


def test_apply_style_restores_a_multi_colour_cycle():
    """funkyfresh sets a single black; that would flatten multi-series figures."""
    apply_style("A&A")
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert len(colors) > 1
    assert colors == list(matplotlib.colormaps[COLOR_CYCLE].colors)


def test_apply_style_can_keep_the_journal_colour_cycle():
    apply_style("A&A", keep_color_cycle=False)
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert colors == ["k"]


def test_apply_style_rejects_unknown_style():
    with pytest.raises(ValueError, match="Unknown journal style 'Nature'"):
        apply_style("Nature")


def test_apply_style_accepts_a_fontsize_override():
    apply_style("A&A", fontsize=12)
    assert matplotlib.rcParams["font.size"] == 12.0
