"""Tests for the paper-size figure sizing in the plotter base class."""

import pytest
from omegaconf import OmegaConf

from orbitdet.visualization.base import (
    A4_HEIGHT_INCHES,
    A4_WIDTH_INCHES,
    DEFAULT_PAPER_SIZE,
    PAPER_SIZES,
    Plot,
    paper_size_inches,
)


class _DummyPlot(Plot):
    """Minimal concrete plotter, so the base class can be instantiated."""

    def _make_figure(self):  # pragma: no cover - never called
        raise NotImplementedError


def _plot(figure_cfg: dict | None = None) -> _DummyPlot:
    cfg = OmegaConf.create({"figure": figure_cfg} if figure_cfg is not None else {})
    return _DummyPlot(cfg)


# -- paper_size_inches ------------------------------------------------------


def test_paper_sizes_are_portrait_a4_portions():
    assert PAPER_SIZES["full"] == (A4_WIDTH_INCHES, A4_HEIGHT_INCHES)
    assert PAPER_SIZES["half_height"] == (A4_WIDTH_INCHES, A4_HEIGHT_INCHES / 2)
    assert PAPER_SIZES["half_width"] == (A4_WIDTH_INCHES / 2, A4_HEIGHT_INCHES)
    assert PAPER_SIZES["quarter_corner"] == (A4_WIDTH_INCHES / 2, A4_HEIGHT_INCHES / 2)


def test_paper_size_inches_returns_named_size():
    assert paper_size_inches("full") == (A4_WIDTH_INCHES, A4_HEIGHT_INCHES)


def test_paper_size_inches_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown paper size 'a5'"):
        paper_size_inches("a5")


# -- Plot._figure_size ------------------------------------------------------


def test_default_is_half_a4():
    assert DEFAULT_PAPER_SIZE == "half_height"
    assert _plot()._figure_size() == PAPER_SIZES["half_height"]


def test_paper_size_from_config_wins():
    plot = _plot({"paper_size": "quarter_corner"})
    assert plot._figure_size() == PAPER_SIZES["quarter_corner"]


def test_paper_size_beats_explicit_dimensions():
    plot = _plot({"paper_size": "full", "width": 3.0, "height": 4.0})
    assert plot._figure_size() == PAPER_SIZES["full"]


def test_explicit_dimensions_used_when_no_paper_size():
    plot = _plot({"width": 3.0, "height": 4.0})
    assert plot._figure_size() == (3.0, 4.0)


def test_partial_dimensions_fall_back_to_default_paper_size():
    plot = _plot({"width": 3.0})
    _, default_height = PAPER_SIZES[DEFAULT_PAPER_SIZE]
    assert plot._figure_size() == (3.0, default_height)


def test_panels_multiply_the_height():
    width, height = PAPER_SIZES[DEFAULT_PAPER_SIZE]
    assert _plot()._figure_size(panels=3) == (width, height * 3)


def test_panels_multiply_an_explicit_height():
    plot = _plot({"width": 3.0, "height": 4.0})
    assert plot._figure_size(panels=2) == (3.0, 8.0)


def test_panels_multiply_a_named_paper_size():
    width, height = PAPER_SIZES["quarter_corner"]
    plot = _plot({"paper_size": "quarter_corner"})
    assert plot._figure_size(panels=6) == (width, height * 6)


def test_cfg_argument_overrides_self_cfg():
    """The validation plotters nest their figure block under a key."""
    cfg = OmegaConf.create(
        {
            "figure": {"paper_size": "full"},
            "integrator_convergence": {"figure": {"paper_size": "quarter_corner"}},
        }
    )
    plot = _DummyPlot(cfg)
    assert plot._figure_size(cfg=cfg.integrator_convergence) == PAPER_SIZES["quarter_corner"]
    assert plot._figure_size() == PAPER_SIZES["full"]


def test_unknown_paper_size_in_config_raises():
    plot = _plot({"paper_size": "a5"})
    with pytest.raises(ValueError, match="Unknown paper size 'a5'"):
        plot._figure_size()
