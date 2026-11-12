"""Tests for the :class:`CovarianceCorrelationPairGrid` corner plot.

The plotter draws three views of one covariance matrix: standard deviations on
the diagonal, covariance ellipses below it and correlation coefficients above
it.  The tests check that each region is populated from the right matrix, and
that the correlations are recomputed from the *plotted* covariance so the upper
triangle always agrees with the ellipses below it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from orbitdet.visualization import CovarianceCorrelationPairGrid  # noqa: E402
from orbitdet.visualization.covariance_correlation_pairgrid import (  # noqa: E402
    _correlations_from_covariance,
)


class _FakeEstimationOutput:
    """Minimal stand-in carrying only the covariance the plotter reads."""

    def __init__(self, covariance: np.ndarray):
        self.covariance = covariance


class _FakeContext:
    """Minimal stand-in for the runtime context (only ``start_epoch`` is read)."""

    start_epoch = 0.0


def _covariance(n: int = 4, seed: int = 0) -> np.ndarray:
    """A symmetric positive-definite covariance with non-trivial correlations."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T * 1e-8 + np.eye(n) * 1e-10


def _cfg(**overrides):
    base = {
        "rotate_to_rsw": False,
        "parameter_names": ["a", "b", "c", "d"],
        "n_sigma": 1.0,
        "annotation": True,
        "decimal_places": 2,
        "cmap": "coolwarm",
        "figure": {"width": 6.0, "height": 6.0},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _plot(covariance=None, **overrides):
    covariance = _covariance() if covariance is None else covariance
    plot = CovarianceCorrelationPairGrid(
        _cfg(**overrides), _FakeEstimationOutput(covariance), None, _FakeContext()
    )
    fig, axes = plot._make_figure()
    return fig, axes, covariance


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -- correlation helper -----------------------------------------------------


def test_correlations_have_unit_diagonal():
    correlations = _correlations_from_covariance(_covariance())

    assert np.allclose(np.diag(correlations), 1.0)


def test_correlations_are_symmetric_and_bounded():
    correlations = _correlations_from_covariance(_covariance())

    assert np.allclose(correlations, correlations.T)
    assert np.all(np.abs(correlations) <= 1.0)


def test_correlations_match_manual_normalization():
    covariance = _covariance()
    sigma = np.sqrt(np.diag(covariance))

    correlations = _correlations_from_covariance(covariance)

    assert correlations[0, 1] == pytest.approx(covariance[0, 1] / (sigma[0] * sigma[1]))


def test_zero_variance_parameter_yields_zero_correlation():
    """A degenerate parameter must not produce ``nan`` and break the heatmap."""
    covariance = np.eye(2)
    covariance[1, 1] = 0.0

    correlations = _correlations_from_covariance(covariance)

    assert np.all(np.isfinite(correlations))
    assert correlations[0, 1] == 0.0


# -- figure layout ----------------------------------------------------------


def test_grid_is_square_with_one_cell_per_parameter():
    _, axes, covariance = _plot()

    assert axes.shape == (covariance.shape[0], covariance.shape[0])


def test_diagonal_draws_one_standard_deviation_bar_per_parameter():
    _, axes, covariance = _plot()
    sigma = np.sqrt(np.diag(covariance))

    for i in range(covariance.shape[0]):
        bars = axes[i, i].patches
        assert len(bars) == 1
        assert bars[0].get_height() == pytest.approx(sigma[i])


def test_lower_triangle_draws_one_ellipse_per_pair():
    _, axes, covariance = _plot()
    n = covariance.shape[0]

    for i in range(n):
        for j in range(i):
            ellipses = [p for p in axes[i, j].patches if isinstance(p, Ellipse)]
            assert len(ellipses) == 1, f"cell ({i}, {j}) should hold one ellipse"


def test_upper_triangle_draws_one_correlation_cell_per_pair():
    _, axes, covariance = _plot()
    n = covariance.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            rectangles = [p for p in axes[i, j].patches if isinstance(p, Rectangle)]
            assert len(rectangles) == 1, f"cell ({i}, {j}) should hold one rectangle"


def test_upper_triangle_annotates_the_correlation_value():
    _, axes, covariance = _plot()
    correlations = _correlations_from_covariance(covariance)

    texts = [t.get_text() for t in axes[0, 1].texts]

    assert texts == [f"{correlations[0, 1]:.2f}"]


def test_annotation_can_be_disabled():
    _, axes, _ = _plot(annotation=False)

    assert len(axes[0, 1].texts) == 0


def test_decimal_places_controls_the_annotation():
    _, axes, covariance = _plot(decimal_places=4)
    correlations = _correlations_from_covariance(covariance)

    assert axes[0, 1].texts[0].get_text() == f"{correlations[0, 1]:.4f}"


def test_ellipse_axes_scale_with_n_sigma():
    """Doubling ``n_sigma`` must double the drawn ellipse, not just its label."""
    _, axes_one, _ = _plot(n_sigma=1.0)
    width_one = axes_one[1, 0].patches[0].width

    _, axes_two, _ = _plot(n_sigma=2.0)
    width_two = axes_two[1, 0].patches[0].width

    assert width_two == pytest.approx(2.0 * width_one)


def test_ellipse_orientation_follows_the_covariance_sign():
    """A positive covariance tilts the ellipse up-right, a negative one down-right.

    ``Ellipse.angle`` is reported modulo 180°, so the negative case comes back as
    ``135°`` rather than ``-45°``; both describe the same orientation.
    """
    positive = np.array([[1.0, 0.8], [0.8, 1.0]])
    negative = np.array([[1.0, -0.8], [-0.8, 1.0]])

    _, axes_pos, _ = _plot(positive, parameter_names=["a", "b"])
    _, axes_neg, _ = _plot(negative, parameter_names=["a", "b"])

    assert axes_pos[1, 0].patches[0].angle == pytest.approx(45.0)
    assert axes_neg[1, 0].patches[0].angle % 180.0 == pytest.approx(135.0)


def test_parameter_names_label_the_outer_edges_only():
    _, axes, _ = _plot()

    # Bottom row carries the x labels, left column the y labels.
    assert axes[-1, 0].get_xlabel() == "a"
    assert axes[-1, 1].get_xlabel() == "b"
    assert axes[0, 0].get_ylabel() == "a"
    assert axes[1, 0].get_ylabel() == "b"
    # Interior cells stay unlabelled.
    assert axes[0, 1].get_xlabel() == ""
    assert axes[0, 1].get_ylabel() == ""


def test_cells_do_not_share_axis_limits():
    """Each cell has its own units, so a shared axis would hide content."""
    _, axes, _ = _plot()

    axes[0, 0].set_xlim(-5.0, 5.0)

    assert axes[0, 1].get_xlim() != (-5.0, 5.0)


def test_title_names_the_frame():
    fig, _, _ = _plot()

    assert "Inertial frame" in fig._suptitle.get_text()


def test_rotated_title_names_the_rsw_frame(monkeypatch):
    """The RSW path is exercised with the frame helpers stubbed out."""
    import orbitdet.visualization.covariance_correlation_pairgrid as module

    monkeypatch.setattr(module, "_get_inertial_state_at_epoch", lambda *a, **k: np.ones(6))
    monkeypatch.setattr(module, "_rotate_covariance_to_rsw", lambda cov, state: cov)

    fig, _, _ = _plot(rotate_to_rsw=True)

    assert "RSW frame" in fig._suptitle.get_text()


def test_rotation_is_applied_before_the_correlations_are_computed(monkeypatch):
    """The upper triangle must describe the rotated matrix, not the inertial one."""
    import orbitdet.visualization.covariance_correlation_pairgrid as module

    covariance = _covariance()
    rotated = np.diag(np.diag(covariance))  # rotation that removes all correlation

    monkeypatch.setattr(module, "_get_inertial_state_at_epoch", lambda *a, **k: np.ones(6))
    monkeypatch.setattr(module, "_rotate_covariance_to_rsw", lambda cov, state: rotated)

    _, axes, _ = _plot(covariance, rotate_to_rsw=True)

    assert axes[0, 1].texts[0].get_text() == "0.00"
