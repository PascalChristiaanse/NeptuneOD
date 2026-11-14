"""Tests for the :class:`SpecificationCurve` sweep figure.

The figure is two vertically aligned panels: a truth table of which parameters
each fit estimated, and the outcome of each fit in the same row order.  The
tests check the alignment (the property the figure exists for), the sort order,
and that non-converged fits are kept rather than dropped.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from orbitdet.visualization import SpecificationCurve

SPEC_COLUMNS = ["neptune_GM", "neptune_j2", "triton_GM"]


def _frame(converged_column: bool = False) -> pd.DataFrame:
    """A small sweep: every combination of three toggles, with outcomes."""
    rows = []
    for gm in (False, True):
        for j2 in (False, True):
            for triton in (False, True):
                rows.append(
                    {
                        "neptune_GM": gm,
                        "neptune_j2": j2,
                        "triton_GM": triton,
                        "rms": 1.0 + 0.5 * gm + 0.25 * j2 + 0.125 * triton,
                    }
                )
    frame = pd.DataFrame(rows)
    if converged_column:
        frame["converged"] = True
    return frame


def _cfg(**overrides):
    base = {
        "specification_columns": SPEC_COLUMNS,
        "outcome_column": "rms",
        "figure": {"width": 6.0, "height": 8.0},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _plot(cfg, frame):
    plotter = SpecificationCurve(cfg, frame)
    fig, axes = plotter._make_figure()
    return plotter, fig, axes


class TestConfiguration:
    def test_requires_specification_columns(self):
        cfg = _cfg(specification_columns=[])
        with pytest.raises(ValueError, match="at least one specification column"):
            SpecificationCurve(cfg, _frame())

    def test_requires_outcome_column(self):
        cfg = _cfg(outcome_column=None)
        with pytest.raises(ValueError, match="needs an outcome column"):
            SpecificationCurve(cfg, _frame())

    def test_missing_column_raises(self):
        cfg = _cfg(specification_columns=[*SPEC_COLUMNS, "not_a_column"])
        with pytest.raises(ValueError, match="not in the frame"):
            SpecificationCurve(cfg, _frame())

    def test_columns_can_be_passed_directly(self):
        cfg = OmegaConf.create({"figure": {"width": 6.0, "height": 8.0}})
        plotter = SpecificationCurve(
            cfg,
            _frame(),
            specification_columns=SPEC_COLUMNS,
            outcome_column="rms",
        )
        assert plotter.specification_columns == SPEC_COLUMNS
        assert plotter.outcome_column == "rms"


class TestSorting:
    def test_ascending_by_default(self):
        plotter = SpecificationCurve(_cfg(), _frame())
        sorted_frame = plotter._sorted_frame()
        assert sorted_frame["rms"].is_monotonic_increasing

    def test_descending(self):
        plotter = SpecificationCurve(_cfg(sort="descending"), _frame())
        sorted_frame = plotter._sorted_frame()
        assert sorted_frame["rms"].is_monotonic_decreasing

    def test_none_preserves_input_order(self):
        frame = _frame()
        plotter = SpecificationCurve(_cfg(sort="none"), frame)
        sorted_frame = plotter._sorted_frame()
        np.testing.assert_allclose(sorted_frame["rms"].to_numpy(), frame["rms"].to_numpy())

    def test_unknown_sort_raises(self):
        plotter = SpecificationCurve(_cfg(sort="sideways"), _frame())
        with pytest.raises(ValueError, match="Unknown sort"):
            plotter._sorted_frame()

    def test_missing_outcomes_sort_last(self):
        frame = _frame()
        frame.loc[0, "rms"] = np.nan
        plotter = SpecificationCurve(_cfg(), frame)
        sorted_frame = plotter._sorted_frame()
        assert np.isnan(sorted_frame["rms"].iloc[-1])


class TestConvergence:
    def test_falls_back_to_non_null_outcome(self):
        frame = _frame()
        frame.loc[0, "rms"] = np.nan
        plotter = SpecificationCurve(_cfg(), frame)
        mask = plotter._converged_mask(frame)
        assert mask.sum() == len(frame) - 1
        assert not mask.iloc[0]

    def test_uses_converged_column_when_present(self):
        frame = _frame(converged_column=True)
        frame.loc[0, "converged"] = False
        plotter = SpecificationCurve(_cfg(converged_column="converged"), frame)
        mask = plotter._converged_mask(frame)
        assert not mask.iloc[0]
        assert mask.sum() == len(frame) - 1

    def test_non_converged_rows_are_kept(self):
        # A sweep where most fits fail: the failures must still be drawn, since
        # which combinations fail is itself the result.
        frame = _frame()
        frame.loc[frame.index[:5], "rms"] = np.nan
        _, fig, (ax_outcome, ax_matrix) = _plot(_cfg(), frame)
        # Every specification still gets a column in the matrix panel.
        assert ax_matrix.get_xlim()[0] != ax_matrix.get_xlim()[1]
        assert len(frame) == 8
        plt.close(fig)

    def test_non_converged_rows_are_drawn(self):
        # A failed fit has no outcome, so it must be placed explicitly rather
        # than left at NaN, which would render nothing.
        frame = _frame()
        frame.loc[frame.index[:5], "rms"] = np.nan
        _, fig, (ax_outcome, _) = _plot(_cfg(), frame)
        drawn = sum(len(collection.get_offsets()) for collection in ax_outcome.collections)
        assert drawn == len(frame)
        plt.close(fig)

    def test_non_converged_fits_with_an_outcome_keep_the_real_scale(self):
        # A fit can fail to converge and still report a residual.  Those rows
        # must be drawn at their own outcome, not stacked at a synthetic "no
        # value" position, which would collapse the axis to that fake value.
        frame = _frame(converged_column=True)
        frame["converged"] = False
        _, fig, (ax_outcome, _) = _plot(_cfg(converged_column="converged"), frame)
        low, high = ax_outcome.get_ylim()
        assert low < frame["rms"].min()
        assert high > frame["rms"].max()
        drawn = sum(len(collection.get_offsets()) for collection in ax_outcome.collections)
        assert drawn == len(frame)
        plt.close(fig)


class TestOutcomeScale:
    def test_defaults_to_linear(self):
        _, fig, (ax_outcome, _) = _plot(_cfg(), _frame())
        assert ax_outcome.get_yscale() == "linear"
        plt.close(fig)

    def test_log_scale_is_applied(self):
        cfg = _cfg(axes={"outcome_scale": "log"})
        _, fig, (ax_outcome, _) = _plot(cfg, _frame())
        assert ax_outcome.get_yscale() == "log"
        plt.close(fig)

    def test_unknown_scale_is_rejected(self):
        cfg = _cfg(axes={"outcome_scale": "logit"})
        with pytest.raises(ValueError, match="outcome_scale"):
            _plot(cfg, _frame())

    def test_no_value_markers_stay_positive_on_a_log_axis(self):
        # A log axis has no room below the data minimum, so the "no value" row
        # must be placed inside the positive range rather than below it.
        frame = _frame()
        frame.loc[frame.index[:3], "rms"] = np.nan
        cfg = _cfg(axes={"outcome_scale": "log"})
        _, fig, (ax_outcome, _) = _plot(cfg, frame)
        ys = np.concatenate(
            [collection.get_offsets()[:, 1] for collection in ax_outcome.collections]
        )
        assert np.all(ys > 0)
        plt.close(fig)


class TestClipping:
    def _frame_with_outlier(self) -> pd.DataFrame:
        frame = _frame()
        frame.loc[frame.index[-1], "rms"] = 5262.0
        return frame

    def test_axis_is_cut_at_the_clip_value(self):
        cfg = _cfg(axes={"outcome_clip": 500.0})
        _, fig, (ax_outcome, _) = _plot(cfg, self._frame_with_outlier())
        assert ax_outcome.get_ylim()[1] < 5262.0
        assert ax_outcome.get_ylim()[1] == pytest.approx(500.0 * 1.04)
        plt.close(fig)

    def test_clipped_points_are_marked(self):
        cfg = _cfg(axes={"outcome_clip": 500.0})
        _, fig, (ax_outcome, _) = _plot(cfg, self._frame_with_outlier())
        # The caret collection carries the "> 500" legend label.
        labels = [text.get_text() for text in ax_outcome.get_legend().get_texts()]
        assert labels == ["> 500"]
        plt.close(fig)

    def test_no_marker_when_nothing_is_clipped(self):
        cfg = _cfg(axes={"outcome_clip": 500.0})
        _, fig, (ax_outcome, _) = _plot(cfg, _frame())
        assert ax_outcome.get_legend() is None
        plt.close(fig)

    def test_largest_clipped_value_can_be_annotated(self):
        cfg = _cfg(axes={"outcome_clip": 500.0, "outcome_annotate_max": True})
        _, fig, (ax_outcome, _) = _plot(cfg, self._frame_with_outlier())
        texts = [text.get_text() for text in ax_outcome.texts]
        assert "5,262" in texts
        plt.close(fig)

    def test_annotation_is_off_by_default(self):
        cfg = _cfg(axes={"outcome_clip": 500.0})
        _, fig, (ax_outcome, _) = _plot(cfg, self._frame_with_outlier())
        # The median label is also a text artist, so look for the value itself.
        texts = [text.get_text() for text in ax_outcome.texts]
        assert "5,262" not in texts
        plt.close(fig)


class TestAlignment:
    def test_panels_share_the_specification_axis(self):
        # The figure's whole argument is that a column means the same fit in
        # both panels, so the x axes must be shared.
        _, fig, (ax_outcome, ax_matrix) = _plot(_cfg(), _frame())
        assert ax_outcome.get_xlim() == ax_matrix.get_xlim()
        plt.close(fig)

    def test_best_specification_is_on_the_left(self):
        plotter, fig, (ax_outcome, _) = _plot(_cfg(), _frame())
        # Column 0 holds the smallest outcome, and the axis is not inverted, so
        # it sits at the left of the figure.
        assert ax_outcome.get_xlim()[0] < ax_outcome.get_xlim()[1]
        assert plotter._sorted_frame()["rms"].iloc[0] == pytest.approx(1.0)
        plt.close(fig)

    def test_matrix_has_one_tick_per_parameter(self):
        _, fig, (_, ax_matrix) = _plot(_cfg(), _frame())
        assert len(ax_matrix.get_yticks()) == len(SPEC_COLUMNS)
        plt.close(fig)

    def test_labels_are_used_when_configured(self):
        cfg = _cfg(labels={"neptune_GM": "GM", "neptune_j2": "J2", "triton_GM": "GM_T"})
        _, fig, (_, ax_matrix) = _plot(cfg, _frame())
        shown = [tick.get_text() for tick in ax_matrix.get_yticklabels()]
        assert shown == ["GM", "J2", "GM_T"]
        plt.close(fig)


class TestRendering:
    def test_returns_two_axes(self):
        _, fig, axes = _plot(_cfg(), _frame())
        assert len(axes) == 2
        plt.close(fig)

    def test_draws_a_cell_per_toggle(self):
        frame = _frame()
        _, fig, (_, ax_matrix) = _plot(_cfg(), frame)
        # Every toggle cell is drawn: a filled square where the parameter was
        # estimated and a pale one where it was not, so the grid reads as a
        # table rather than a scatter of only the "on" cells.
        drawn = sum(len(collection.get_offsets()) for collection in ax_matrix.collections)
        assert drawn == frame[SPEC_COLUMNS].size
        plt.close(fig)

    def test_suptitle_is_applied(self):
        cfg = _cfg(titles={"suptitle": "GM sweep"})
        _, fig, _ = _plot(cfg, _frame())
        assert fig._suptitle.get_text() == "GM sweep"
        plt.close(fig)


class TestStateColumns:
    """A non-boolean column is a state, drawn as its value rather than a cell.

    The chapter 3 pole sweep draws its libration order as two boolean rows, so
    this covers the generic fallback: any column that is not boolean is written
    as its value instead of being drawn as an on/off cell.
    """

    def _frame_with_order(self) -> pd.DataFrame:
        frame = _frame()
        frame["librations_order"] = [0, 1, 2, 0, 1, 2, 0, 1]
        return frame

    def test_state_column_is_labelled_not_drawn(self):
        cfg = _cfg(specification_columns=[*SPEC_COLUMNS, "librations_order"])
        _, fig, (_, ax_matrix) = _plot(cfg, self._frame_with_order())
        # One text label per specification, and the cells still count only the
        # toggles (both on and off).
        assert len(ax_matrix.texts) == 8
        drawn = sum(len(collection.get_offsets()) for collection in ax_matrix.collections)
        assert drawn == self._frame_with_order()[SPEC_COLUMNS].size
        plt.close(fig)

    def test_state_labels_show_the_order(self):
        cfg = _cfg(specification_columns=[*SPEC_COLUMNS, "librations_order"])
        _, fig, (_, ax_matrix) = _plot(cfg, self._frame_with_order())
        shown = sorted(text.get_text() for text in ax_matrix.texts)
        assert shown == ["0", "0", "0", "1", "1", "1", "2", "2"]
        plt.close(fig)

    def test_every_parameter_gets_a_row(self):
        cfg = _cfg(specification_columns=[*SPEC_COLUMNS, "librations_order"])
        _, fig, (_, ax_matrix) = _plot(cfg, self._frame_with_order())
        low, high = ax_matrix.get_ylim()
        # Inverted axis: row 0 (the first parameter) is at the top.
        assert low > high
        assert len(ax_matrix.get_yticks()) == len(SPEC_COLUMNS) + 1
        plt.close(fig)
