"""Tests for the estimation parameter frame and the configured table view.

The producer derives one wide dataframe from an estimation output; a table
config then selects columns from it.  The tests use a stand-in estimation output
carrying only the attributes the producer reads, so they exercise the arithmetic
without a real fit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from orbitdet.analysis.tables import EstimationParameterTable, TableView
from orbitdet.analysis.tables.parameter_table import (
    COLUMNS,
    _change_factor,
    _initial_parameters,
    _relative_change_percent,
    _relative_formal_error_percent,
    _shift_sigma,
    build_parameter_frame,
)


class _FakeEstimationOutput:
    """Minimal stand-in carrying the attributes the producer reads."""

    def __init__(
        self,
        history: np.ndarray,
        final: np.ndarray,
        formal_errors: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ):
        self.parameter_history = history
        self.final_parameters = final
        self.formal_errors = formal_errors
        self.covariance = covariance


def _output(
    initial=(1.0, 2.0, 4.0),
    final=(1.1, 1.0, 4.0),
    formal_errors=None,
    covariance=None,
) -> _FakeEstimationOutput:
    """An output whose history is parameter-major, as TudatPy stores it."""
    initial = np.asarray(initial, dtype=float)
    final = np.asarray(final, dtype=float)
    history = np.column_stack([initial, final])
    return _FakeEstimationOutput(history, final, formal_errors, covariance)


# -- arithmetic -------------------------------------------------------------


def test_relative_change_percent():
    change = _relative_change_percent(np.array([1.0, 2.0]), np.array([1.1, 1.0]))
    assert change == pytest.approx([10.0, -50.0])


def test_relative_change_is_nan_for_zero_initial():
    change = _relative_change_percent(np.array([0.0]), np.array([1.0]))
    assert np.isnan(change[0])


def test_change_factor():
    factor = _change_factor(np.array([2.0, 4.0]), np.array([1.0, 8.0]))
    assert factor == pytest.approx([0.5, 2.0])


def test_change_factor_is_nan_for_zero_initial():
    factor = _change_factor(np.array([0.0]), np.array([1.0]))
    assert np.isnan(factor[0])


def test_relative_formal_error_percent():
    relative = _relative_formal_error_percent(np.array([0.5, 1.0]), np.array([2.0, 0.0]))
    assert relative[0] == pytest.approx(25.0)
    assert np.isnan(relative[1])


def test_shift_sigma_is_nan_without_a_formal_error():
    shift = _shift_sigma(np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert np.isnan(shift[0])


def test_initial_parameters_is_first_history_column():
    output = _output(initial=(5.0, 6.0), final=(7.0, 8.0))
    assert _initial_parameters(output) == pytest.approx([5.0, 6.0])


def test_missing_history_raises():
    output = _FakeEstimationOutput(np.empty((0, 0)), np.array([1.0]))
    with pytest.raises(ValueError, match="no parameter history"):
        _initial_parameters(output)


# -- the wide frame ---------------------------------------------------------


def test_frame_has_every_derived_column():
    frame = build_parameter_frame(_output())
    assert list(frame.columns) == list(COLUMNS)


def test_frame_values():
    frame = build_parameter_frame(_output())

    assert frame["parameter"].tolist() == ["x", "y", "z"]
    assert frame["index"].tolist() == [0, 1, 2]
    assert frame["initial"].tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert frame["final"].tolist() == pytest.approx([1.1, 1.0, 4.0])
    assert frame["absolute_change"].tolist() == pytest.approx([0.1, -1.0, 0.0])
    assert frame["relative_change_percent"].tolist() == pytest.approx([10.0, -50.0, 0.0])
    assert frame["change_factor"].tolist() == pytest.approx([1.1, 0.5, 1.0])


def test_frame_uses_supplied_parameter_names():
    frame = build_parameter_frame(_output(), parameter_names=["a", "b", "c"])
    assert frame["parameter"].tolist() == ["a", "b", "c"]


def test_frame_pads_short_parameter_names():
    frame = build_parameter_frame(_output(), parameter_names=["a"])
    assert frame["parameter"].tolist() == ["a", "p1", "p2"]


def test_frame_rejects_mismatched_vectors():
    output = _FakeEstimationOutput(np.ones((3, 2)), np.ones(4))
    with pytest.raises(ValueError, match="differ in length"):
        build_parameter_frame(output)


def test_frame_carries_formal_errors_and_sigmas():
    output = _output(
        initial=(1.0, 2.0, 4.0),
        final=(1.1, 1.0, 4.0),
        formal_errors=np.array([0.1, 0.5, 2.0]),
    )
    frame = build_parameter_frame(output)

    assert frame["formal_error"].tolist() == pytest.approx([0.1, 0.5, 2.0])
    assert frame["relative_formal_error_percent"].tolist() == pytest.approx(
        [0.1 / 1.1 * 100.0, 0.5 / 1.0 * 100.0, 2.0 / 4.0 * 100.0]
    )
    # The estimate moved 0.1 with a 0.1 sigma, so it moved one sigma.
    assert frame["shift_sigma"].tolist() == pytest.approx([1.0, -2.0, 0.0])


def test_frame_without_formal_errors_is_nan_not_an_error():
    frame = build_parameter_frame(_output())
    assert frame["formal_error"].isna().all()
    assert frame["shift_sigma"].isna().all()


def test_frame_correlations_with_previous():
    covariance = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 9.0, -3.0],
            [0.0, -3.0, 16.0],
        ]
    )
    frame = build_parameter_frame(_output(formal_errors=np.ones(3), covariance=covariance))

    correlations = frame["correlation_with_previous"].tolist()
    assert np.isnan(correlations[0])
    assert correlations[1] == pytest.approx(1.0 / (3.0 * 2.0))
    assert correlations[2] == pytest.approx(-3.0 / (4.0 * 3.0))


def test_frame_without_covariance_has_nan_correlations():
    frame = build_parameter_frame(_output())
    assert frame["correlation_with_previous"].isna().all()


# -- the table producer -----------------------------------------------------


def test_producer_builds_the_wide_frame():
    frame = EstimationParameterTable(_output())._build()
    assert list(frame.columns) == list(COLUMNS)


def test_producer_accepts_parameter_names():
    frame = EstimationParameterTable(_output(), parameter_names=["a", "b", "c"])._build()
    assert frame["parameter"].tolist() == ["a", "b", "c"]


# -- the configured view ----------------------------------------------------


def _patch_output_dir(monkeypatch, tmp_path):
    """Point ``HydraConfig.get().runtime.output_dir`` at *tmp_path*."""
    from hydra.core.hydra_config import HydraConfig

    runtime = SimpleNamespace(output_dir=str(tmp_path))
    monkeypatch.setattr(
        HydraConfig, "get", staticmethod(lambda: SimpleNamespace(runtime=runtime))
    )


def test_view_selects_and_orders_columns():
    frame = build_parameter_frame(_output())
    view = TableView(frame, name="t", columns=["final", "parameter"])
    assert list(view.select().columns) == ["final", "parameter"]


def test_view_drops_missing_columns_with_a_warning(caplog):
    frame = build_parameter_frame(_output())
    view = TableView(frame, name="t", columns=["parameter", "nope"])
    assert list(view.select().columns) == ["parameter"]
    assert "nope" in caplog.text


def test_view_without_columns_keeps_everything():
    frame = build_parameter_frame(_output())
    assert list(TableView(frame, name="t").select().columns) == list(COLUMNS)


def test_view_publishes_csv_markdown_and_latex(tmp_path, monkeypatch):
    from orbitdet.analysis.tables import LatexTableWriter

    _patch_output_dir(monkeypatch, tmp_path)
    frame = build_parameter_frame(_output())
    TableView(
        frame,
        name="parameter_change",
        columns=["parameter", "initial", "final"],
        writers={"booktabs": LatexTableWriter(caption="Cap.")},
    ).publish()

    assert (tmp_path / "parameter_change.csv").is_file()
    assert (tmp_path / "parameter_change.md").is_file()
    # A single style writes the unsuffixed name.
    assert (tmp_path / "parameter_change.tex").is_file()
    assert pd.read_csv(tmp_path / "parameter_change.csv").shape == (3, 3)


def test_view_publishes_one_latex_file_per_style(tmp_path, monkeypatch):
    from orbitdet.analysis.tables import AATableWriter, LatexTableWriter

    _patch_output_dir(monkeypatch, tmp_path)
    TableView(
        build_parameter_frame(_output()),
        name="parameter_change",
        columns=["parameter", "initial"],
        writers={
            "booktabs": LatexTableWriter(caption="Cap."),
            "aa": AATableWriter(caption="Cap."),
        },
    ).publish()

    booktabs = (tmp_path / "parameter_change_booktabs.tex").read_text()
    aa = (tmp_path / "parameter_change_aa.tex").read_text()
    assert r"\begin{tabular}" in booktabs
    assert r"\begin{deluxetable}" in aa
    # Both describe the same numbers.
    assert "x & 1 \\\\" in booktabs
    assert "x & 1 \\\\" in aa


def test_view_without_writers_publishes_no_latex(tmp_path, monkeypatch):
    _patch_output_dir(monkeypatch, tmp_path)
    TableView(build_parameter_frame(_output()), name="t").publish()

    assert (tmp_path / "t.csv").is_file()
    assert not list(tmp_path.glob("*.tex"))
