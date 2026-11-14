"""Tests for the parameter-combination sweep comparison table.

The sweep's specifications are declared by the chapter 3 Snakefile, not by the
library, so these tests pin the generic machinery: the convergence reading (a
fit that hit the iteration limit is not converged), the truth-table extraction
from a specification mapping, and the table's row order and missing-fit
handling.  The enumeration itself is exercised end-to-end by the toy pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbitdet.analysis.tables import (
    COMPARISON_COLUMNS,
    SweepComparisonTable,
    build_sweep_frame,
    read_convergence,
    specification_cells,
    specification_columns,
    summarise_fit,
)

#: A fit the convergence checker stopped: no iteration-cap line.
_CONVERGED_LOG = """Calculating residuals and partials 108
Current residual: 34576.3
Calculating residuals and partials 108
Current residual: 34576.2
Best iteration: 4 out of 4
Rms residual from best iteration: 34576.2
Cost function from best iteration: 1.38865e-07
"""

#: A fit that ran out of iterations, so the parameters were still moving.
_ITERATION_LIMIT_LOG = """Calculating residuals and partials 108
Current residual: 760544
Calculating residuals and partials 108
Current residual: 76098.5
Maximum number of iterations reached
Best iteration: 4 out of 4
Rms residual from best iteration: 76098.5
"""

#: A fit the checker stopped, but whose best iteration is no better than its
#: starting parameters: it never converged on anything.
_NO_IMPROVEMENT_LOG = """Calculating residuals and partials 108
Current residual: 34576.2
Calculating residuals and partials 108
Current residual: 34576.2
Best iteration: 2 out of 2
Rms residual from best iteration: 34576.2
"""

#: A fit that got worse: the best iteration is above the starting residual.
_WORSE_LOG = """Calculating residuals and partials 108
Current residual: 1000.0
Calculating residuals and partials 108
Current residual: 5000.0
Best iteration: 2 out of 2
Rms residual from best iteration: 5000.0
"""


class _FakeOutput:
    """Minimal stand-in carrying the attributes the summary reads."""

    def __init__(self, residuals, n_parameters: int = 6):
        self.final_residuals = np.asarray(residuals, dtype=float)
        self.final_parameters = np.zeros(n_parameters)


_GRAVITY_SPECS = [
    {
        "slug": "gravity_baseline",
        "label": "gravity_baseline",
        "neptune_GM": False,
        "triton_GM": False,
        "neptune_j2_j4": False,
    },
    {
        "slug": "gravity_nep_GM",
        "label": "gravity_nep_GM",
        "neptune_GM": True,
        "triton_GM": False,
        "neptune_j2_j4": False,
    },
    {
        "slug": "gravity_tri_GM",
        "label": "gravity_tri_GM",
        "neptune_GM": False,
        "triton_GM": True,
        "neptune_j2_j4": False,
    },
]

_POLE_SPECS = [
    {
        "slug": "pole_baseline",
        "label": "pole_baseline",
        "iau_rotation_model_pole": False,
        "iau_rotation_model_pole_rate": False,
        "iau_rotation_model_pole_librations_order1": False,
        "iau_rotation_model_pole_librations_order2": False,
    },
    {
        "slug": "pole_pole_rate_librations_order2",
        "label": "pole_pole_rate_librations_order2",
        "iau_rotation_model_pole": True,
        "iau_rotation_model_pole_rate": True,
        "iau_rotation_model_pole_librations_order1": False,
        "iau_rotation_model_pole_librations_order2": True,
    },
]


class TestSpecificationCells:
    def test_slug_and_label_are_not_cells(self):
        cells = specification_cells(_GRAVITY_SPECS[1])
        assert "slug" not in cells
        assert "label" not in cells
        assert cells == {"neptune_GM": True, "triton_GM": False, "neptune_j2_j4": False}

    def test_columns_are_the_union_in_first_seen_order(self):
        specifications = [
            {"slug": "a", "x": True},
            {"slug": "b", "y": True, "x": False},
        ]
        assert specification_columns(specifications) == ("x", "y")

    def test_columns_of_a_heterogeneous_group_are_complete(self):
        # A specification that omits a column must not shrink the table.
        specifications = [{"slug": "a", "x": True}, {"slug": "b"}]
        assert specification_columns(specifications) == ("x",)


class TestReadConvergence:
    def test_checker_stopped_fit_is_converged(self):
        assert read_convergence(_CONVERGED_LOG) == (True, 4, 4)

    def test_iteration_limit_is_not_converged(self):
        # The residual is small, but the solver stopped on the cap, so the
        # parameters were still moving.
        assert read_convergence(_ITERATION_LIMIT_LOG) == (False, 4, 4)

    def test_unreadable_log_is_not_converged(self):
        assert read_convergence("") == (False, None, None)

    def test_fit_that_never_improved_is_not_converged(self):
        # The checker fired, but the best iteration is no better than the start,
        # so the fit settled on its initial parameters rather than converging.
        assert read_convergence(_NO_IMPROVEMENT_LOG) == (False, 2, 2)

    def test_fit_that_got_worse_is_not_converged(self):
        assert read_convergence(_WORSE_LOG) == (False, 2, 2)

    def test_log_without_a_best_residual_is_not_converged(self):
        log = "Current residual: 100.0\nBest iteration: 2 out of 2\n"
        assert read_convergence(log) == (False, 2, 2)


class TestSummariseFit:
    def test_row_carries_truth_table_and_statistics(self):
        specification = _GRAVITY_SPECS[1]
        output = _FakeOutput([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        row = summarise_fit(specification, output, _CONVERGED_LOG)

        assert row["specification"] == "gravity_nep_GM"
        assert row["neptune_GM"] is True
        assert row["triton_GM"] is False
        assert row["converged"] is True
        assert row["n_parameters"] == 6
        assert row["n_residuals"] == 6
        assert row["rms"] == pytest.approx(np.sqrt((9 + 16) / 6))
        assert row["max_abs"] == pytest.approx(4.0)
        # Two observations of three components: R = [3, 0], S = [4, 0], W = [0, 0].
        assert row["rms_R"] == pytest.approx(np.sqrt(9 / 2))
        assert row["rms_S"] == pytest.approx(np.sqrt(16 / 2))
        assert row["rms_W"] == pytest.approx(0.0)

    def test_pole_row_carries_libration_order(self):
        specification = _POLE_SPECS[1]
        row = summarise_fit(specification, _FakeOutput([1.0, 1.0, 1.0]), _CONVERGED_LOG)
        assert row["iau_rotation_model_pole_librations_order1"] is False
        assert row["iau_rotation_model_pole_librations_order2"] is True
        assert row["iau_rotation_model_pole"] is True
        assert row["iau_rotation_model_pole_rate"] is True


class TestBuildSweepFrame:
    def _rows(self, specifications):
        return [
            {
                "specification": specification["slug"],
                "label": specification["label"],
                **specification_cells(specification),
                "converged": True,
                "rms": float(index),
            }
            for index, specification in enumerate(specifications)
        ]

    def test_rows_follow_the_specification_order(self):
        rows = list(reversed(self._rows(_GRAVITY_SPECS)))
        frame = build_sweep_frame(_GRAVITY_SPECS, rows)
        assert list(frame["specification"]) == [s["slug"] for s in _GRAVITY_SPECS]

    def test_missing_fit_is_listed_as_not_converged(self):
        rows = self._rows(_GRAVITY_SPECS)[1:]
        frame = build_sweep_frame(_GRAVITY_SPECS, rows)
        missing = frame[frame["specification"] == _GRAVITY_SPECS[0]["slug"]].iloc[0]
        assert bool(missing["converged"]) is False
        assert np.isnan(missing["rms"])

    def test_columns_are_present(self):
        frame = build_sweep_frame(_POLE_SPECS, self._rows(_POLE_SPECS))
        for column in COMPARISON_COLUMNS:
            assert column in frame.columns
        columns = specification_columns(_POLE_SPECS)
        assert list(frame.columns[: len(columns) + 3]) == [
            "specification",
            "label",
            *columns,
            "converged",
        ]

    def test_table_builds_the_same_frame(self):
        rows = self._rows(_GRAVITY_SPECS)
        table = SweepComparisonTable(_GRAVITY_SPECS, rows)
        pd.testing.assert_frame_equal(table._build(), build_sweep_frame(_GRAVITY_SPECS, rows))
        assert table.name == "sweep_comparison"


# -- The chapter 3 sweep CSVs ----------------------------------------------
#
# The sweep is a CSV per group: a row per fit, a column per parameter.  These
# tests pin the row count and the mutual exclusion of the libration orders,
# which is the one invariant a hand-written CSV can break.

_SWEEPS = Path(__file__).resolve().parents[1] / "conf" / "sweeps"


class TestSweepCsv:
    def test_gravity_has_every_combination_of_three_parameters(self):
        frame = pd.read_csv(_SWEEPS / "gravity.csv")
        assert len(frame) == 8
        assert len(frame.drop_duplicates()) == 8
        assert list(frame.columns) == ["neptune_GM", "triton_GM", "neptune_j2_j4"]

    def test_pole_has_every_combination_of_two_parameters_and_three_orders(self):
        frame = pd.read_csv(_SWEEPS / "pole.csv")
        assert len(frame) == 12
        assert len(frame.drop_duplicates()) == 12

    def test_libration_orders_are_never_combined(self):
        frame = pd.read_csv(_SWEEPS / "pole.csv")
        both = (
            frame["iau_rotation_model_pole_librations_order1"]
            & frame["iau_rotation_model_pole_librations_order2"]
        )
        assert not both.any()

    def test_every_column_is_boolean(self):
        for name in ("gravity", "pole"):
            frame = pd.read_csv(_SWEEPS / f"{name}.csv")
            assert all(frame[column].isin([True, False]).all() for column in frame.columns)
