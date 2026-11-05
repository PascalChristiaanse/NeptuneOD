"""Tests for :mod:`orbitdet.observations.selection`.

A selection is a pandas query over the standard frame, so these tests focus on
the three things that can go wrong: the query is invalid, it names a column that
does not exist, or it produces something that is not a per-row boolean.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from orbitdet.observations.selection import (
    STANDARD_COLUMNS,
    Selection,
    compile_selection,
)
from orbitdet.preprocessing.schema import empty_frame


def _frame():
    """A small frame with the columns the tests select on."""
    frame = empty_frame()
    frame["data_source"] = ["nsdb", "nsdb", "gaia_archive", "voyager"]
    frame["observatory_code"] = ["689", "675", "Gaia", "Voyager 2"]
    frame["epoch_tdb"] = [1.0, 2.0, 3.0, 4.0]
    return frame


# ===========================================================================
# Compilation
# ===========================================================================


class TestCompileSelection:
    def test_none_is_the_universal_selection(self):
        selection = compile_selection(None)
        assert selection.is_universal
        assert selection.query is None
        assert selection.mask(_frame()).all()

    def test_query_string_is_kept_verbatim(self):
        selection = compile_selection("data_source == 'nsdb'")
        assert selection.query == "data_source == 'nsdb'"
        assert not selection.is_universal

    def test_whitespace_is_stripped(self):
        assert compile_selection("  data_source == 'nsdb'  ").query == "data_source == 'nsdb'"

    def test_mapping_form_with_where(self):
        selection = compile_selection({"where": "data_source == 'nsdb'"})
        assert selection.query == "data_source == 'nsdb'"

    def test_omegaconf_mapping_form(self):
        selection = compile_selection(OmegaConf.create({"where": "epoch_tdb > 2.0"}))
        assert selection.query == "epoch_tdb > 2.0"

    def test_mapping_without_where_raises(self):
        # An unknown key is reported before the missing 'where', which is the
        # more useful message: it names what the caller actually wrote.
        with pytest.raises(ValueError, match="Unknown selection key"):
            compile_selection({"select": "data_source == 'nsdb'"})

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="must have a 'where' key"):
            compile_selection({})

    def test_mapping_with_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown selection key"):
            compile_selection({"where": "epoch_tdb > 1.0", "extra": 1})

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compile_selection("   ")

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="must be a query string"):
            compile_selection(42)


# ===========================================================================
# Validation against the schema
# ===========================================================================


class TestValidation:
    def test_unknown_column_is_rejected_at_compile_time(self):
        # The whole point of validating up front: a typo fails loudly here
        # rather than silently matching nothing at run time.
        with pytest.raises(ValueError, match="unknown column"):
            compile_selection("observatory_cod == '689'")

    def test_syntax_error_is_rejected(self):
        with pytest.raises(ValueError, match="not a valid expression"):
            compile_selection("data_source ==")

    def test_every_standard_column_is_selectable(self):
        for column in STANDARD_COLUMNS:
            # A bare column name is a valid (truthiness) expression.
            compile_selection(column)

    def test_derived_columns_are_selectable(self):
        # The analysis layer adds these on top of the schema.
        compile_selection("rejected == False")
        compile_selection("observation_set_index == 0")

    def test_arbitrary_python_is_not_reachable(self):
        with pytest.raises(ValueError):
            compile_selection("__import__('os').system('true')")


# ===========================================================================
# Evaluation
# ===========================================================================


class TestMask:
    def test_simple_equality(self):
        mask = compile_selection("data_source == 'nsdb'").mask(_frame())
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_conjunction(self):
        mask = compile_selection(
            "data_source == 'nsdb' and observatory_code != '689'"
        ).mask(_frame())
        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_disjunction(self):
        mask = compile_selection(
            "observatory_code == '689' or observatory_code == 'Gaia'"
        ).mask(_frame())
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_numeric_comparison(self):
        mask = compile_selection("epoch_tdb >= 3.0").mask(_frame())
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_in_operator(self):
        mask = compile_selection("observatory_code in ['689', '675']").mask(_frame())
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_string_method(self):
        mask = compile_selection("observatory_code.str.startswith('Voy')").mask(_frame())
        np.testing.assert_array_equal(mask, [False, False, False, True])

    def test_mask_is_aligned_with_the_frame(self):
        frame = _frame()
        assert compile_selection("epoch_tdb > 0").mask(frame).shape == (len(frame),)

    def test_select_returns_the_matching_rows(self):
        selected = compile_selection("data_source == 'nsdb'").select(_frame())
        assert selected["observatory_code"].tolist() == ["689", "675"]

    def test_matches_reports_any_row(self):
        frame = _frame()
        assert compile_selection("data_source == 'nsdb'").matches(frame)
        assert not compile_selection("data_source == 'nope'").matches(frame)

    def test_matches_on_an_empty_frame_is_false(self):
        assert not compile_selection("data_source == 'nsdb'").matches(empty_frame())

    def test_universal_selection_matches_an_empty_frame(self):
        # "every row" of nothing is vacuously false for `matches`, but the mask
        # must still be a well-formed empty boolean array.
        selection = compile_selection(None)
        assert selection.mask(empty_frame()).shape == (0,)
        assert not selection.matches(empty_frame())

    def test_missing_column_at_evaluation_time_raises(self):
        # Compilation validates against the schema, but a caller may hand in a
        # frame that lacks the column (e.g. a hand-built one).
        frame = pd.DataFrame({"epoch_tdb": [1.0]})
        with pytest.raises(ValueError, match="could not be evaluated"):
            compile_selection("data_source == 'nsdb'").mask(frame)

    def test_non_boolean_result_raises(self):
        # A query that yields a scalar cannot be used as a row mask.
        with pytest.raises(ValueError):
            compile_selection("epoch_tdb.sum()").mask(_frame())


# ===========================================================================
# Representation
# ===========================================================================


class TestRepr:
    def test_repr_shows_the_query(self):
        assert "data_source" in repr(compile_selection("data_source == 'nsdb'"))

    def test_repr_of_universal_selection(self):
        assert "all rows" in repr(compile_selection(None))

    def test_selection_is_hashable(self):
        # frozen dataclass: usable as a dict key / in a set.
        a = compile_selection("epoch_tdb > 1.0")
        b = compile_selection("epoch_tdb > 1.0")
        assert len({a, b}) == 1

    def test_direct_construction_skips_validation(self):
        # Documented escape hatch: no schema check, so evaluation is the only guard.
        selection = Selection(query="epoch_tdb > 1.0", description="manual")
        assert selection.mask(_frame()).tolist() == [False, True, True, True]
