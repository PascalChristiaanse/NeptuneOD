"""Tests for the LaTeX table writers.

The writers are the typesetting half of the table layer: a producer hands them a
dataframe and they return LaTeX source.  The tests pin the structure of both
environments (``booktabs`` and the A&A/AAS ``deluxetable``), the shared cell
formatting, and the escaping rules that keep a stray ``_`` from breaking a
build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orbitdet.analysis.tables import AATableWriter, LatexTableWriter, escape_latex


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": ["x", "v_x"],
            "initial": [1.0, 2.0],
            "final": [1.1, 2.2],
            "relative_change_percent": [10.0, np.nan],
        }
    )


# -- escaping ---------------------------------------------------------------


def test_escape_latex_escapes_table_breakers():
    assert escape_latex("a_b & c % d # e") == r"a\_b \& c \% d \# e"


def test_escape_latex_leaves_math_untouched():
    """A header such as ``$\\Delta$`` is already LaTeX and must survive."""
    assert escape_latex(r"$\Delta$") == r"$\Delta$"


# -- booktabs writer --------------------------------------------------------


def test_booktabs_structure():
    latex = LatexTableWriter(caption="Cap.", label="tab:x").render(_frame())

    assert r"\begin{table}[htbp]" in latex
    assert r"\caption{Cap.}" in latex
    assert r"\label{tab:x}" in latex
    assert r"\begin{tabular}{l r r r}" in latex
    assert r"\toprule" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex
    assert r"\end{tabular}" in latex


def test_booktabs_headers_and_formats():
    latex = LatexTableWriter(
        headers={"initial": "Initial"},
        formats={"initial": ".3e", "parameter": "math"},
    ).render(_frame())

    assert "parameter & Initial & final" in latex
    assert "$x$ & 1.000e+00" in latex


def test_booktabs_non_finite_becomes_dashes():
    latex = LatexTableWriter(formats={"relative_change_percent": ".2f"}).render(_frame())
    assert "--" in latex


def test_booktabs_empty_frame_returns_comment():
    assert LatexTableWriter().render(pd.DataFrame()) == "% No data."


def test_booktabs_columns_restrict_and_order():
    latex = LatexTableWriter(columns=["final", "parameter"]).render(_frame())
    header = latex.splitlines()[4]
    assert header.startswith("final & parameter")


def test_booktabs_footer_rows():
    latex = LatexTableWriter(footer=[["Total", "3.0"]]).render(_frame())
    assert "Total & 3.0" in latex


def test_booktabs_notes():
    latex = LatexTableWriter(notes=["A note."]).render(_frame())
    assert r"\footnotesize A note." in latex


def test_booktabs_raw_column_is_not_escaped():
    frame = pd.DataFrame({"label": [r"$v_x$"]})
    latex = LatexTableWriter(raw=["label"]).render(frame)
    assert r"$v_x$" in latex
    assert r"v\_x" not in latex


def test_booktabs_callable_format():
    latex = LatexTableWriter(formats={"initial": lambda value: f"<{value}>"}).render(_frame())
    assert "<1.0>" in latex


# -- A&A writer -------------------------------------------------------------


def test_aa_writer_uses_deluxetable():
    latex = AATableWriter(caption="Cap.", label="tab:x").render(_frame())

    assert r"\begin{deluxetable}{l r r r}" in latex
    assert r"\tablewidth{0pt}" in latex
    assert r"\tablecaption{Cap.\label{tab:x}}" in latex
    assert r"\tablehead{" in latex
    assert r"\colhead{parameter}" in latex
    assert r"\startdata" in latex
    assert r"\enddata" in latex
    assert r"\end{deluxetable}" in latex
    assert r"\toprule" not in latex


def test_aa_writer_typesize_and_notes():
    latex = AATableWriter(typesize=r"\scriptsize", notes=["A note."]).render(_frame())

    assert r"\tabletypesize{\scriptsize}" in latex
    assert r"\tablecomments{A note.}" in latex


def test_aa_writer_inherits_cell_formatting():
    latex = AATableWriter(formats={"parameter": "math", "initial": ".3e"}).render(_frame())
    assert "$x$ & 1.000e+00" in latex


def test_aa_writer_empty_frame_returns_comment():
    assert AATableWriter().render(pd.DataFrame()) == "% No data."


# -- subclassing ------------------------------------------------------------


def test_subclass_can_override_only_the_environment():
    """A journal writer should only have to describe its own preamble."""

    class _Mine(LatexTableWriter):
        def _begin(self, frame):
            return [r"\begin{mytable}", r"\toprule"]

        def _end(self, frame):
            return [r"\bottomrule", r"\end{mytable}"]

    latex = _Mine(formats={"initial": ".3e"}).render(_frame())

    assert r"\begin{mytable}" in latex
    assert r"\end{mytable}" in latex
    # Inherited header and body still render.
    assert "parameter & initial" in latex
    assert "1.000e+00" in latex


@pytest.mark.parametrize("writer", [LatexTableWriter(), AATableWriter()])
def test_writers_agree_on_cells(writer):
    """Both environments must typeset the same numbers identically."""
    latex = writer.render(_frame())
    assert "$x$" not in latex  # no math format configured
    assert "1.1" in latex


# -- style selection --------------------------------------------------------


def _writer_cfg(**overrides):
    from omegaconf import OmegaConf

    base = {
        "styles": ["booktabs", "aa"],
        "caption": "Cap.",
        "label": "tab:x",
        "headers": {"initial": "Initial"},
        "formats": {"initial": ".3e"},
        "style_options": {"aa": {"typesize": r"\scriptsize"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_build_writers_returns_one_writer_per_style():
    from orbitdet.analysis.tables import build_writers

    writers = build_writers(_writer_cfg())

    assert list(writers) == ["booktabs", "aa"]
    assert isinstance(writers["booktabs"], LatexTableWriter)
    assert isinstance(writers["aa"], AATableWriter)


def test_build_writers_shares_content_between_styles():
    """Both styles must describe the same table, so the formats cannot drift."""
    from orbitdet.analysis.tables import build_writers

    writers = build_writers(_writer_cfg())
    frame = _frame()

    booktabs = writers["booktabs"].render(frame)
    aa = writers["aa"].render(frame)

    assert "Initial" in booktabs and "Initial" in aa
    assert "1.000e+00" in booktabs and "1.000e+00" in aa
    assert "Cap." in booktabs and "Cap." in aa


def test_build_writers_applies_style_options_to_one_style_only():
    from orbitdet.analysis.tables import build_writers

    writers = build_writers(_writer_cfg())

    assert writers["aa"].typesize == r"\scriptsize"
    assert not hasattr(writers["booktabs"], "typesize")


def test_build_writers_defaults_to_booktabs():
    from orbitdet.analysis.tables import build_writers

    writers = build_writers(_writer_cfg(styles=None))

    assert list(writers) == ["booktabs"]


def test_build_writers_without_a_block_returns_nothing():
    from orbitdet.analysis.tables import build_writers

    assert build_writers(None) == {}


def test_build_writers_rejects_an_unknown_style():
    from orbitdet.analysis.tables import build_writers

    with pytest.raises(ValueError, match="Unknown table style 'mnras'"):
        build_writers(_writer_cfg(styles=["mnras"]))


def test_build_writer_returns_the_first_style():
    from orbitdet.analysis.tables import build_writer

    assert isinstance(build_writer(_writer_cfg()), LatexTableWriter)
