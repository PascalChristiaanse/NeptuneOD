"""Journal figure styling, applied through ``funkyfresh``.

Every figure in the thesis should share one typographic style, so the style is
applied once per process before any figure is built rather than per plotter.

The style is a *journal* style (``A&A``, ``AAS``, ``AGU``, ``MNRAS``, ...), not
a matplotlib style sheet: ``funkyfresh.set_style`` sets the rcParams that make a
figure match a journal's column width, font and font size.

Configuration
-------------
A figure config may name its own style::

    figure:
      style: "MNRAS"

Anything that names none gets :data:`DEFAULT_STYLE`.
"""

from __future__ import annotations

import logging
import shutil

logger = logging.getLogger(__name__)

#: Journal style applied when a figure config names none.
DEFAULT_STYLE = "A&A"

#: Colour cycle restored after the journal style is applied.
#:
#: ``funkyfresh`` sets ``axes.prop_cycle`` to a single black, which is right for
#: a journal that prints in greyscale but wrong here: several figures in this
#: project distinguish series by colour alone (residuals grouped by observatory,
#: two pole models overlaid).  Restoring the default cycle keeps those readable
#: while still taking the journal's fonts, sizes and line widths.
COLOR_CYCLE = "tab10"


def available_styles() -> list[str]:
    """Return the journal styles ``funkyfresh`` provides."""
    from funkyfresh import get_available_styles

    return list(get_available_styles())


def _latex_available() -> bool:
    """Whether a LaTeX engine is on ``PATH``.

    The journal styles set ``text.usetex: True``, which makes matplotlib shell
    out to LaTeX for every label.  Without a TeX installation that fails at
    draw time with ``FileNotFoundError: 'latex'``, so the style is applied with
    ``text.usetex`` disabled instead of crashing the figure.
    """
    return shutil.which("latex") is not None


def apply_style(
    name: str | None = None,
    fontsize: int | float | None = None,
    silent: bool = True,
    keep_color_cycle: bool = True,
) -> str:
    """Apply a journal style to matplotlib's runtime configuration.

    Parameters
    ----------
    name : str, optional
        Journal style name, e.g. ``"A&A"``.  Defaults to
        :data:`DEFAULT_STYLE`.
    fontsize : int or float, optional
        Override the style's default font size.
    silent : bool, optional
        Suppress ``funkyfresh``'s report of the parameters it set.  Defaults to
        ``True`` so a pipeline run is not flooded with style output.
    keep_color_cycle : bool, optional
        Restore :data:`COLOR_CYCLE` after applying the style.  The journal
        styles set a single black colour, which would make every multi-series
        figure indistinguishable.  Defaults to ``True``.

    Returns
    -------
    str
        The style name that was applied.

    Raises
    ------
    ValueError
        If *name* is not a style ``funkyfresh`` provides.
    """
    import matplotlib
    from funkyfresh import set_style

    style = name or DEFAULT_STYLE

    known = available_styles()
    if style not in known:
        raise ValueError(f"Unknown journal style '{style}'. Known: {', '.join(known)}.")

    logger.info("Applying journal style '%s'.", style)
    set_style(style, fontsize=fontsize, silent=silent)

    if keep_color_cycle:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
            color=matplotlib.colormaps[COLOR_CYCLE].colors
        )

    # The journal styles render text through LaTeX.  Fall back to matplotlib's
    # own mathtext when no TeX engine is installed, so a figure run does not
    # fail on a machine without LaTeX.
    if matplotlib.rcParams.get("text.usetex") and not _latex_available():
        logger.warning(
            "Journal style '%s' requests LaTeX rendering but no 'latex' executable "
            "was found on PATH; falling back to matplotlib mathtext.",
            style,
        )
        matplotlib.rcParams["text.usetex"] = False

    return style


__all__ = ["COLOR_CYCLE", "DEFAULT_STYLE", "apply_style", "available_styles"]
