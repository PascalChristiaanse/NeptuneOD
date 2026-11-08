"""Named colour palettes for the plotters.

A figure config names a palette rather than listing hex codes, so the same
colourblind-safe set can be reused across figures and changed in one place::

    styling:
      palette: "okabe_ito"

The palette is cycled in order, so a two-series figure takes the first two
colours and a five-series figure takes the first five.

Why named palettes
------------------
Listing hex codes per figure invites drift: two figures that should look alike
end up with slightly different blues.  A name keeps them consistent, and makes
"is this figure colourblind-safe?" a question about one table rather than about
every YAML file.
"""

from __future__ import annotations

#: Okabe & Ito (2008), the standard colourblind-safe qualitative palette.
#:
#: Ordered so the first two entries are the canonical two-series pair (blue and
#: vermillion), which stay distinguishable under deuteranopia, protanopia and
#: tritanopia.  The remaining colours are ordered by how well they separate from
#: the ones before them.
OKABE_ITO: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Paul Tol's "bright" scheme, also designed for colour vision deficiency.
TOL_BRIGHT: list[str] = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

#: Paul Tol's "muted" scheme, a lower-saturation colourblind-safe variant.
TOL_MUTED: list[str] = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]

#: Matplotlib's default qualitative cycle.
TAB10: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

#: A single colour, for figures where the curve shape is the message.
BLACK: list[str] = ["#000000"]

#: Every palette a figure config may name.
PALETTES: dict[str, list[str]] = {
    "okabe_ito": OKABE_ITO,
    "tol_bright": TOL_BRIGHT,
    "tol_muted": TOL_MUTED,
    "tab10": TAB10,
    "black": BLACK,
}

#: Palette used when a figure config names none.
DEFAULT_PALETTE = "black"


def palette(name: str | None = None) -> list[str]:
    """Return the colours of a named palette.

    Parameters
    ----------
    name : str, optional
        A key of :data:`PALETTES`.  Defaults to :data:`DEFAULT_PALETTE`.

    Returns
    -------
    list[str]
        A copy of the palette's colours, in cycling order.

    Raises
    ------
    ValueError
        If *name* is not a known palette.
    """
    key = name or DEFAULT_PALETTE
    try:
        return list(PALETTES[key])
    except KeyError:
        known = ", ".join(sorted(PALETTES))
        raise ValueError(f"Unknown palette '{key}'. Known: {known}.") from None


__all__ = [
    "BLACK",
    "DEFAULT_PALETTE",
    "OKABE_ITO",
    "PALETTES",
    "TAB10",
    "TOL_BRIGHT",
    "TOL_MUTED",
    "palette",
]
