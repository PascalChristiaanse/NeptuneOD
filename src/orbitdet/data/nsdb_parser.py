"""Parsing of NSDB dataset descriptor pages into :class:`NSDBRecord` objects.

NSDB (the IMCCE Natural Satellite Data Base) publishes each dataset as a pair of
files: a ``.html`` descriptor and a ``.txt`` data file. This module reads the
descriptor only. It is pure: no network access, no Tudat, no dataframe — just
HTML text in, a record out.

Descriptor layout
-----------------
The descriptor's ``<pre>`` block is divided into sections introduced by a
heading line::

    Contents.        # what the dataset holds: planet, satellites, dates, ...
    [References.]    # bibliography (heading spelling varies, see below)
    Informations.    # the astrometric metadata: frame, equinox, time scale, ...
    [Comments.]      # free text
    Format.          # numbered description of the data columns

Each of ``Contents.`` and ``Informations.`` is a ``key: value`` block where a
long value continues on indented following lines.

Heading spellings in the wild
-----------------------------
Across the 83 datasets in ``data/raw`` the heading is *not* consistent, and a
parser that insists on one spelling silently loses sections:

===================  =====
Spelling             Count
===================  =====
``References.``      49
``Reference.``       26
``Reference``         5
``Informations.``    82
``Comments.``        68
none of the above     3
===================  =====

So headings are matched case-insensitively, with or without the trailing period,
by :meth:`NSDBParser.normalise_heading`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bs4
import pyparsing as pp

logger = logging.getLogger(__name__)

__all__ = [
    "NSDBRecord",
    "NSDBParser",
    "SECTION_ALIASES",
    "FORMAT_COLUMN_SEPARATOR",
]


#: Section heading (lowercased, period stripped) to canonical section name.
#: Both the singular and plural reference headings occur in real data.
SECTION_ALIASES: dict[str, str] = {
    "contents": "contents",
    "informations": "informations",
    "information": "informations",
    "reference": "reference",
    "references": "reference",
    "comments": "comments",
    "comment": "comments",
    "format": "format",
}

#: A rule of four or more hyphens, marking the start of the worked data example
#: at the end of the Format section.
FORMAT_COLUMN_SEPARATOR = re.compile(r"^-{4,}\s*$")

#: ``N. description`` — the numbered column declaration itself.
_FORMAT_ENTRY = re.compile(r"^(\d+)\.\s*(.*)$")

#: ``083 - Golosseevo-Kiev``. The code is bounded to three digits, which is what
#: MPC codes are, so a stray bibliography line such as ``1874MNRAS__35___49D``
#: or ``1874 - P. 49`` cannot masquerade as an observatory.
_OBSERVATORY_ENTRY = re.compile(r"^(\d{1,3})\s*-\s*(.+)$")

#: ``999 Bordeaux-Floirac`` — code and name with no separator dash (nm0018).
_OBSERVATORY_ENTRY_NO_DASH = re.compile(r"^(\d{1,3})\s+([A-Za-z].*)$")

#: ``H80 - Halsted observatory`` — MPC also issues codes beginning with a
#: letter (nm0028).
_OBSERVATORY_ENTRY_ALPHANUMERIC = re.compile(r"^([A-Z]\d{2})\s*-\s*(.+)$")

#: ``Cambridge`` — a name with no code at all (nm0051).
_OBSERVATORY_ENTRY_NAME_ONLY = re.compile(r"^([A-Za-z][A-Za-z .,'()\-]*)$")

#: ``083 - DL: A telescope`` or ``1 - Reflector, D=156 cm``. Whitespace is
#: required on *both* sides of the dash. Without that requirement the pattern
#: also matches free-text descriptions such as ``26-inch refractor``,
#: ``40-inch`` and ``1-m reflectors``, eight of which occur in real data — each
#: would be split into a bogus entry with index 26, 40 or 1.
_TELESCOPE_ENTRY = re.compile(r"^(\d+)\s+-\s+(.+)$")

#: ``code 337`` embedded in a telescope description.
_EMBEDDED_CODE = re.compile(r"code\s+(\d+)", re.IGNORECASE)

_DATE_RANGE = re.compile(r"^(\d{4})\s*-\s*(\d{4})$")
_BARE_YEAR = re.compile(r"^(\d{4})$")
_PLANET_ENTRY = re.compile(r"^(\d+)\s*-\s*(.*)$")
_YEAR = re.compile(r"\d{4}")

#: ``key: value``. Keys are words, spaces and periods only. The exclusion of
#: ``-`` and ``:`` is deliberate and load-bearing: it is what stops a
#: continuation line such as ``188 - Z6: The 60-cm Zeiss Reflector`` from being
#: promoted to a key of its own.
_KEY = pp.Regex(r"[\w .]+")("key")
_VALUE = pp.Regex(r"[^\n]*")("value")
_FIELD = _KEY + pp.Suppress(pp.Literal(":")) + _VALUE


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_lines(value: str | list[str] | None) -> list[str]:
    """Return a field's value as a list of non-empty stripped lines."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_scalar(value: str | list[str] | None) -> str | None:
    """Collapse a possibly multi-line value into a single spaced string.

    The Informations fields are semantically single-valued even when NSDB wraps
    them across lines, so joining with a space and collapsing runs of whitespace
    reconstructs the intended value. ``telescope`` and ``observatory`` are the
    exceptions and are never passed through here.
    """
    lines = _as_lines(value)
    if not lines:
        return None
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _resolve_telescope_code(
    name_to_code: dict[str, int | None],
    wanted: str,
) -> int | None:
    """Resolve a telescope name written in a Format column to its observatory code.

    A Format column names its instruments in whatever shorthand its author used,
    which is not always the full name in the ``telescope`` field it refers to.
    ``sm0074`` writes ``1 - DLFA`` where the field says ``083 - DLFA (Double Long
    Focus Astrograph), D/f=0.4/5.5m``, so an exact-name lookup finds nothing and
    the dataset is rejected.

    An exact match is tried first, so a dataset that resolves today keeps its
    mapping.  Only then is a *unique* prefix match accepted, and only at a word
    boundary so that an abbreviation cannot match a longer unrelated name.  An
    ambiguous abbreviation yields ``None`` rather than a guess: pointing the
    observations at the wrong observatory is a silent positional error, whereas
    ``None`` is reported and investigated.

    Args:
        name_to_code: Telescope full name to observatory code.
        wanted: The shorthand from the Format column.

    Returns:
        The observatory code, or ``None`` if it cannot be determined uniquely.
    """
    if wanted in name_to_code:
        return name_to_code[wanted]

    candidates = {
        code
        for name, code in name_to_code.items()
        if name.startswith(wanted) and (len(name) == len(wanted) or name[len(wanted)] in " (,")
    }
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _first_int(text: str) -> int | None:
    """Return the first run of digits in *text* as an int, or ``None``."""
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NSDBRecord:
    """Everything a descriptor page says about one NSDB dataset.

    Attributes:
        identifier: Dataset name, e.g. ``nm0001``.
        sections: Raw text of each section, keyed by canonical name.
        contents: The parsed ``key: value`` mapping of the Contents section.
        informations: The parsed ``key: value`` mapping of the Informations
            section.
        format_columns: Data column index (1-based) to description.
        planet: ``{"number": int, "name": str}``, or ``None``.
        satellites: Short name to ``{"designation", "number_of_observations"}``.
        total_number: Declared observation count, or ``None``.
        dates: ``{"start_year", "end_year"}``, or ``None``.
        observatories: ``[{"code", "name", "telescopes"}]`` in file order.
        telescopes: ``[{"code", "name", "description", "index"}]``, or a plain
            descriptive string for single-telescope datasets that do not use
            observatory-code prefixes, or ``None``.
        telescope_index: Data column value to observatory code, for the two
            multi-observatory datasets. Empty when not applicable.
        observables: The fused observation type, e.g.
            ``absolute_absolute_CCD_nsdb``. This is the key the observation
            factories are registered under, so it must not be renamed.
        parameters: The remaining Informations fields, already mapped to their
            snake_case config names.
        reference: Bibliography text, or ``None``.
        comments: Free-text comments, or ``None``.
        unparsed_observatory_lines: Observatory-section lines that did not match
            the ``NNN - Name`` form. Recorded rather than discarded so that a
            parsing gap is visible instead of silently dropping data.
    """

    identifier: str
    sections: dict[str, str] = field(default_factory=dict)
    contents: dict[str, Any] = field(default_factory=dict)
    informations: dict[str, Any] = field(default_factory=dict)
    format_columns: dict[int, str] = field(default_factory=dict)

    planet: dict[str, Any] | None = None
    satellites: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_number: int | None = None
    dates: dict[str, int] | None = None
    observatories: list[dict[str, Any]] = field(default_factory=list)
    telescopes: list[dict[str, Any]] | str | None = None
    telescope_index: dict[str, int | None] = field(default_factory=dict)
    observables: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    reference: str | None = None
    comments: str | None = None
    unparsed_observatory_lines: list[str] = field(default_factory=list)

    @property
    def primary_observatory_code(self) -> int | str | None:
        """The first observatory's MPC code.

        This is the value the observation factories actually consume. It comes
        from the ``observatory:`` line itself, so it is the one entry that a
        continuation-absorbing parse of the section cannot corrupt.

        Returns:
            The code as it appears in the source: normally an ``int``, but a
            ``str`` for MPC's letter-prefixed codes such as ``H80``, and
            ``None`` for the few datasets that name an observatory without a
            code.
        """
        return self.observatories[0].get("code") if self.observatories else None

    def to_config(self, data_file: str | Path | None = None) -> dict[str, Any]:
        """Build the flat mapping written out as a dataset YAML config.

        The key order reproduces the previously generated configs, and
        ``type`` keeps its fused form, because the observation factories are
        registered under that exact string and the 82 existing configs in
        ``conf/datasets/generated`` were written with it.

        Args:
            data_file: Path to the dataset's ``.txt`` file. When given it is
                added under ``file`` as an absolute path, which is how the
                factories locate the data.

        Returns:
            An ordered mapping ready for ``yaml.safe_dump``.
        """
        config: dict[str, Any] = {
            "identifier": self.identifier,
            "format_columns": {
                str(index): self.format_columns[index] for index in sorted(self.format_columns)
            },
        }

        if self.planet is not None:
            config["planet"] = self.planet
        if self.satellites:
            config["satellites"] = self.satellites
        if self.total_number is not None:
            config["total_number"] = self.total_number
        if self.observables is not None:
            config["type"] = self.observables
        if self.dates is not None:
            config["dates"] = self.dates
        if self.observatories:
            config["observatory"] = self.observatories
        if "relative_to" in self.parameters:
            config["relative_to"] = self.parameters["relative_to"]

        # The Informations fields, in their established order.
        for key in (
            "reference_frame",
            "centre_of_frame",
            "epoch_of_equinox",
            "time_scale",
            "reduction",
            "coordinates",
            "diff_refraction",
            "receptor",
            "observers",
            "data_included_in_standard_data_file",
        ):
            if key in self.parameters:
                config[key] = self.parameters[key]

        if self.telescopes is not None:
            config["telescope"] = self.telescopes

        # Both spellings are emitted so a consumer may use either.
        if "centre_of_frame" in self.parameters:
            config["center_of_frame"] = self.parameters["centre_of_frame"]
        elif "center_of_frame" in self.parameters:
            config["center_of_frame"] = self.parameters["center_of_frame"]

        if self.telescope_index:
            config["telescope_index"] = self.telescope_index

        if self.reference is not None:
            config["reference"] = self.reference
        if self.comments is not None:
            config["comments"] = self.comments
        if data_file is not None:
            config["file"] = str(Path(data_file).resolve())

        return config


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


class NSDBParser:
    """Turn an NSDB descriptor page into an :class:`NSDBRecord`.

    The class is stateless; a single instance can be reused for any number of
    datasets and may safely be shared between threads.

    Examples:
        >>> parser = NSDBParser()
        >>> record = parser.parse_file("data/raw/nm0001/nm0001.html")
        >>> record.planet["name"]
        'Neptune'
        >>> record.primary_observatory_code
        673
    """

    # ------------------------------------------------------------------
    # Generic tokenisers: text in, structure out
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_heading(line: str) -> str | None:
        """Return the canonical section name if *line* is a section heading.

        A heading is a single token, optionally followed by a period, and never
        contains a colon. Requiring a single token is what keeps bibliography
        lines such as ``no. 404, Berkeley`` from being mistaken for headings.

        Args:
            line: One line of the descriptor's ``<pre>`` block.

        Returns:
            The canonical section name, or ``None`` if the line is not a
            heading.
        """
        text = line.strip()
        if not text or ":" in text:
            return None
        cleaned = text.rstrip(".").strip()
        if not cleaned or any(character.isspace() for character in cleaned):
            return None
        return SECTION_ALIASES.get(cleaned.lower())

    @classmethod
    def split_sections(cls, text: str) -> dict[str, str]:
        """Split the ``<pre>`` text into its sections, keyed by canonical name.

        An unrecognised heading does **not** terminate the section being read.
        That matters for ``nm0036``, which legitimately omits ``Informations.``
        entirely — a parser that ended the current section at any heading-like
        line would drop the rest of the Contents block there.

        Args:
            text: The descriptor's ``<pre>`` text.

        Returns:
            Raw section text by canonical name. Sections absent from the input
            are absent from the result.
        """
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        boundaries: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            name = cls.normalise_heading(line)
            if name is not None:
                boundaries.append((index, name))

        sections: dict[str, str] = {}
        for position, (start, name) in enumerate(boundaries):
            end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
            body = "\n".join(lines[start + 1 : end]).strip()
            # First occurrence wins, so a repeated heading cannot clobber a
            # section that was already read.
            sections.setdefault(name, body)
        return sections

    @staticmethod
    def parse_key_value_block(text: str) -> dict[str, str | list[str]]:
        """Parse a ``key: value`` block, joining indented continuation lines.

        Single-line values are returned as strings; multi-line values as lists
        of strings in file order. A value may itself contain colons — everything
        after the first one is part of the value.

        Args:
            text: The section body.

        Returns:
            Key to value. Keys are stripped but not case-folded, so the caller
            can see exactly what the source said.

        Raises:
            ValueError: If the same key appears twice in one section, which
                would silently discard the earlier value.
        """
        result: dict[str, str | list[str]] = {}
        current_key: str | None = None
        current_values: list[str] = []

        def finalise() -> None:
            nonlocal current_key, current_values
            if current_key is not None:
                result[current_key] = (
                    current_values[0] if len(current_values) == 1 else list(current_values)
                )
                current_key, current_values = None, []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                # A blank line terminates the field being read. NSDB uses a
                # blank line to close a block, and treating it as a
                # continuation is what let nm0020's bibliography be absorbed
                # into its observatory list. No legitimate field value in the
                # 83 real datasets spans a blank line.
                finalise()
                continue

            try:
                parsed = _FIELD.parse_string(line)
            except pp.ParseException:
                # Not a field declaration: an indented continuation of the
                # previous field, or leading prose that has no owner.
                if current_key is not None:
                    current_values.append(line)
                continue

            key = parsed.key.strip()
            finalise()
            if key in result:
                raise ValueError(f"Duplicate key {key!r} in section.")
            current_key = key
            current_values = [parsed.value.strip()]

        finalise()
        return result

    @staticmethod
    def parse_format_block(text: str) -> dict[int, str]:
        """Parse the Format section into a column index to description mapping.

        Parsing stops at the first rule of hyphens. Everything after it is a
        worked example of the data layout, and several real datasets publish an
        example row that starts with a number — ``1903  2 16  8 45 43  272.99``
        in ``nm0076``, for instance. Treating such a row as a column
        declaration produced bogus entries (indices 65, 197 and 272 across five
        datasets) that then propagated into the dataframe column names.

        Args:
            text: The Format section body.

        Returns:
            Column index (1-based) to description.
        """
        columns: dict[int, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if FORMAT_COLUMN_SEPARATOR.match(line):
                break
            match = _FORMAT_ENTRY.match(line)
            if match:
                # NSDB pads the label column with runs of spaces for alignment
                # (``Year   of the moment``); collapse them so the description
                # reads as prose. The existing generated configs already have
                # them collapsed, so this also keeps the contract stable.
                description = re.sub(r"\s+", " ", match.group(2)).strip()
                columns[int(match.group(1))] = description
        return columns

    # ------------------------------------------------------------------
    # Domain extraction: NSDB's own vocabulary
    # ------------------------------------------------------------------

    @staticmethod
    def parse_planet(value: str | list[str] | None) -> dict[str, Any] | None:
        """Parse ``8 - Neptune`` into ``{"number": 8, "name": "Neptune"}``."""
        text = _as_scalar(value)
        if not text:
            return None
        match = _PLANET_ENTRY.match(text)
        if match:
            return {"number": int(match.group(1)), "name": match.group(2).strip()}
        return {"number": _first_int(text), "name": text}

    @staticmethod
    def parse_satellites(value: str | list[str] | None) -> dict[str, dict[str, Any]]:
        """Parse satellite lines into a short-name keyed mapping.

        Accepts ``N2-Nereid   :174``, ``N1 - Triton : 10`` and a bare
        designation with no count. Several lines are merged rather than the last
        one winning.

        Args:
            value: The ``satellites`` field value.

        Returns:
            Short name to ``{"designation", "number_of_observations"}``.
        """
        satellites: dict[str, dict[str, Any]] = {}
        for entry in _as_lines(value):
            designation = entry
            count: int | None = None
            if ":" in entry:
                designation, count_text = entry.split(":", 1)
                designation = designation.strip()
                count = _first_int(count_text)
            if not designation:
                continue
            short_name = designation.split("-")[-1].strip() if "-" in designation else designation
            satellites[short_name] = {
                "designation": designation,
                "number_of_observations": count,
            }
        return satellites

    @staticmethod
    def parse_dates(value: str | list[str] | None) -> dict[str, int] | None:
        """Parse a ``dates`` field into a first-start/last-end year pair.

        Real values include a single range (``1993-1998``), a single year
        (``1874``) and *discontinuous* ranges (``1898-1899, 1901-1902``). The
        last form used to be mis-parsed by stripping non-digits from the whole
        string, which fused ``1899`` and ``1901`` into the year ``18991901``.
        Multiple ranges are now parsed separately, and the returned span is the
        outer envelope.

        Args:
            value: The ``dates`` field value.

        Returns:
            ``{"start_year", "end_year"}``, or ``None`` when no year is present.
        """
        text = _as_scalar(value)
        if not text:
            return None

        starts: list[int] = []
        ends: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            match = _DATE_RANGE.match(part)
            if match:
                starts.append(int(match.group(1)))
                ends.append(int(match.group(2)))
                continue
            year = _BARE_YEAR.match(part)
            if year:
                starts.append(int(year.group(1)))
                ends.append(int(year.group(1)))
                continue
            years = _YEAR.findall(part)
            if years:
                starts.append(int(years[0]))
                ends.append(int(years[-1]))

        if not starts:
            return None
        return {"start_year": min(starts), "end_year": max(ends)}

    @staticmethod
    def parse_observatory_entries(
        value: str | list[str] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Parse observatory entries, returning those that did not match.

        NSDB is not consistent about this field. Four forms occur in real data:

        ============================  ==========================
        Form                          Example
        ============================  ==========================
        ``NNN - Name``                ``083 - Golosseevo-Kiev``
        ``NNN Name`` (no dash)        ``999 Bordeaux-Floirac``
        ``XNN - Name``                ``H80 - Halsted observatory``
        ``Name`` (no code)            ``Cambridge``
        ============================  ==========================

        All four are accepted. An entry with no code yields ``code: None``
        rather than being dropped, because the name is still meaningful and
        discarding it would lose the dataset's only positional information.

        Args:
            value: The ``observatory`` field value.

        Returns:
            A tuple of (observatories, unmatched lines). The second element is
            normally empty; a non-empty value signals prose that reached this
            field, which the caller surfaces as a warning.
        """
        observatories: list[dict[str, Any]] = []
        unmatched: list[str] = []
        for entry in _as_lines(value):
            match = (
                _OBSERVATORY_ENTRY.match(entry)
                or _OBSERVATORY_ENTRY_ALPHANUMERIC.match(entry)
                or _OBSERVATORY_ENTRY_NO_DASH.match(entry)
            )
            if match:
                code: int | str = match.group(1)
                if code.isdigit():
                    code = int(code)
                observatories.append({"code": code, "name": match.group(2).strip()})
                continue

            name_only = _OBSERVATORY_ENTRY_NAME_ONLY.match(entry)
            if name_only:
                observatories.append({"code": None, "name": name_only.group(1).strip()})
                continue

            unmatched.append(entry)
        return observatories, unmatched

    @staticmethod
    def parse_telescope_entry(entry: str) -> dict[str, Any]:
        """Parse one telescope entry.

        Two layouts occur. In the first the leading number is the observatory
        code, as in ``083 - DL: The Tepfer Double Long-Focus Astrograph``. In
        the second it is a telescope index and the observatory code is embedded
        in the description, as in ``1 - Reflector, D=156 cm (at the Sheshan
        Station, code 337)``.

        Args:
            entry: One telescope line.

        Returns:
            ``{"code", "name", "description", "index"}``.
        """
        head, _, rest = entry.partition("-")
        index = _first_int(head)
        rest = rest.strip()
        if ":" in rest:
            short_name, _, description = rest.partition(":")
            short_name, description = short_name.strip(), description.strip()
        else:
            short_name, description = rest, ""

        embedded = _EMBEDDED_CODE.search(entry)
        code = int(embedded.group(1)) if embedded else index

        return {
            "code": code,
            "name": short_name,
            "description": description,
            "index": index,
        }

    @classmethod
    def parse_telescopes(cls, value: str | list[str] | None) -> list[dict[str, Any]] | str | None:
        """Parse the ``telescope`` field.

        A dataset that prefixes its telescope entries with observatory codes is
        describing several instruments, and is parsed structurally. A single
        free-text description such as ``Reflector, D = 0.61 m, f/16`` is kept as
        a string.

        Args:
            value: The ``telescope`` field value.

        Returns:
            A list of telescope dicts, a plain string, or ``None`` when absent.
        """
        lines = _as_lines(value)
        if not lines:
            return None
        if any(_TELESCOPE_ENTRY.match(line) for line in lines):
            telescopes: list[dict[str, Any]] = []
            previous_code: int | str | None = None
            for line in lines:
                entry = cls.parse_telescope_entry(line)
                if entry["code"] is None:
                    # A continuation line lists another instrument from the same
                    # observatory and repeats no code of its own.  ``sm0079``
                    # writes ``083 - DL: ...`` then ``- DW: ...`` on the next
                    # line, where both instruments are at code 083.  Inheriting
                    # the preceding code is what the layout means; leaving it
                    # None rejects the dataset.
                    entry["code"] = previous_code
                else:
                    previous_code = entry["code"]
                telescopes.append(entry)
            return telescopes
        return re.sub(r"\s+", " ", " ".join(lines)).strip() or None

    @staticmethod
    def parse_telescope_index(
        format_columns: dict[int, str],
        telescopes: list[dict[str, Any]] | str | None,
    ) -> dict[str, int | None]:
        """Map a data column's telescope value to an observatory code.

        Two cases occur in real data. Either the column enumerates telescope
        indices that name telescopes listed in the descriptor (``nm0083``), or
        the column holds observatory codes directly (``nm0019``), in which case
        the mapping is the identity.

        Args:
            format_columns: Parsed Format section.
            telescopes: Parsed telescope list, if any.

        Returns:
            Data column value (as a string) to observatory code. Empty when no
            suitable column exists.
        """
        if not isinstance(telescopes, list):
            return {}

        selector: str | None = None
        for index in sorted(format_columns):
            label = format_columns[index]
            lowered = label.lower()
            if "telescope" in lowered or "observatory" in lowered:
                selector = label
                break
        if selector is None:
            return {}

        lowered = selector.lower()

        # The column already holds observatory codes.
        if "observatory" in lowered and "telescope" not in lowered:
            return {code: int(code) for code in re.findall(r"\b(\d{3})\b", selector)}

        # The column holds telescope indices; resolve each through its name.
        mapping_text = selector.split(":", 1)[1] if ":" in selector else selector
        name_to_code = {
            telescope["name"]: telescope.get("code")
            for telescope in telescopes
            if telescope.get("name")
        }
        index_to_code: dict[str, int | None] = {}
        for part in mapping_text.split(","):
            entry = part.strip()
            match = re.match(r"^(\d+)\s*-\s*(.+)$", entry)
            if match is None:
                # A label with no colon puts its own leading words in front of
                # the first entry - ``Telescope (1 - P26, 2 - PNA)`` - so drop
                # the opening bracket and retry.  Continuation fragments such as
                # ``D/f=0.4/5.5m`` have no bracket and stay unmatched, which is
                # what keeps them out of the mapping.
                entry = re.sub(r"^[^\d]*\(", "", entry, count=1)
                match = re.match(r"^(\d+)\s*-\s*(.+)$", entry)
            if match:
                # The final entry of a comma-separated list is often followed by
                # the closing parenthesis of the column label - ``..., 4 - Z600)``
                # - so strip trailing punctuation before resolving the name.
                name = match.group(2).strip().rstrip(").,;:")
                index_to_code[match.group(1)] = _resolve_telescope_code(name_to_code, name)
        return index_to_code

    @staticmethod
    def fuse_observables(
        observation_type: str | list[str] | None,
        coordinates: str | list[str] | None,
        receptor: str | list[str] | None,
    ) -> str | None:
        """Build the fused observation type the factories are registered under.

        ``absolute`` + ``absolute`` + ``CCD`` becomes
        ``absolute_absolute_CCD_nsdb``. Spaces in the coordinate kind become
        underscores. This string is a load-bearing contract: it is the key in
        the dataset-factory registry, and all 82 existing generated configs use
        it, including the source's own spelling mistakes (``photograpihc``,
        ``no inf``), which are preserved rather than corrected.

        Args:
            observation_type: The ``type`` field, e.g. ``absolute``.
            coordinates: The ``coordinates`` field, e.g. ``X,Y``.
            receptor: The ``receptor`` field, e.g. ``CCD``.

        Returns:
            The fused type, or ``None`` if any component is missing.
        """
        kind = _as_scalar(observation_type)
        coordinate_kind = _as_scalar(coordinates)
        detector = _as_scalar(receptor)
        if not kind or not detector:
            return None
        coordinate_kind = (coordinate_kind or "").replace(" ", "_")
        return f"{kind}_{coordinate_kind}_{detector}_nsdb"

    #: Informations keys that map onto config names. ``telescope`` and
    #: ``relative to`` are handled separately because they are structured.
    INFORMATION_FIELDS: dict[str, str] = {
        "reference frame": "reference_frame",
        "center of frame": "center_of_frame",
        "centre of frame": "centre_of_frame",
        "epoch of equinox": "epoch_of_equinox",
        "time scale": "time_scale",
        "reduction": "reduction",
        "coordinates": "coordinates",
        "diff. refraction": "diff_refraction",
        "receptor": "receptor",
        "observers": "observers",
        "data included in standard data file": "data_included_in_standard_data_file",
    }

    @classmethod
    def parse_informations(cls, block: dict[str, str | list[str]]) -> dict[str, Any]:
        """Map an Informations block onto its config names.

        Args:
            block: Output of :meth:`parse_key_value_block`.

        Returns:
            Config-name keyed mapping containing ``relative_to`` when present.
        """
        parameters: dict[str, Any] = {}
        for key, value in block.items():
            normalised = re.sub(r"\s+", " ", key.strip().lower())

            if normalised == "relative to":
                designation, _, count_text = (_as_scalar(value) or "").partition(":")
                parameters["relative_to"] = {
                    "designation": designation.strip(),
                    "number_of_observations": _first_int(count_text) if count_text else None,
                }
                continue

            field_name = cls.INFORMATION_FIELDS.get(normalised)
            if field_name is not None:
                parameters[field_name] = _as_scalar(value)
                continue

            logger.debug("Ignoring unrecognised Informations key %r.", key)

        # A "centre of frame" naming is exposed under both spellings.
        if "centre_of_frame" in parameters and "center_of_frame" not in parameters:
            parameters["center_of_frame"] = parameters["centre_of_frame"]
        return parameters

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pre_text(html: str | bytes) -> str:
        """Extract the descriptor text from an HTML document.

        Args:
            html: Full HTML, or the already-extracted ``<pre>`` text.

        Returns:
            The text of the ``<pre>`` block, or the whole document's text when
            there is no ``<pre>``.

        Raises:
            ValueError: If a ``<pre>`` block is present but empty.
        """
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")

        lowered = html.lower()
        if "<pre" not in lowered and "<html" not in lowered and "<head" not in lowered:
            # Already the bare descriptor text.
            return html

        soup = bs4.BeautifulSoup(html, "html.parser")
        node = soup.find("pre")
        if node is None:
            return soup.get_text("\n")
        # BeautifulSoup joins a <pre> block's lines, so ask for them back.
        text = node.get_text("\n")
        if not text.strip():
            raise ValueError("The <pre> block of the descriptor is empty.")
        return text

    @classmethod
    def recover_informations(
        cls, sections: dict[str, str]
    ) -> tuple[dict[str, str | list[str]], str | None]:
        """Find an Informations block that is missing its heading.

        ``nm0036`` publishes its astrometric metadata with no ``Informations.``
        heading at all — the fields simply follow the ``References.`` block.
        The block is identifiable by its key vocabulary rather than by position,
        so this scans the other sections for keys that only occur in
        Informations and returns them.

        The keys are specific enough for this to be safe: no bibliography entry
        is of the form ``reference frame: astrometric``. Note that a citation
        line may still contain a colon (a URL, for instance), but such lines do
        not begin with a ``[\w .]+`` key, so they are ignored by the tokeniser.

        Args:
            sections: Output of :meth:`split_sections`.

        Returns:
            A tuple of (recovered block, name of the section it came from). The
            block is empty when nothing was recovered.
        """
        known = set(cls.INFORMATION_FIELDS) | {"relative to", "telescope"}

        for name, text in sections.items():
            if name == "informations":
                continue
            block = cls.parse_key_value_block(text)
            recovered = {
                key: value
                for key, value in block.items()
                if re.sub(r"\s+", " ", key.strip().lower()) in known
            }
            if recovered:
                return recovered, name
        return {}, None

    def parse(self, html: str | bytes, identifier: str | None = None) -> NSDBRecord:
        """Parse a descriptor page into a record.

        Args:
            html: The descriptor HTML, or its ``<pre>`` text.
            identifier: Dataset name. When omitted, the document's ``<title>``
                is used, falling back to ``"unknown"``.

        Returns:
            The parsed :class:`NSDBRecord`.

        Raises:
            ValueError: If the ``Contents.`` or ``Format.`` section is missing,
                or the Format section declares no columns. Those make the
                record unusable, so failing loudly is better than emitting a
                config that cannot be read.
        """
        text = self.extract_pre_text(html)
        sections = self.split_sections(text)

        if identifier is None:
            identifier = self._title_of(html) or "unknown"

        if "contents" not in sections:
            raise ValueError(
                f"Descriptor for {identifier!r} has no Contents. section; found {sorted(sections)}."
            )
        if "format" not in sections:
            raise ValueError(
                f"Descriptor for {identifier!r} has no Format. section; found {sorted(sections)}."
            )

        if "informations" not in sections:
            # A few datasets omit the heading. Try to recover the block from
            # wherever it actually landed before giving up on the metadata.
            informations_block, source = self.recover_informations(sections)
            if informations_block:
                logger.warning(
                    "Descriptor for %s has no Informations. heading; recovered "
                    "the astrometric metadata from the %s. section instead.",
                    identifier,
                    source,
                )
            else:
                logger.warning(
                    "Descriptor for %s has no Informations. section and none "
                    "could be recovered; astrometric metadata (frame, equinox, "
                    "time scale) will be missing.",
                    identifier,
                )
        else:
            informations_block = self.parse_key_value_block(sections["informations"])

        contents_block = self.parse_key_value_block(sections["contents"])
        format_columns = self.parse_format_block(sections["format"])
        if not format_columns:
            raise ValueError(f"Format section for {identifier!r} declares no numbered columns.")

        parameters = self.parse_informations(informations_block)
        observatories, unmatched = self.parse_observatory_entries(contents_block.get("observatory"))
        telescopes = self.parse_telescopes(informations_block.get("telescope"))
        if unmatched:
            logger.warning(
                "Descriptor for %s: %d observatory line(s) did not match the "
                "'NNN - Name' form and were dropped: %s",
                identifier,
                len(unmatched),
                unmatched[:3],
            )

        return NSDBRecord(
            identifier=identifier,
            sections=sections,
            contents=contents_block,
            informations=informations_block,
            format_columns=format_columns,
            planet=self.parse_planet(contents_block.get("planet")),
            satellites=self.parse_satellites(contents_block.get("satellites")),
            total_number=_first_int(_as_scalar(contents_block.get("total number")) or ""),
            dates=self.parse_dates(contents_block.get("dates")),
            observatories=observatories,
            telescopes=telescopes,
            telescope_index=self.parse_telescope_index(format_columns, telescopes),
            observables=self.fuse_observables(
                contents_block.get("type"),
                parameters.get("coordinates"),
                parameters.get("receptor"),
            ),
            parameters=parameters,
            reference=sections.get("reference") or None,
            comments=sections.get("comments") or None,
            unparsed_observatory_lines=unmatched,
        )

    def parse_file(self, path: str | Path, identifier: str | None = None) -> NSDBRecord:
        """Parse a descriptor file from disk.

        The file is read as UTF-8 with ``errors="replace"`` and replacement
        characters turned into question marks, because the descriptors carry
        legacy Latin-1 author names.

        Args:
            path: Path to the ``.html`` descriptor.
            identifier: Dataset name. Defaults to the file's stem.

        Returns:
            The parsed :class:`NSDBRecord`.
        """
        path = Path(path)
        raw = path.read_bytes().decode("utf-8", errors="replace").replace("\ufffd", "?")
        return self.parse(raw, identifier=identifier or path.stem)

    @staticmethod
    def _title_of(html: str | bytes) -> str | None:
        """Return the document's ``<title>`` text, if any."""
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        title = bs4.BeautifulSoup(match.group(1), "html.parser").get_text(strip=True)
        return title or None
