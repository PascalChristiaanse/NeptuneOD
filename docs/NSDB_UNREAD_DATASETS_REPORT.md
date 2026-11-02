# NSDB Unread Datasets: What It Would Take to Recover Them

**Status:** planning document — nothing here is implemented.
**Date:** 2026-11-02
**Scope:** all four corpora on disk (Neptune, Uranus, Saturn, Jupiter).
**Method:** every dataset run through `NSDBSource.frame()` + `verify(strict=False)`; failures bucketed by message and root cause, then each root cause verified against the raw files before being written up.

---

## 1. Where we are

| Corpus | Path | Valid | Failed | Observation rows still unread |
|---|---|---|---|---|
| Neptune | `data/raw` | 80/83 (96%) | 3 | 10 |
| Uranus | `data/uranus_raw` | 80/122 (66%) | 42 | 5,256 |
| Saturn | `data/saturn_raw` | 65/88 (74%) | 23 | **65,773** |
| Jupiter | `data/jupiter_raw` | 48/73 (66%) | 25 | 6,120 |
| **Total** | | **273/366 (75%)** | **93** | **77,159** |

140,319 observation rows currently reach a valid standard frame. The 93 failures would add roughly **77,000 more — a 55% increase in available data**, and the majority of it is Saturnian mutual-event and astrometric data rather than a long tail of oddities.

The failures are not 93 separate problems. They collapse into **15 work items**, of which **four cover 84% of the unread rows**.

---

## 2. Priority

Items 1–15 are ranked by observation rows recovered, since that is what the fit consumes. Effort is a rough estimate for someone already familiar with this reader. **Item 0 is out of order deliberately** — it recovers no rows at all, but it corrupts rows that already load.

| # | Work item | Datasets | Rows | Effort | Risk |
|---|---|---|---|---|---|
| 0 | **Sentinel declaration mismatch (live bug)** | 21 | *1,374 bad values* | Low | **Critical — silent** |
| 1 | **Fixed-width file reader** | 11 | 52,632 | High | Low — self-verifying |
| 2 | **Telescope index: derive from star catalogue** | 1 | 6,487 | Low | Medium |
| 3 | **Angular unit vocabulary** | 9 | 5,200 | Low | **High** — silent-error class |
| 4 | **`Grinwich` → `Greenwich` typo** | 9 | ~500 | Trivial | None |
| 5 | Required column null (source gaps) | 10 | 4,182 | Low | Low |
| 6 | Value out of range | 8 | 2,390 | Medium | **High** — silent-error class |
| 7 | Unrecognised coordinate labels | 12 | 2,231 | Medium | Low |
| 8 | Descriptor omits a column (one-column shift) | 3 | 510 | Low | Low |
| 9 | Local mean time with no meridian | 1 | 1 | Medium | Low |
| 10 | Duplicate Format headings | 5 | 727 | Medium | Medium |
| 11 | Relative pair columns not found | 16 | 652 | Medium | Low |
| 12 | RA/Dec columns not found | 1 | 209 | Low | Low |
| 13 | Epoch columns not inferrable | 3 | 120 | Low | Low |
| 14 | Format section in a foreign layout | 1 | 0 | Trivial | None |
| 15 | Duplicate key in an Informations block | 1 | 0 | Low | Low |

**Item 0 is not a recovery item — it is a bug fix.** It does not add a single new row; it stops **1,374 invalid values from being fitted as real measurements** in 21 datasets that pass validation today. It is ranked first because corrupting data that already looks good is worse than failing to read data that obviously does not. Details in §3.6.

**Items 1–4 alone recover ~64,800 rows — 84% of the total — for four pieces of work.** That is the whole argument for prioritising in this order: the effort is not proportional to the value, because one dataset family (Saturn's `sm0034` group) happens to be enormous.

---

## 3. Work items in detail

### 3.1 Fixed-width file reader — 11 datasets, 52,632 rows ⭐ highest value

**This is the single largest opportunity and it is not a data-quality problem. It is a missing format.**

Eleven datasets declare their layout as **byte ranges** rather than as a list of whitespace-separated columns. The current reader splits on whitespace (`sep=r"\s+"`), which cannot work for these files: the fields are positional, adjacent fields touch, and blank fields are simply absent.

Three sub-dialects carry the same information:

**(a) NSDB prose with byte ranges** — `sm0034`, `sm0035`, `sm0036` (52,538 rows)

```
Format.
  1. Column 1- 3. Opposition number (1 - 1874, 112 - 1989)
  2. Column 5- 8. Year of observation
  3. Column 10-11. Month of observation
  4. Column 13-21. UTC date of observation (DD.dddddd)
  ...
  8. Column 35. Observation type (see coordinates:)
  9. Column 37-38. Subject satellite, reference satellite
 10. Column 40. Presence flag for first observation of pair
 11. Column 41. Presence flag for second observation of pair
 12. Column 42-53. First observation of pair
 13. Column 54-65. Second observation of pair
```

Verified: these three files are **perfectly fixed-width** — every line is exactly 79 characters, and the declared last byte is 79. The byte mapping is exact:

```
        0000000001111111111222222222233333333334444444444555555555566666666667777777777
        1234567890123456789012345678901234567890123456789012345678901234567890123456789
  raw      1 1874  7 15.257084 -2.7 787  1 3 70 01    0.000000   98.883003 1 0  0.0 -0.9

   opposition  bytes  1- 3: 1
   year        bytes  5- 8: 1874
   month       bytes 10-11: 7
   UTC date    bytes 13-21: 15.257084
   type        bytes 35-35: 3
   obs1        bytes 42-53: 0.000000
   obs2        bytes 54-65: 98.883003
```

Note it declares **17 columns but the data has 16 whitespace fields** — because bytes 67 and 69 (equinox, centre) are single-character codes that merge into neighbouring tokens.

**(b) CDS/VizieR byte tables** — `sm0020`, `sm0021`, `sm0022`, `sm0041`, `sm0045`, `sm0044`, `um0024` (4,151 rows)

```
Format.
Byte-by-byte Description of file: *dat.txt
------------------------------------------------------------------------------
   Bytes Format Units  Label   Explanations
------------------------------------------------------------------------------
   1-  3  I3    ---    Opp     Opposition year code number (105=1989, 111=1988)
   5-  8  I4    yr     Obs.Y   Year of observation
  10- 11  I2    ---    Obs.M   Month of observation
  13- 21  F9.6  d      Obs.D   UTC date of observation (DD.dddddd)
```

This is strictly *more* information than the prose form — it gives the FORTRAN format (`I3`, `F9.6`), the unit, and a label. The parser currently rejects it because it looks for `N. description` lines and finds none.

**(c) FORTRAN format string** — `sm0039` (11,408 rows)

```
Format.
 In a FORTRAN code, one line is read with the format:
 (i3,i5,i3,f11.7,f7.3,2i4,i2,1x,a2,a1,i2,i1,
 2(1x,f13.7),2i2,2(1x,f7.3),3i3,2f10.3)
```

Here the byte offsets must be **computed** by accumulating the format descriptors. The file is uniform-width (123 chars), so the arithmetic is checkable.

#### What implementing it requires

1. **Detect the dialect.** Three new `parse_format_block` strategies:
   - `Column A- B.` / `Column N.` → prose byte ranges
   - `Bytes Format Units Label` header line → CDS table
   - `(i3,i5,...)` → FORTRAN format string
2. **Represent the layout.** `format_columns` is currently `dict[int, str]` (1-based index → name). These files need a *span*: `int → (start_byte, end_byte | None, name)`. This is a real interface change — `apply_format_columns` and everything downstream keys off ordinal position, not byte position. Cleanest option is a parallel `format_spans` field on `NSDBRecord`, leaving the existing field untouched for whitespace files.
3. **Read by slice.** `pd.read_fwf(io.StringIO(text), colspecs=[(a-1, b), ...])`, or manual slicing. `read_fwf` handles the ragged cases natively.
4. **Multi-mode rows.** These files encode the *coordinate kind per row* in a type column. `sm0034` uses all four values:
   - type `0` = absolute (α, δ) — 5,459 rows
   - type `1` = X, Y — 2,462 rows
   - type `2` = differential (Δα, Δδ) — 927 rows
   - type `3` = position angle, separation — 17,297 rows

   Plus presence flags at bytes 40/41 (`01`/`10`/`11`) marking which half of a pair was observed. **This requires the formatter to emit a frame with mixed `coordinate_kind` per row**, which it currently cannot do — `format()` classifies once for the whole dataset. This is the genuinely new architectural piece, and it is why the estimate is "High" rather than "a few hours".

   The good news: the schema already carries `first_unit`/`second_unit` per row, so the *output* shape is already able to express this. What is missing is the classification step accepting a per-row kind.

   Note the three files are near-identical layouts. `sm0036` uses only types 1/2/3 (20,686 rows) and `sm0035` only type 0 (5,707 rows), so a working multi-mode reader handles all three.

**Verification strategy:** the byte ranges are self-checking. Assert that every line's length equals the declared maximum byte column; assert that numeric fields parse in the declared FORTRAN format; cross-check `ET-UTC` (bytes 23-26) against the existing time-scale machinery.

---

### 3.2 Telescope index derived from the star catalogue — 1 dataset, 6,487 rows

`sm0084` (6,487 rows) declares:

```
   5. Telescope (1 - P26, 2 - PNA, 3 - ZDA)
telescope: 1 - Pulkovo 26-inch refractor (D=0.65m, F=10.413m),
           2 - Pulkovo normal astrograph (D=0.33m, F=3.463m),
           3 - Abastumani Zeiss Double Astrograph (D=0.4m, F=3.024m)
```

The Format column uses **acronyms** (`P26`, `PNA`, `ZDA`) that share no prefix with the full names, so prefix matching correctly refuses (`_resolve_telescope_code` returns `None`). This was left deliberately unresolved rather than guessed.

**The acronyms are decodable, but not from this descriptor.** `P26` = Pulkovo 26-inch, `PNA` = Pulkovo Normal Astrograph, `ZDA` = Zeiss Double Astrograph — the initials match if you take the *significant* words rather than a strict prefix. Two viable approaches:

- **Bespoke mapping** in the descriptor for this one dataset. Honest and safe, but doesn't generalise.
- **Fuzzy initial-match** (`P26` → first letters of significant words + a number). This is the kind of heuristic that will silently mis-resolve a *different* dataset later, so it would need to be opt-in per dataset.

**Recommendation:** treat as a one-off. Add an explicit override so the mapping is data, not a heuristic. 6,487 rows for a hand-written dict is a good trade; a general acronym matcher is how you get a wrong observatory with no error message.

**Good news on the station positions:** the descriptor already supplies both host observatories — `[{code: 84, 'Pulkovo'}, {code: 119, 'Abastuman'}]` — with `centre: topocentre`. So once the index resolves, the existing per-observatory grouping and `add_observatory_to_SOB` path should work unchanged. No additional metadata is needed.

---

### 3.3 Angular unit vocabulary — 9 datasets, 5,200 rows ⚠️ silent-error class

`relative_position_scale()` infers a column's angular unit from its **name**, and raises rather than guessing (correctly — a wrong unit is a large astrometric error). Nine datasets use unit words the vocabulary does not cover:

| Dataset | Column label | Unit actually meant |
|---|---|---|
| `um0037` | `delta of right ascension (s)` | **seconds of time** |
| `um0083` | `S (separation) in revolutions of the micrometer screw` | **micrometre revolutions** |
| `um0134` | `... in the sense Sat 1 minus Sat 2, mas` | milliarcseconds |
| `sm0002` | `X, minutes of arc` | arcminutes |
| `sm0055` | `Observation type (1 = X, Y), Type.` | *(column is a label, not a measurement — see below)* |
| `sm0069` | `Minutes of arc of separation from center of Saturn` | arcminutes |
| `jg0007` | `Type of coordinate (10 - apparent distance,` | *(label column)* |
| `jg0008` | *(same family)* | |
| `jg0009` | *(same family)* | |

**Two sub-cases, and they need different treatment:**

1. **Genuinely missing units.** `mas`, `minutes of arc`, `s` (seconds of time), and `revolutions of the micrometer screw`. The first three are trivial additions to `relative_position_scale` — microseconds of arc, arcminutes, and seconds-of-time all have existing constants (`SECONDS_OF_TIME_TO_RADIANS`, `ARCSECOND_TO_RADIANS`). The micrometre-screw case needs a **per-dataset calibration factor** (revolutions → arcsec) that is not in the descriptor; that one is probably not worth recovering.

   **Risk note:** `minutes of arc` must NOT be matched by a loose `min` substring test, or `minutes of arc`, `minute of RA` and `Minimum` all collide. Match on the full phrase.

2. **The wrong column is being selected.** For `sm0055`/`jg0007-0009`, the finder is landing on a *type/label* column (`Observation type (1 = X, Y)`) instead of a measurement column. That is a **column-selection bug of the same family as the `O-C` residual bug fixed this session** — the label happens to contain an `X, Y` token. These datasets are not really a units problem at all and should be re-bucketed once the finder is tightened.

---

### 3.4 `Grinwich` → `Greenwich` — 9 datasets, ~500 rows

Nine Uranian datasets write the time scale as **`Grinwich Mean Time`** (missing `e`). `time_scales.py` does not recognise it, falls through to "local mean time", and then correctly refuses because no meridian is named.

**Verified:** passing `time_scale="Greenwich Mean Time"` explicitly makes **all nine** parse. This is a one-character source typo, fixable in the same place as `photograpihc` (already handled by `_RECEPTOR_LABELS`), and it is the cheapest 500 rows in the document.

One caveat: `um0073` also has `observatory: []` and `comments: no information`, so even after the typo fix it may still lack a station position for topocentric correction. Check whether `centre: topocentre` with no station longitude is acceptable for these — if not, `um0073` moves to item 3.9.

---

### 3.5 Required column null — 10 datasets, 4,182 rows

`second_value` (or `epoch_tdb`) is null for a handful of rows. Three distinct causes:

| Cause | Datasets | Detail |
|---|---|---|
| **Malformed source value** | `sm0006`, `sm0037`, `sm0038` | Declination seconds written as `33,36` (comma pair) and `.00.82` / `.20.58` (double decimal point). `pd.to_numeric` coerces to NaN. |
| **Genuine gap** | `nm0005`, `um0030`, `sm0079`, `um0018`, `um0019`, `jg0029` | 1–3 rows each where the source omitted a value. |
| **Whole column empty** | `jg0011` (1,314 rows), `jg0074` | See item 3.8 — a column-shift defect, not a data gap. |

**For the malformed values** (`33,36`, `.00.82`): these are unambiguous once seen — the comma is a delimiter that leaked into a numeric field, and the double dot is a stray keystroke. But "unambiguous once seen" is exactly the reasoning that produces silent errors, so any repair should be **per-pattern and logged**, not a general "strip unexpected characters" pass. Recommended: attempt the repair, log a warning naming the dataset and row, and keep the raw value in the frame for audit.

**For genuine gaps** (1–3 rows in otherwise-good datasets): the cheapest correct answer is to **drop the offending rows** rather than fail the dataset. 4,182 rows are currently lost to ~15 bad ones. This is a policy decision, and the schema already has the mechanism — `NOT_GIVEN` for the paired-value columns. The blocker is that `second_value` is non-nullable for `ra_dec`, so either that constraint relaxes or the rows drop. **Dropping is safer**: a null epoch cannot be fitted, and a null declination is not a declination.

---

### 3.6 Value out of range — 8 datasets, 2,390 rows ⚠️ silent-error class

A range check fires, which is the system working. Two sub-cases:

**(a) Position angle and separation swapped or mis-scaled** — `um0051`, `um0052`, `um0056`, `um0057`

Separation values are ~`-4.8e-05` rad (about −10 arcsec), i.e. **negative**, and the check correctly rejects them for `[0, inf]`. Looking at `um0053`'s actual data:

```
P (position angle) in degrees (999.9 if P is not given):   min  2.89, median 680.49, max 999.99
S (separation) in arcsec (-99.99 if S is not given):       min -9.99, median   0.43, max  75.82
```

**Investigated: this is a real bug, and it is a sentinel-declaration mismatch.**

The descriptor declares the sentinel as `999.9`, but the data writes **`999.99`** — an extra digit:

```
P (position angle) in degrees (999.9 if P is not given)   ← descriptor
 4  3  1904  7 13  17 38 53   999.99   75.73             ← data
```

Counts: the P column holds **148** values of exactly `999.99` (out of 296 rows — half the dataset), while the S column's `-99.99` matches its own declaration exactly and is handled correctly. So the sentinel mechanism works; the *declared number* is one digit short of what the file contains.

This is the same failure shape as the epoch-offset bug found earlier: the descriptor is authoritative in principle, but here it is simply written wrong. **`_declared_sentinel` should not blindly trust the declaration** — a value in the column that agrees with the declaration to a different number of trailing digits is far more likely to be the real sentinel than a genuine measurement. Recommended fix: match the declared sentinel *and* its one-extra-digit variants, logging which was used.

Note `um0053` also has a genuine PA of `360.98°` — a legitimate wrap. Position angles should reduce modulo 2π regardless.

#### This bug is not confined to the failing datasets

The same mismatch occurs in **21 datasets, leaking 1,374 sentinel values**, most of which are *currently passing validation*:

| Corpus | Datasets | Leaked values |
|---|---|---|
| Neptune | `nm0021`, `nm0027`, `nm0071`, `nm0073`, `nm0075`, `nm0076` | 236 |
| Uranus | `um0043`, `um0048`, `um0052`, `um0053`, `um0061`, `um0062`, `um0064`–`um0073`, `um0078` | 1,138 |

Worst offenders: `um0061` (339), `um0062` (173), `um0053` (148), `nm0076` (147), `um0070` (140).

Verified on `nm0076`, which **passes `verify()` today**:

```
nm0076: VALID = True, n = 223
first_value max = 9999.0000 rad  (= 572,900 degrees)
values above 2π (impossible for a position angle): 147
```

The schema's range check covers `ra_dec` coordinates but not `position_angle_separation`, so these pass. **Every one of those 1,374 values is a "P is not given" that has been read as a real position angle.** This is a silent-error bug in shipped code, and it should be fixed independently of any of the work items above — it is arguably the most valuable single finding in this document.

Note `um0053` also has one genuine PA of `360.98°` — a legitimate wrap. Position angles should reduce modulo 2π.

**(b) RA in the wrong unit** — `nm0001`, `nm0005`, `um0005`, `sm0058`

`first_value` lands at 11–15 rad where RA must be < 2π. These are **glued-field** cases (`nm0001`/`nm0005`, see 3.8) and single-row outliers (`sm0058`: one row of 1,340). `um0005`: the RA hour and declination degree columns have merged.

---

### 3.7 Unrecognised coordinate labels — 12 datasets, 2,231 rows

The classifier has a fixed vocabulary. Twelve datasets use labels it does not cover — but they split into four groups, only two of which are real vocabulary gaps:

| Group | Datasets | Resolution |
|---|---|---|
| **Abbreviation only** | `um0038`, `um0041` (`Pos. angle, distance`), `um0117` (`Delta(R.A.), Delta(Declination)`), `sm0035` (`0 - absolute (alpha, delta),`) | Add `pos. angle` and `Delta(R.A.)`/`Delta(Decl.)` to the patterns. `sm0035`'s label is an *enumerated* form — see below. |
| **Typo** | `um0086` (`differences in rigth ascension and declination`) | `rigth`. Trivially fixable. |
| **Unknown / genuinely vague** | `um0103` (`???`) | Not fixable. The columns are labelled `delta(Alpha), arcsec ???` and `delta(Delta), arcsec ???` — the *data* is usable but the descriptor admits it doesn't know. Requires human judgement. |
| **New quantity type** | `jg0050`, `jg0051`, `jg0052`, `jg0061`, `jg0062`, `um0134` | **Not coordinate pairs at all.** See below. |

**The enumerated-label group matters for item 3.1.** `sm0035` declares:

```
coordinates: 0 - absolute (alpha, delta),
             1 - x, y,
             2 - differential (delta alpha, delta delta),
             3 - position angle, separation,
```

That is *not* an unrecognised label — it is a **per-row coordinate kind table**, exactly the multi-mode structure from §3.1. `sm0034`, `sm0035`, `sm0036` all share it. Treating it as a vocabulary gap would be wrong; it needs the type-column reader.

**The new-quantity group** (`Central Instant of Mutual approximations`, `Impact parameter and the moment of mutual approximation`) records *when two satellites appear closest together* and *how close* — timing and separation of a mutual event, not sky position. These cannot be expressed as an RA/Dec pair and **should not be forced into the schema**. They are legitimate exclusions unless the fit is ever extended to use mutual-event timing.

---

### 3.8 Descriptor omits a column — 3 datasets, 510 rows

`um0127`, `um0128`, `jg0045` declare 10 columns:

```
1. Number of satellite
2. Year of the moment of observation      ← actually holds JD (2455075.37761)
3. Month of the moment of observation     ← actually holds the year (2009)
4. Day of the moment of observation       ← actually holds the month (8)
```

The raw file has **11** whitespace fields:

```
3  2455075.37761  2009  8 31.877610  23 43 25.795  -2 39 13.59
```

The descriptor simply **omits the JD column**, so every name is shifted one position to the left. Verified: renaming column 2 to `JD` yields calendar years matching the descriptor's declared span (um0127: 2008.7–2011.7 vs declared 2008–2011; jg0045: 2013.2–2015.3 vs declared 2013–2015).

**Detection is already possible and cheap:** a column named `Year` whose median exceeds 100,000 is unambiguously a Julian Date. That is a strong, false-positive-free signature. Two options:

- **Repair:** detect the shift, warn loudly, and re-align. Risky — re-aligning assumes the descriptor is wrong and the data is right, which is usually true here but not universally.
- **Diagnose:** detect it and raise a message naming the likely missing column. The operator fixes the descriptor or adds a one-line override.

**Recommendation: diagnose, don't repair.** A one-column shift is exactly the class of error that, repaired automatically, could align a *correct* descriptor wrongly. The detection is valuable; the automatic fix is not.

Note `nm0001`/`nm0005` are the mirror case — the *data* is damaged (two fields glued: `30.45754620` should be `30.457546` + `20`), needing a fixed-width read to recover.

---

### 3.9 Local mean time with no meridian — 1 dataset, 1 row

`nm0051` declares `Cambridge M. T.` with `observatory: Cambridge` (no code) and no longitude. Cambridge, UK vs Cambridge, Massachusetts differ by ~71° of longitude ≈ 4.7 hours. The code correctly refuses.

This one is **not worth automatic resolution.** The dataset is a single observation and the ambiguity is real. Resolve manually if ever needed.

---

### 3.10 Duplicate Format headings — 5 datasets, 727 rows

`um0047`, `um0049`, `um0058`, `um0059`, `um0060`, `sm0044` declare the **same heading text twice** — because the file holds several groups of the same quantity (e.g. three separate time triples, one per satellite). `df[name]` then returns a DataFrame and every later lookup raises an opaque `TypeError`, so `_reject_duplicate_headings` fails loudly instead of guessing.

**Partial auto-resolution is possible and safe in specific shapes:**

- `um0049`, `um0058`, `um0059`, `um0060` duplicate *time components* (`Hour of the observation` ×2) — these are two separate observation epochs for the P and S measurements of the same event. The column layout is regular, and the descriptor's ordering makes the grouping unambiguous.
- `um0047` duplicates `Voyager Picture Identification (PICNO)` — also a regular two-group layout.
- `sm0044` is **a genuine descriptor typo**: columns 11 and 12 are both labelled `Value of X, arcsec.`, but the data (`0.0020605, -0.0307932`) is plainly X and Y. Fixable by hand, and `sm0044` is also a fixed-width file (item 3.1), so it needs both.

**Approach:** disambiguate by appending a positional suffix derived from the group index (`Hour of the observation [P]`), driven by a small per-dataset group-size declaration. Do **not** attempt a general "auto-number duplicates" pass — the guard exists precisely because guessing attaches the right name to the wrong observations.

---

### 3.11 Relative pair columns not found — 16 datasets, 652 rows

Two distinct sub-cases, and only one is a genuine gap:

**(a) Label contradicts the columns** — `um0014`, `um0022`, `um0026`, `um0028`, `um0029`, `um0084`, `um0120`, `jg0026`, `jg0028`, `jg0036`, `jg0039`, `jg0048`, `jg0058`, `jg0064`, `jg0066`, `jg0068`

`um0014` declares `coordinates: X, Y (pos angle, separation)` but the columns are `Pos. angle, degrees` / `Separation, arcsec` — **it is a position-angle pair labelled as X/Y.** The classifier trusts the label, asks for X/Y columns, and finds none.

The classifier already distrusts a `radec` label when X/Y columns are present. The same distrust should extend here: if the label says X/Y but no X/Y columns exist and a PA/separation pair does, **the columns win**. This is a small, well-scoped change to `classify_coordinates`, and it covers 16 datasets for little risk.

**(b) Genuinely incomplete data** — `jg0058`, `jg0064`, `jg0068`, `um0120`

These declare `Position angle (topocentric for mutual occultation...)` with columns `Position angle A in degrees` + `Precision of apparent position along the apparent relative trajectory`. There is **a position angle and a precision, but no separation**. The descriptor even says: *"Flag R assigned showing the reason why only one coordinate was determined"* — the source knows it recorded a partial measurement.

Cannot be recovered as a PA/separation pair. Would need either a new schema shape (angle + uncertainty, no separation) or exclusion. **Recommend exclusion** unless partial mutual-event data is specifically wanted.

---

### 3.12–3.15 Minor items

| Item | Datasets | Rows | Notes |
|---|---|---|---|
| RA/Dec columns not found | `jg0014` | 209 | Its Format section is a mix of `column N.` prose and byte ranges. Falls out of item 3.1. |
| Epoch columns not inferrable | `sm0033`, `sm0072`, `um0020` | 120 | `sm0072` is fixed-width, so item 3.1 may fix it. `sm0033`/`um0020` need inspection — likely a genuine column-naming gap. |
| Format section in a foreign layout | `sm0019` | 0 | Single dataset using `column N : description` with a **colon** instead of a period, plus a `Format:` line — despite the file having no `Format.` heading. Trivial; also the format string is `x,i2,3x,f14.6,...` (FORTRAN), so it belongs with item 3.1. |
| Duplicate key in an Informations block | `jg0072` | 0 | `X57:` and `W95:` appear under both `telescope` and `observers`, because continuation lines list instruments/observers *per observatory code*. The block parser promotes them to field keys and the second occurrence trips the duplicate guard. Fixable by treating an all-caps/digits key as a continuation when it repeats — but the same rule would have to not break legitimate repeated keys (`Washington` appears 6× in `um0062`'s references). Low value (0 rows). |

---

## 4. Cross-cutting notes

### 4.1 The silent-error class is the one to be careful with

Four of the fifteen items — angular units, value-out-of-range, physical shape contradictions, and the fixed-width type column — are **silent-error** risks rather than loud failures. This session found three bugs where a dataset was *passing* while reading the wrong column:

- `sm0026`/`sm0028`: an `O-C` residual read as the declination, ~1000× too small.
- `um0124`/`um0125`: a milliarcsecond *error* read as right ascension.
- `jg0011`: a whole column shifted, giving 15.6 rad of "RA".

All three passed every check until a range validator or an explicit column inspection caught them. **Any work on this list should be validated by asserting on the resulting physical values, not on the row count.** A dataset that starts failing after a change may have been passing wrongly before it.

**The sentinel bug (§3.6) is a fourth instance, and the only one found that is still live in passing datasets**: 1,374 "P is not given" markers read as position angles in 21 datasets, because the schema range-checks `ra_dec` but not `position_angle_separation`, and because the declared sentinel is one digit shorter than the value in the file. It was found only by asking what *values* were in the column, not whether the dataset loaded. If one lesson is taken from this document, it is that the row-count anchors below are necessary but not sufficient.

### 4.2 What is not worth doing

For honesty, these are the items where the recommendation is to leave them:

| | Reason |
|---|---|
| `um0103` (`???` coordinates) | The descriptor does not know what the data is. Not our call. |
| `jg0050/51/61/62`, `um0134` (mutual-event timing) | Not sky-position measurements. Out of schema scope. |
| `um0083` (micrometer screw revolutions) | Needs a calibration constant that is not published in the descriptor. |
| `nm0051` (Cambridge, unnamed) | One row, genuinely ambiguous meridian. |
| `jg0058/64/68/120` (angle with no separation) | Incomplete by the source's own admission. |

That is 8 datasets and ~500 rows out of 93 and 77,159 — so **the recoverable ceiling is roughly 76,600 rows, or 99% of what is currently unread.**

### 4.3 Suggested order of work

1. **Sentinel declaration mismatch** (item 0) — a live bug affecting 21 datasets that currently pass. Fix independently of everything else; it is small and the failure is silent. Then extend the schema's range check to `position_angle_separation` so a recurrence is caught loudly.
2. **`Grinwich` typo** (item 3.4) — minutes of work, ~500 rows, zero risk. Do it next time the file is open.
3. **Angular units + column-selection tightening** (item 3.3, plus the `sm0055`/`jg0007` re-bucketing) — small, and closes a silent-error class.
4. **`classify_coordinates` column-over-label trust** (item 3.11a) — small, 16 datasets.
5. **Fixed-width reader** (item 3.1) — the big one. Start with the CDS dialect (`sm0020`, `sm0041`, `sm0045`, `um0024`) because the byte ranges and FORTRAN formats are given explicitly and the row counts are small, then take on `sm0034`/`sm0036` once the multi-mode row handling is settled. That last step is 44,000 of the 52,632 rows.
6. Everything else in whatever order the fit needs.

### 4.4 How to verify any of it

The out-of-corpus benchmark that produced this document is reproducible and should be kept:

```
NSDBDownloader(root).fetch_available_datasets(planet)   # e.g. "Jupiter"
# for each dataset: NSDBSource(root).frame(name) + verify(strict=False)
# bucket failures by message, then verify each root cause against the raw file
```

Anchors to hold constant while working:

| Corpus | Valid | Rows | Not-given |
|---|---|---|---|
| Neptune `data/raw` | 80/83 | 20,582 | 4,141 |
| Uranus `data/uranus_raw` | 80/122 | 57,156 | 2,197 |
| Saturn `data/saturn_raw` | 65/88 | 43,877 | 0 |
| Jupiter `data/jupiter_raw` | 48/73 | 18,704 | 604 |

The Neptune anchor is the strongest regression signal: its original 78 datasets must remain at **19,935 rows / 4,112 not-given**, and its four recovered datasets (`nm0001`, `nm0011`, `nm0087`, `nm0089`) account for exactly the +647 rows / +29 not-given above that.

---

## 5. Summary

93 datasets and 77,159 observation rows are currently unread. **Four work items recover 84% of them**, and one of those — the fixed-width reader — recovers 68% on its own while being a *missing format* rather than a data-quality problem. Only 8 datasets are genuinely unrecoverable.

Separately, and more urgently: **1,374 sentinel values are currently being fitted as real position angles across 21 datasets that pass validation.** That is a live silent-error bug, not a recovery item, and it should be fixed first regardless of whether any of the work above is ever done.

1,402 tests pass against the current reader; every anchor above is stable.
