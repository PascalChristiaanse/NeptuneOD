# Data / Observations Refactor — Current-State Survey and Fit to the `preprocessing` Proposal

**Status:** findings only. No code changed.
**Branch:** `refactor/data-module` (4 commits ahead of `main`: design doc + two ORM prototypes)
**Scope examined:** `src/orbitdet/data/`, `src/orbitdet/observations/`, `src/orbitdet/data_acquisition/`, `src/orbitdet/transformations.py`, `conf/datasets/**`, `data/raw/**`
**Baseline:** `511 passed` (`NeptuneODWTudat`); `NeptuneOD` env is currently broken independently of this work (`ImportError: create_radiation_pressure_interface`).

---

## 1. Executive summary

The two modules you named do not currently divide work the way the proposed design does:

| | Proposal | Reality today |
|---|---|---|
| `data` | download + preprocess → **one standard dataframe** | Downloaders *and* the `GaiaQuery` → `ObservationCollection` converter *and* astrometric corrections |
| `preprocessing` | pure, reusable formatting/astrometry tooling | **Does not exist**; its intended content lives in `observations/nsdb_helpers.py`, `observations/helpers.py`, and `orbitdet/transformations.py` |
| `observations` | dispatch a standard dataframe into model/set factories | Currently also *does the preprocessing* (reads raw files, infers columns, converts time and frames, builds the standard frame) |

So the proposed split is not a relocation exercise — it is a **phase inversion**. Today the pipeline is

```
raw NSDB file ──► observations/<modality>.py ──► infer columns ──► time → TDB ──► frame → J2000 ──► SingleObservationSet
                        ▲
                        └── orchestrates everything: parsing, inference, time, frames, link ends, models
```

and the proposal wants

```
raw file ──► data (download) ──► preprocessing (format) ──► standard DataFrame ──► observations (dispatch) ──► sets/models
                                        ▲
                                        └── pure functions, no SystemOfBodies, no Tudat
```

**The single most important finding** is that your proposal's standard dataframe is *already half-specified in the codebase* — `src/orbitdet/data_acquisition/datamodels.py` contains a `STANDARD_DATAFRAME_COLUMNS` dataclass whose field names match your list almost exactly. It is dead code: `StandardDataFrameVerifier` has **zero methods**, `STANDARD_DATAFRAME_COLUMNS` requires all 13 fields as constructor arguments and is never instantiated, and nothing imports either. This is the natural home for the schema and should be promoted from dead code to the contract that `preprocessing` emits and `observations` consumes.

**Three findings that change the design, not just the layout:**

1. **The standard dataframe cannot be built without `SystemOfBodies`.** Time conversion is per-observatory (`TimeScaleConverter.convert_time` needs the Earth-fixed station position for UT1/TDB), and the relative X/Y factories need the target's *ephemeris* to invert the gnomonic projection. A "pure dataframe in → dataframe out" preprocessing API therefore cannot cover everything you listed. §5.2 proposes a split: **pure** functions in `preprocessing`, and a thin **context-aware** layer that takes `SystemOfBodies`.

2. **The proposal's `Epoch of equinox`→J2000 step is not needed for most current data, but is needed for the data you actually have.** 48/82 generated configs declare `epoch_of_equinox: date`, and the `Date` frame path is implemented — but only `relative_xy_nsdb.py` passes a `time_column`. `absolute_ccd_nsdb.py` calls `convert_radec_frame(...)` **without** `time_column`, which raises `ValueError: time_column is required when converting to/from the 'Date' frame`. That dataset family is latently broken and no test covers it.

3. **The "Reduction catalogue correction" step you list is the one thing with no implementation at all, and it is currently masked by a bug.** `reduction` is parsed into configs (e.g. `UCAC2`, `Gaia DR2`, `Catalogue Tycho-2`) and read by **nothing** in `src/`. It is not even in your proposed dataframe column list. §5.4 argues it *must* be a column and gives the tie-to-ICRF/CRF3 design.

---

## 2. Inventory: what exists, where it lives, and where it belongs

### 2.1 `data/` — downloaders plus one large exception

| File | Lines | Responsibility | Verdict |
|---|---|---|---|
| `nsdb.py` | 596 | `NSDBManager`: HTTP download, **HTML metadata scraping** (`_parse_contents_metadata`, `_parse_contents`, `_parse_informations`, `_parse_telescope_entries`, `_build_telescope_index`), Hydra YAML emission | Download ✅ stays. **Scraping → `preprocessing/nsdb/`**, because it is "format raw source into a standard shape", not download |
| `kernel.py` | 257 | `KernelManager`: SPICE kernel/data-file download, tar extraction, furnishing | ✅ stays |
| `voyager_data.py` | 186 | Reads two manual CSVs, merges, renames `date_jed`→`jd`, `jd`→`epoch_TDB`, **plus `build_voyager_tabulated_state_history`** (frame rotation + gradients + unit scaling) | Reader/merge → `preprocessing`; state-history builder is a *simulation* concern (consumed by `simulation/environment.py`) → `simulation/` |
| `gaia.py` | 720 | `GaiaQuery`: archive fetch + joblib week cache, **`correct_observations`** (photocenter + light deflection, in-place on the table), **`to_tudat`** (frame rotation, `GaiaAstrometry.to_observation_collection`, scan-angle + error registries, weight caching) | This is the *only* place `data` builds Tudat objects. Fetch/cache ✅ stays; corrections → `preprocessing`; `to_tudat` → `observations/gaia.py` |
| `photocenter.py` | 133 | `photocenter_offset_spherical` (Fuentes-Muñoz 2024) | → `preprocessing/corrections/` |
| `light_deflection.py` | 177 | `relativistic_light_deflection` (Klioner PN) | → `preprocessing/corrections/` |
| `__init__.py` | 17 | Eager-imports **all** of the above, incl. `GaiaQuery` | Eager import of `gaia` is what forces the CI stub in root `conftest.py`. Splitting corrections/`to_tudat` out removes that coupling |

Note: the two `.md`-adjacent "data" modules are pure astrometric *theory*, not acquisition — they are the clearest `preprocessing` candidates in the tree.

### 2.2 `observations/` — the factories, plus two helper modules that *are* preprocessing

**Factory / infrastructure (correctly placed):**

| File | Lines | Role |
|---|---|---|
| `registry.py` | 75 | `_FACTORY_REGISTRY` + `@register_dataset_factory(type)` decorator |
| `factory.py` | 54 | `create_observation_dataset(cfg, dataset_cfg, system_of_bodies)` — dispatcher |
| `collection.py` | 207 | `create_observation_collection`: loops datasets, merges, assigns metadata FIFO indices |
| `configs.py` | 48 | `ObservationDatasetConfig`, `SimulatedObservationConfig` dataclasses (near-unused; configs are `DictConfig`) |
| `utils.py` | 42 | `get_set_identifier` from receiver link end |
| `weighting/` (12 files), `outlier_rejection/` (6), `grouping/` (4) | — | post-processing subsystems, out of scope |

**Modality factories:**

| File | Lines | Delegates preprocessing to | Notes |
|---|---|---|---|
| `absolute_ccd_nsdb.py` | 173 | `set_iso_time_column`, `set_ra_dec_columns`, `convert_radec_frame`, `convert_time_to_seconds_since_j2000_TDB`, `add_observatory_to_SOB`, `resolve_observatory_codes`, `group_rows_by_observatory` | **`convert_radec_frame` called without `time_column`** → latently broken for `Date`/`date` equinox |
| `relative_ccd_nsdb.py` | 172 | same, with `set_relative_position_columns` | + `RELATIVE_ANGULAR_POSITION_TYPE = ObservableType(9)` workaround |
| `relative_xy_nsdb.py` | 403 | same + **`_topocentric_planet_direction` + `_standard_to_relative_radec` + 4 conversion round-trips through `convert_radec_frame`** | The biggest and most reference-frame-heavy factory. The X/Y→RA/Dec gnomonic inversion is *preprocessing*, not model building |
| `gaia.py` | 107 | `GaiaQuery` (from `data`) | Reads `epoch_of_equinox` as the **input frame** (default `ICRS`) |
| `voyager.py` | 79 | `load_and_merge_voyager_tables` (from `data`) + `set_ra_dec_columns` + `convert_radec_frame` | |
| `simulated.py` | 142 | none — generates its own epochs from ISO strings | Genuinely different: no file, no dataframe. See §5.6 |

**The two modules that are entirely preprocessing and are misplaced by name:**

- **`nsdb_helpers.py` (512 lines)** — every function in it is a dataframe formatter:
  `set_iso_time_column` (JD **or** year/month/day/hour/min/sec → ISO), `set_ra_dec_columns` (h/m/s + d/m/s → decimal degrees, with a large exclusion-rule search), `set_relative_position_columns` (Δα/Δδ or X/Y → **radians**, unit sniffed from the column name), `resolve_observatory_codes`, `group_rows_by_observatory`, plus the fuzzy finders `_find_column` / `_find_column_substring` / `_normalize_column_name` / `_numeric_series`.
  This is 512 lines of exactly what you described as `preprocessing` — **and it is imported by `observations/voyager.py`**, i.e. a non-NSDB modality depends on an `nsdb_`-prefixed module.

- **`helpers.py` (202 lines)** — mixed bag:
  `convert_time_to_seconds_since_j2000_TDB` (**timescale conversion**), `get_observatory_info` / `get_observatory_location` / `add_observatory_to_SOB` (**reference data + environment mutation**), `normalize_observatory_code` (identifier normalisation).

### 2.3 `transformations.py` (461 lines) — the existing partial `preprocessing`

Already a standalone module and already the right *idea*, but it is an ad-hoc collection:

| Function | Proposal bucket | Notes |
|---|---|---|
| `convert_radec_frame` | precession + frame tie | Router: SPICE ↔ Astropy ↔ `Date`, with ICRS frame-bias special-casing |
| `_radec_icrs_to_j2000` | frame tie / ICRS↔FK5 | Astropy, ~17–20 mas, constant |
| `_precess_radec_date_to_j2000` / `_precess_radec_j2000_to_date` | **precessions** | Astropy FK5, time-dependent |
| `_radec_spice_convert` | frame tie | SPICE `pxform` |
| `_canonicalize_frame` | frame name normalisation | `ICRS`/`ICRF`→`J2000`, `J2015.0`→`J2000`, `J2000.0`→`J2000` |
| `convert_cartesian_frame` | frame tie | SPICE `pxform`, row-wise `df.apply` (slow) |
| `convert_observation_to_apparent_direction` | — | RA/Dec → unit vectors; **no callers** |

Gaps relative to your list: **no Besselian epochs**, **no `astrometric`/`apparent` handling**, **no stellar aberration**, and `_canonicalize_frame` silently maps `J2015.0`→`J2000` with only a `warning`. Verified failure: frames `'1905.0'`, `'1899.0'`, `'astrometric'` reach SPICE and raise `SpiceUNKNOWNFRAME`.

---

## 3. Directive-by-directive gap analysis

### 3.1 Standard dataframe columns

Your 13 columns map onto existing config fields and Tuples almost perfectly. The `data_acquisition` dataclass is the proof:

| Your column | Config field today | `STANDARD_DATAFRAME_COLUMNS` field | Status |
|---|---|---|---|
| Observed object | `satellites` keys / `planet.name` | `observed_object` | present |
| Observatory code (or name) | `observatory[].code` / `.name` | `observatory_code` | present, **but shape varies** (§4.1) |
| Data source (NSDB + set ID) | *implicit* in filename/`identifier` | `data_source` | **not modelled** |
| Reference (academic) | `reference` | `reference` | present (raw free text) |
| Type (relative, absolute) | encoded in `type` string only | `type` | present as fused string |
| Coordinates (PS, RA/DEC, XY) | `coordinates` + `type` | `coordinates` | present, inconsistent spellings |
| Center of frame | `centre_of_frame` **or** `center_of_frame` | `center_of_frame` | **duplicate keys** (§4.2) |
| Original epoch of equinox | `epoch_of_equinox` | `original_epoch_of_equinox` | present |
| **Original reduction** | `reduction` | `original_reduction` | present in config, **unread in code, absent from your list** |
| Receptor type | `receptor` | `receptor_type` | present, free text (`micrometric`, `photograpihc`, `no inf`) |
| Observation epoch (TDB, s since J2000) | computed `epoch_TDB` | `observation_epoch` | present as ad-hoc column |
| First value (RA, PA, X) | inferred column name | `first_observation_value` | **not modelled** |
| Second value (DEC, Sep, Y) | inferred column name | `second_observation_value` | **not modelled** |

Missing from your list but needed: **`reference_frame`** (`astrometric` ×32 / **`apparent` ×48** / `BCRS` ×2 — this drives the stellar-aberration step you asked for) and **`reduction`**. Both are already in every generated config.

Also missing: the standard frame has **no unit/scale column**. Values arrive as degrees, hours, arcsec, seconds-of-time, and radians; today the unit is smuggled through the inferred *column name* (`_relative_position_scale` sniffs `"arcsec"`/`"sec of time"`/`"degree"` from the header text). A standard frame needs an explicit `first_unit`/`second_unit` or a guaranteed-radian contract.

### 3.2 Preprocessing submodules

| Proposed submodule | Existing implementation | Completeness |
|---|---|---|
| **Time format** (multicolumn, ISO, JD, MJD → s since J2000) | `nsdb_helpers.set_iso_time_column`; `voyager_data` `date_jed`; `simulated.py` via `iso_string_to_epoch` | JD ✅, calendar ✅, ISO ✅. **MJD ✗** — `_find_column(df,"jd")` matches `"MJD of observation"` (substring `jd`), then feeds it to `pd.to_datetime(..., origin="julian")` → `OutOfBoundsDatetime` (verified). No MJD appears in current configs (latent). Precision: routes through an **ISO string**, capping ~1 µs (verified 12 µs error vs exact JD) |
| **Timescale** (WMT, GMT, UTC, UT1, TCB, ET → TDB) | `helpers.convert_time_to_seconds_since_j2000_TDB` | **Handles only `UT1`/`UTC`/`TT`/`TDB`; anything else raises `ValueError`.** 50/82 configs use a non-supported label: `Washington Mean Time`, `WMT`, `Greenwich M. T.`, `G.M.T.`, `GMT=UTC`, `Paris M. T.`, `Pulkovo Mean Time`, `Mt. Hamilton M. T.`, `M.Z.Strasb`, `T.m.de Nice`, `PST`, `CST`, `90 degree time`, `TCB`… The active collection is all `UTC`, so this is latent, but it is **most of your `WMT`/`GMT` requirement, unimplemented**. Fulcrum point: local mean times need the station meridian, and pre-1925 epochs need the noon-to-noon astronomical-day `+12 h` correction |
| **Precession** (mean-of-date, true-of-date, B1xxx/J2xxx → J2000) | `transformations._precess_radec_date_to_j2000` (Astropy FK5) | **Mean-of-date only.** True-of-date ✗. Besselian `B1950` delegated to SPICE (works); `'1899.0'`/`'1905.0'` reach SPICE → `SpiceUNKNOWNFRAME`. `J2015.0` silently → `J2000` |
| **Reduction catalogue** (UCAC, HCRF, USNO → Gaia-CRF3/ICRF) | **none** | Nothing anywhere. `reduction` is parsed and unread. No proper-motion correction anywhere in the tree |
| **Apparent ↔ astrometric** (stellar aberration) | **none** | 48 configs declare `reference_frame: apparent`; no code reads it. `propagation.py:204` passes `aberration_corrections="none"` for *initial states* — unrelated |

---

## 4. Bugs and inconsistencies found (relevant to the refactor)

These are latent — the active `atanas_triton_generated` collection happens to satisfy every precondition. They are exactly the kind of defect the refactor should fix rather than port.

### 4.1 `observatory` is `DictConfig` or `ListConfig` depending on age
60/82 generated configs have a **dict**, 22 have a **list**. `absolute_ccd_nsdb.py` does `dataset_cfg.observatory[0].code`, which raises `ConfigKeyError: Missing key 0` on the dict form. The main collection uses only list-form datasets. `collection.py:_observatory_code` handles *both* — so the inconsistency is known but only patched at one call site.

### 4.2 `centre_of_frame` vs `center_of_frame`
`nsdb.py` deliberately emits **both** spellings, but `absolute_ccd_nsdb.py` reads `dataset_cfg.center_of_frame`, which raises `ConfigAttributeError: Key 'center_of_frame' is not in struct` in struct mode. All 22 active datasets have both (because they also happen to be the 22 list-form ones), so it works by coincidence of regeneration orders.

### 4.3 Stale configs vs. current parser
Regenerating would change behaviour: the current parser emits `observatory` as a **list** and both frame spellings; older configs have the dict form and only `centre_of_frame`. **Config schema and parser output have drifted.** Any refactor must pin the emitted schema and validate configs against it.

### 4.4 `Date`-frame absolute observations are broken
`absolute_ccd_nsdb.py` omits `time_column` in `convert_radec_frame`; `convert_radec_frame` raises for `Date`. 48/82 configs are `date`/`Date`. `relative_xy_nsdb.py` — which needs `Date` too — passes `time_column="epoch_TDB"` and works. No test covers `Date` in the up-link path.

### 4.5 Time conversion is a no-op wrapper that cannot be tested
`convert_time_to_seconds_since_j2000_TDB` takes `system_of_bodies` **only to read a ground-station position**. It also writes `epoch_TDB` in place *and* returns a `Series` (dual API). No unit test exists; the only coverage is end-to-end through factories.

### 4.6 `normalize_observatory_code(-31)` → `'-31'`
`zfill(3)` on a negative yields `-31` (not `0-31`/`−031`). Benign for SPICE because `get_set_identifier` special-cases `code < 0` → body name, but it means the "or name if not available" branch of your spec is implicit, not explicit, in the code.

### 4.7 Duplicated format-column application, three times
Identical `_apply_format_columns` / `_keyfunc` blocks in `absolute_ccd_nsdb.py`, `relative_ccd_nsdb.py`, `relative_xy_nsdb.py` (the last one extracted into a function). Pure copy-paste; belongs in `preprocessing`.

### 4.8 `ObservableType(9)` hard-coded in two factories
`RELATIVE_ANGULAR_POSITION_TYPE = ObservableType(9)` with the comment that this `tudatpy` build mis-binds `relative_angular_position_type`. Duplicated in `relative_ccd_nsdb.py` and `relative_xy_nsdb.py`.

### 4.9 Dead / dangling artifacts
- `STANDARD_DATAFRAME_COLUMNS` + `StandardDataFrameVerifier` — dead (§1).
- `__all__` in `observations/__init__.py` lists **`"relative_xy_radec_nsdb"`** — no such module or import. (`__all__` also omits `grouping.partition_by_sections`' siblings correctly, but the module entry is simply wrong.)
- `convert_observation_to_apparent_direction` — no callers.
- `data/` imports eagerly, which is what forces the CI stub in root `conftest.py`.

---

## 5. Proposed `preprocessing` design, and where the proposal does not fit

### 5.0 Proposed layout

```
src/orbitdet/
├── preprocessing/
│   ├── __init__.py
│   ├── schema.py            # ObservationFrame schema + construction + verify()  (promote STANDARD_DATAFRAME_COLUMNS)
│   ├── columns.py           # _find_column / _normalize_column_name / _numeric_series / apply_format_columns
│   ├── time_formatting.py   # multicolumn | ISO | JD | MJD  ->  s since J2000 (no timescale applied)
│   ├── time_scales.py       # WMT/GMT/UTC/UT1/TCB/ET -> TDB  (station-meridian + astronomical-day aware)
│   ├── precession.py        # mean-of-date | true-of-date | B1xxx | J2xxx -> J2000
│   ├── catalogues.py        # reduction catalogue -> Gaia-CRF3 / ICRF  (+ proper motion)
│   ├── aberration.py        # apparent -> astrometric (stellar aberration)
│   ├── frames.py            # frame-name normalisation + dispatch (absorbs transformations.py router)
│   ├── corrections/         # light deflection + photocenter (moved from data/)
│   ├── nsdb/                # contents/informations scraping + NSDB-specific column mapping
│   └── sources/             # per-source adapters: nsdb, voyager, gaia
└── data/                    # download only: kernel.py, nsdb.download, gaia.fetch, ...
```

All of `observations/nsdb_helpers.py` moves here. `normalize_observatory_code` and the `observatories.txt` lookups move to a new small `preprocessing/observatories.py` (or `reference_data.py`) — they are reference-data + identifier utilities, not modelling.

### 5.1 What does *not* fit: `SystemOfBodies`

Your stated contract is that `data` "*produces a Pandas dataframe as the primary interface*" for `observations`. But two of your own requirements cannot be satisfied by a pure dataframe→dataframe function:

**(a) Timescale conversion needs the station.** `TimeScaleConverter.convert_time(..., earth_fixed_position)` — verified signature default `array([0.,0.,0.])`. For `UT1`/`TDB` the result depends on the observer's Earth-fixed position. `observations/helpers.py` currently fetches it via `system_of_bodies.get("Earth").get_ground_station(code).station_state.cartesian_position_at_reference_epoch`.

Two ways out, and I recommend **both**:
- Accept `(longitude, latitude, altitude)` **or** the Earth-fixed position vector as a plain argument. The observatory *is* known from the standard frame's `observatory_code`, and `observatories.txt` already provides the geodetic triple (`get_observatory_location` currently returns `(lon, lat, alt)` in radians/m). This makes the common case pure and unit-testable.
- For `UT1` specifically, the *correct* full solution needs ΔUT1 = UT1 − UTC from an EOP kernel (`earth_orientation` / `EOP2`, or SPICE `DELTET`), not just the station position. Today's code approximates this via `convert_time`, which is acceptable but should be an explicit, documented approximation in one place.

**(b) Relative X/Y needs the target ephemeris.** `relative_xy_nsdb.py` interprets X/Y as **gnomonic standard coordinates about the topocentric planet direction**, then *inverts* them through `_standard_to_relative_radec` using the planet's topocentric RA/Dec at each epoch, obtained from `system_of_bodies` ephemerides and Earth's rotation model. This is unavoidably context-dependent — but note that **the frame chain is not**: both round-trips through `convert_radec_frame` (`global→Date→global`) cancel, and the function computes *relative spherical offsets in the global frame*. So the gnomonic inverse itself is a clean preprocessing primitive `standard_to_relative_radec(xi, eta, ra0, dec0)`; only `ra0/dec0` need the SOB.

**Recommendation:** make it explicit and explicit-only — `preprocessing` has two layers:

```python
# Layer 1: pure. No Tudat, no SOB, fully unit-testable.
to_tdb(epochs, scale, station_earth_fixed_position=None) -> np.ndarray
precess_to_j2000(ra, dec, equinox, epochs) -> (ra, dec)
tie_to_icrf(ra, dec, pm, pm_epoch, catalogue, epochs) -> (ra, dec)
apply_stellar_aberration(ra, dec, epochs, observer_state, frame) -> (ra, dec)

# Layer 2: context-aware orchestration, still dataframe-in/dataframe-out.
build_observation_frame(dataset_cfg, raw, *, system_of_bodies | (station_position)) -> pd.DataFrame
```

`build_observation_frame` is what `data` calls; `observations/*.py` then sees **only** the standard columns and never touches a raw file. The signature keeps SOB optional so the 22 NSDB `UTC`/`topocentre` datasets — the entire active collection — run through Layer 1 alone.

### 5.2 What does *not* fit: `Epoch of equinox` and `reference_frame` are conflated

Your spec lists `Original Epoch of equinox` as a dataframe column and describes the precession step as equinox→J2000. But two different config fields both describe "which frame is this in":

- `epoch_of_equinox` ∈ {`J2000`, `date`, `B1950`, `J2015.0`, `1899.0`, `1905.0`, `no information`} — the *orientation* (precession).
- `reference_frame` ∈ {`astrometric` ×32, `apparent` ×48, `BCRS` ×2} — the *kind* of coordinate (aberration, and for `apparent`, also light deflection).

`gaia.py` reads `epoch_of_equinox` as the **input frame for `convert_radec_frame`** with default `ICRS` — i.e. it is used as "the frame name", not "the equinox". The two roles must be separated in the schema: keep `original_epoch_of_equinox` **and** add `original_reference_frame`, then have `preprocessing` apply precession and aberration as two independent, individually-testable steps. Applying them as one fused "to J2000" function is what makes the current `transformations.py` router hard to reason about.

`no information` (nm0080) and `J2015.0` (nm0081) then become **explicit, reported fallbacks** in one place instead of scattered `logger.warning`s.

### 5.3 What does *not* fit: `observed_object` and `relative_to` are two different things

Your `Observed object` column is unambiguous for absolute data, but for relative data NSDB carries a *second* object: `relative_to.designation` (`"planet"`, `"8 - Neptune"`, `"N1-Triton"`, or literal `"absolute"`). `relative_ccd_nsdb.py` parses it with `dataset_cfg.relative_to.designation.split(" - ")[1]` — which raises `IndexError` on the literal `"absolute"` present in every absolute config — while `relative_xy_nsdb.py` has a *different*, more robust `_resolve_relative_to_body_name`. Two implementations of one rule.

The standard frame needs **both** columns: `observed_object` and `relative_to_object` (nullable). That also makes "Type (relative, absolute)" derivable rather than a separate free-text field, and it removes the `IndexError` by construction.

### 5.4 What does *not* fit: `reduction` must be in the frame, and its correct home is `catalogues.py`

You listed the catalogue-correction step but omitted `reduction` from the column list, and today nothing reads it. If the tie-to-ICRF step is to run, the frame **must** carry the catalogue identity and the *original catalogue equatorial coordinates + proper motion*, because:

- The correction is coordinate-dependent, not a constant rotation. `UCAC2`→`CRF3`, `Tycho-2`→`CRF3`, `Gaia DR2`→`CRF3` are all different tie fields.
- Proper motion is required: nm0013-style `Catalogue Tycho-2` datasets are from **1900–1950**. Without `μ` and an epoch, every star is displaced by `μ·Δt` — for a Tycho-2 star with μ ~ 50 mas/yr over a century that is **5 arcsec**, i.e. orders of magnitude larger than the mas-level frame-bias effects the rest of the pipeline worries about (cf. the measured light-deflection/photocenter corrections of ~2–8 mas).

So the schema needs, per observation: `original_reduction` (which catalogue), `catalogue_epoch`, `pm_ra`/`pm_dec` (nullable), and — because NSDB "Catalogue given in the data" cases (×3) store them — possibly per-row reminder coordinates.

**Critical caveat:** `reduction` free text is *dirty*: `Catalogue diven in the data` (typo), `Catalogue - Tyc2`, `Catalog SAO`, `1 - Catalogue AST, 2 -Catalogue Tyc-2 (ICRF)`, `positions are corrected for full relativistic aberration but not for relativistic`. `catalogues.py` must normalise to a small enum (`UCAC2|UCAC4|TYCHO2|GAIA_DR1|GAIA_DR2|SAO|AST|NONE|UNKNOWN`) **and** log unmapped values rather than silently passing them through. Note `(ICRF)` in the last example — the source sometimes already states the tie.

Also flag: one `reduction` says *"corrected for full relativistic aberration but not for relativistic [deflection]"*. That is direct evidence that `reference_frame: apparent` vs `astrometric` and the `reduction` text **overlap** and must be reconciled once, at ingest, not independently acted on.

### 5.5 What does *not* fit: time *formatting* and time *scale* are separable, and should be

Your submodule split ("time formatting", "time conversion") is correct and worth enforcing hard. Today they are fused inside `set_iso_time_column` (which produces an ISO string with *no* scale) and `convert_time_to_seconds_since_j2000_TDB` (which consumes the string and applies the scale). Consequences, all verified:

- **Precision is lost to string round-tripping.** `set_iso_time_column` → `pd.Timestamp` (µs resolution, and `pd.to_datetime` on JD loses ~300 µs against Astropy) → `Time.from_iso_string<long double>`. Measured end-to-end error for nm0001: **12 µs**. Acceptable now, but it is a silent cap that blocks mas-level work, and it is an artifact of the fusion.
- **MJD is mis-detected** as JD (substring match) → crash.
- **`MJD`/`JD`/calendar all share one column name `iso_time`**, so the frame cannot express "this is JD UT1" vs "this is ISO TDB".

Proposed hard split:
```python
# time_formatting.py — source → a (calendar or JD/MJD) representation. NO scale.
to_julian_date(raw: pd.DataFrame|pd.Series, *, is_mjd: bool=False) -> pd.Series   # or
to_calendar(raw) -> pd.DataFrame  # year/month/day/hour/min/sec columns
# time_scales.py — representation + scale + station → TDB s since J2000, via DateTime
```
Route to `DateTime(...).to_epoch()` (long double) rather than an ISO string. This also gives a natural home for the astronomical-day rule and meridian offsets: `to_tdb(..., scale="Washington Mean Time", longitude=..., epoch_pre_1925=...)`.

### 5.6 What does *not* fit: `simulated.py` has no dataframe and should not be forced into one

`simulated.py` generates epochs from ISO strings and calls `simulate_observations`. Forcing it through the standard frame would mean inventing an `observatory_code`/`reference`/`reduction` for synthetic data. **Recommendation:** leave `simulated.py` as the one factory that bypasses `preprocessing`, document that explicitly, and have the dispatcher accept either a standard frame or a generator. Ambiguity here is what makes "is this factory a preprocessing consumer or a producer?" hard to answer today.

### 5.7 What does *not* fit: `frames.py` vs `precession.py` boundary

`transformations.convert_radec_frame` is a **router**: it dispatches between (a) Astropy frame-bias, (b) Astropy precession, (c) SPICE `pxform`. If precession moves to its own submodule, the router must stay and *call into* it — otherwise you get two competing `convert_radec_frame`s. Concretely:

- `precession.py` owns `precess(ra, dec, from_equinox, to_equinox, epochs)` — mean-of-date/true-of-date/Besselian.
- `frames.py` keeps the dispatch, delegation, `_canonicalize_frame`, and the `ICRS↔J2000` bias.
- `convert_cartesian_frame` / `convert_radec_frame` remain the public API (they have real callers: 4 factories + `voyager_data`), so this is a *reorganisation behind a stable signature*, not a rename.

I'd also replace the internal `df.apply(..., axis=1)` rotation-matrix loop with a single batched `compute_rotation_matrix_between_frames` per frame pair where epochs allow — a pure performance note, but the refactor is the moment to do it.

### 5.8 Proposed dispatch contract for `observations`

Today's dispatcher keys on a **fused type string** (`absolute_absolute_CCD_nsdb`, `relative_X,_Y_photographic_nsdb`, `relative_Position_angle,_Distance_visual_nsdb`, `absolute_absolute_meridian transit circle_nsdb`…) that encodes *category + coordinates + receptor + source*, including typos (`photograpihc` ×2, `no inf`). The registered factory names must match those strings **exactly**, which is why the commented-out registrations in `relative_ccd_nsdb.py` exist and why several config types have no factory at all.

With a standard frame, dispatch should key on the **columns actually present** plus a small normalised enum:

```python
dispatch_key = (observable_kind, coordinate_kind)   # e.g. ("absolute", "radec"), ("relative", "xy")
```
with `type` reduced to a normalised enum produced by `preprocessing/nsdb/`. That collapses 15+ fused strings onto ~4 registered factories and removes the typo sensitivity. `partition_by_sections`-style grouping (already in `observations/grouping/`) is unaffected.

---

## 6. Suggested first step (when you're ready to implement)

1. **`preprocessing/schema.py`** — revive `STANDARD_DATAFRAME_COLUMNS` as the contract. Decide the two open questions: **units** (recommend: guaranteed radians + explicit `first_unit`/`second_unit` provenance columns) and the **added columns** (`original_reference_frame`, `original_reduction`, `relative_to_object`, `catalogue_epoch`, `pm_*`). Add a real `verify(df)` that raises with every offending column listed.
2. **`preprocessing/columns.py`** — move `_find_column`/`_find_column_substring`/`_normalize_column_name`/`_numeric_series`/`apply_format_columns` verbatim from `nsdb_helpers` + the three duplicated copies. Pure, already tested.
3. **`preprocessing/time_formatting.py`** + **`time_scales.py`** — split `set_iso_time_column` and `convert_time_to_seconds_since_j2000_TDB`; add MJD and the mean-time labels (the 50-config gap) with the station-meridian and pre-1925 astronomical-day rules.
4. **`preprocessing/frames.py`** + **`precession.py`** — re-home `transformations.py` behind its existing signatures, then add Besselian + true-of-date, and make `Date`-without-`time_column` an explicit error at the *builder* level so `absolute_ccd_nsdb` can't hit it (§4.4).
5. Only then touch the factories: each `observations/<modality>.py` becomes "standard frame → link ends + model settings", and the `Date`/`dict`-vs-`list`/`centre`-vs-`center`/`reduction` bugs get fixed once at the boundary.

Steps 1–4 are all pure-function moves with existing or newly-added unit tests; step 5 is where the semantics change and where the `511 passed` baseline should be used as the regression gate.

---

## 7. Open questions for you

1. **Units in the standard frame** — **RESOLVED: radians everywhere.** Radians are the internal TudatPy unit and the pipeline does not deviate from it. Source units are converted on ingest; `first_unit`/`second_unit` preserve provenance. Implemented in `preprocessing/schema.py`.
2. **`reduction` catalogue data source** — for proper motion, do we assume the *original catalogue* coordinates are recoverable from the NSDB file (they are, for the "Catalogue given in the data" cases), or do we require a per-row catalogue lookup (UCAC2/UCAC4/Tycho-2 via Vizier/`astroquery`)? The latter is far more correct but adds a network dependency to preprocessing.
3. **ICRF vs J2000 strictly** — your spec says J2000 and ICRS differ <100 mas but the distinction must be made. Current code applies the ~20 mas frame bias for `ICRS` input and treats everything else as FK5/J2000. Do you want the standard frame to target **ICRF/CRF3** (requiring the bias for FK5 inputs) or **J2000** (requiring it for Gaia/ICRS inputs)? This inverts which direction the bias is applied and should be pinned before `catalogues.py` is written.
4. **`data_acquisition/`'s future** — **RESOLVED: removed.** The two ORMs and the design doc were deleted; `peewee` and `SQLAlchemy` were dropped from `environment.yml`. The schema now lives in `preprocessing/schema.py` and is decoupled from any persistence layer.
5. **`relative_to` for single-object absolute data** — currently the literal string `"absolute"`. Normalise to `None`? **RESOLVED: yes** — the schema uses a nullable `relative_to_object` column, so absolute observations carry null rather than a sentinel string.

---

## 8. Implementation log

### Step 1 — remove the database schemas (done)

- Deleted `src/orbitdet/data_acquisition/` (both ORMs + `data_module_database_design.md`).
- Removed `peewee` and `SQLAlchemy` from `environment.yml`.
- No dangling references remained.

### Step 2 — revive `STANDARD_DATAFRAME_COLUMNS` (done)

New package `src/orbitdet/preprocessing/` with `schema.py` as the dataframe contract.

**`STANDARD_DATAFRAME_COLUMNS`** is a tuple of `ColumnSpec` dataclasses (name, dtype, required, description, unit) — the single source of truth. `COLUMN_SPECS` is an alias. 23 columns: 15 required, 8 optional.

Required (15): `observed_object`, `observatory_code`, `data_source`, `source_set_id`, `reference`, `observation_type`, `coordinate_kind`, `frame_center`, `original_equinox`, `original_reference_frame`, `original_reduction`, `receptor_type`, `epoch_tdb`, `first_value`, `second_value`.

Optional (8): `relative_to_object`, `first_unit`, `second_unit`, `catalogue_epoch`, `pm_first`, `pm_second`, `first_uncertainty`, `second_uncertainty`.

**Columns added beyond the original proposal**, each for a reason established in §3–§5:

| Column | Why |
|---|---|
| `source_set_id` | "Data source (NSDB **with set ID**)" needs the set ID as its own field, not fused into `data_source` |
| `original_reference_frame` | `reference_frame` (astrometric/apparent/BCRS) is orthogonal to `original_equinox`; 48/82 configs are `apparent` and nothing reads it today |
| `relative_to_object` | `observed_object` and `relative_to` are different bodies; replaces the `"absolute"` sentinel and the two divergent parsers |
| `first_unit` / `second_unit` | Provenance for the radian conversion; today the unit is smuggled through the inferred column *name* |
| `catalogue_epoch`, `pm_first`, `pm_second` | Required for the reduction-catalogue tie; Tycho-2 data is 1900–1950 so μ·Δt reaches ~5 arcsec |
| `first_uncertainty` / `second_uncertainty` | Per-observation sigmas exist in several sources (Voyager, Gaia, Showalter) and are currently dropped |

**Controlled vocabularies** as `StrEnum`s, so `verify` can reject `"NSDB"`/`"CCD"`/`"topocentre"` casing and spelling drift: `ObservationType`, `CoordinateKind`, `FrameCenter`, `ReferenceFrame`, `ReceptorType`, `ReductionCatalogue`, `DataSource`.

**`verify(dataframe, *, strict=True, allow_empty=True)`** collects *every* problem before raising, and checks: required columns present; declared dtypes; required columns non-null; enum membership; and angular ranges per `coordinate_kind`. The range check is deliberately loose on RA (±2π, accepting both wrap conventions) so its real job is catching **unit mistakes** — a value in degrees fails immediately with a message saying so.

**`build_frame(records)`** is the preferred constructor: it guarantees canonical order and dtypes, fills omitted columns with null, and rejects unknown keys so typos surface immediately. It exists because assigning rows into `empty_frame()` one at a time makes pandas coerce all-null object columns to `float64`; `_check_dtype` tolerates all-null columns for that reason.

**Tests:** `test/test_preprocessing_schema.py`, 57 tests. Full suite **568 passed** (511 baseline + 57 new), `ruff check` and `ruff format --check` clean.

### Remaining steps

4. `preprocessing/time_formatting.py` — split the ISO/JD/calendar *formatting* out of `set_iso_time_column`; add MJD. This is the remaining half of the time split.
5. `preprocessing/columns.py` — move `_find_column` / `_find_column_substring` / `_normalize_column_name` / `_numeric_series` / `apply_format_columns` from `nsdb_helpers` + the three duplicated copies.
6. `preprocessing/catalogues.py` — the reduction-catalogue tie and the proper-motion correction, neither of which exists today.
7. An `environment` helper to add ground stations on the fly (currently `add_observatory_to_SOB`, to be moved out of `observations/helpers.py`).
8. Rewire the factories to consume the standard frame, and decide whether `reference_frame: apparent` (48 configs) should start driving a real aberration correction.

Steps that were on this list and are now done: the `frames.py` router (step 5 below) and `aberration.py`.

### Step 4 — precession (done)

`preprocessing/precession.py` handles every equinox label in the real configs — **9 distinct labels, 0 unresolved**:

| Label | Resolves to |
|---|---|
| `J2000`, `J2000.0` | FK5 at J2000 |
| `J2015.0` | FK5 at 2015.0 |
| `date`, `Date`, `of date`, `MOD` | mean equinox of the observation date |
| `true of date`, `TOD` | true equinox of date (precession **and** nutation) |
| `apparent` | **not an equinox** — aberration kind, see step 5 |
| `B1950` | FK4, Besselian |
| `1899.0`, `1905.0` | **FK4, Besselian** (bare years) |
| `no information`, empty | J2000, with a warning |

The bare-year case matters: `1899.0` and `1905.0` are Besselian FK4 epochs in NSDB, and treating them as Julian would misplace them by ~20 arcsec. The regex for the Julian form therefore requires an explicit `J`, so a bare year can only match the Besselian branch. This was verified as a real bug during development — the first implementation had `j?` optional and silently classified both as FK5.

**All functions are pure numpy in, numpy out** — radians, no dataframes, no config. A dataframe may contain the call, but no conversion reaches into one. `precess_radec` handles the FK5↔FK5, FK4↔FK5, of-date and true-of-date families, requires `epoch_tdb` only when either equinox is of-date, and preserves NaN rows rather than aborting.

#### The FK4 convention problem

SPICE and Astropy **disagree about FK4/B1950**, and this was the most consequential finding of the step:

- SPICE's `B1950` is a fixed rotation matrix.
- Astropy's `FK4` applies position-dependent E-terms of aberration.

They differ by a **constant 0.525 arcsec rotation about z** (verified over the whole sky), reaching **~4 arcsec in RA** at the declinations in a random sample, because the E-term RA component carries a 1/cos(dec) factor. That is larger than every astrometric correction the pipeline currently applies, so it cannot be left implicit.

`Fk4Convention` makes it an explicit choice:

| Value | Behaviour |
|---|---|
| `spice` (**default**) | SPICE's fixed `B1950` matrix. What the existing Voyager and NSDB reductions were fitted against. |
| `astropy` | Astropy `FK4` with position-dependent E-terms. |
| `astropy_no_eterms` | Astropy `FK4NoETerms`. |

The default preserves existing behaviour deliberately: silently switching would shift every B1950 observation by half an arcsecond, which is exactly the kind of change this refactor must not smuggle in.

Also documented: **FK4↔FK5 is not exactly invertible** (round trip is accurate to ~0.001 arcsec, not machine precision) because the E-terms and frame bias are one-way catalogue corrections, not a group. The pipeline only ever converts *towards* J2000, so this is a property, not a defect.

`rotation_matrix` raises for non-SPICE frames with a message naming the Astropy route, and `SPICE_FRAMES` records the verified set (`J2000`, `B1950`, `ECLIPJ2000`, `ECLIPB1950`) — `ICRF`, `DATE` and `J2015.0` are **not** SPICE frames.

**Tests:** 80 new in `test/test_preprocessing_precession.py`. Full suite **745 passed**, ruff clean.

### Step 3 — observatory reference data + timescales (done)

Two new modules, plus a decision that changes the design:

**`preprocessing/observatories.py`** — MPC `observatories.txt` loading and identifier handling, moved out of `observations/helpers.py`:
- `normalize_observatory_code` — zero-pads to 3 digits, but returns **negative codes unchanged** (`-31` stays `-31`; the old `zfill(3)` produced `-31`, which happened to work only because `get_set_identifier` special-cased it).
- `is_numeric_observatory_code` — makes the "or name if not available" rule explicit instead of implicit.
- `ObservatoryInfo` dataclass with `longitude_radians`, `latitude_radians` and `geodetic_position` (the `[alt, lat_rad, lon_rad]` order `add_ground_station` expects).
- Path handling via `set_observatories_path` / `configure_from_config(cfg)` + `lru_cache`, so the pure functions need no config plumbing but still work with `cfg.data_folder`.

**`preprocessing/time_scales.py`** — all 31 distinct timescale labels found in the real NSDB configs now resolve, **zero unresolved**. Covers WMT, Washington/Paris/Pulkovo/Strasbourg/Nice/Hamilton/McCormick mean time, US Eastern/Central/Mountain/Pacific standard time, "90 degree time", GMT, GMT=UTC, UTC, UT1, TT, TAI, TDB, TCB and TCG. Also handles the dotted (`G.M.T.`, `M. T.`), parenthetical (`P.S.T. (Pacific S. T.)`) and spelled-abbreviation (`Central Stand. Time`) forms.

The two historical corrections are both implemented:
- **Astronomical day**: `+12 h` for every epoch before 1925-01-01, applied to *all* timescales (it is a dating convention, not a timescale), and disableable via `allow_pre_1925_shift=False`.
- **Meridian offsets**: named meridians carry their longitude (`washington = -77.0667°` → 5h08m20s), and an explicit `station_longitude_radians` overrides the named meridian. Zone times get a fixed hour offset.

**The important discovery:** `TimeScaleConverter.convert_time` **cannot convert UT1 before 1961-01-01** — the underlying leap-second table has no entries that far back, so it raises `RuntimeError: Provided caledar date cannot properly give Delta AT`. Since Washington mean time, GMT and every zone time resolve to UT1, that would have failed for essentially all pre-1961 data. The workarounds were both wrong:
- **SPICE `DELTET`** clamps every epoch before 1972 to a constant `+41.184 s` — wrong by over 40 s at 1900.
- **ERFA/`erfa.dat`** returns `0.0` before ~1961 and `+32.0 s` at J2000 (true value 63.8 s).

So the module implements the standard identity $\mathrm{TT} = \mathrm{UT1} + \Delta T$ with the **Espenak & Meeus (2006)** polynomials for 1900–1986, then converts TT→TDB through TudatPy. Verified against published values (1900: model −2.79 vs known −2.70; 1950: +29.07 vs +29.10; 1960: +33.10 vs +33.20), continuous across all polynomial boundaries, and epochs after 1961 still go through the native TudatPy path untouched. Accuracy is a few tenths of a second at worst — far below the precision of the historical astrometry that needs it.

**Tests:** 97 new in `test/test_preprocessing_time_scales.py` (including a `TestDeltaT` class). Full suite **665 passed** (568 after step 2 + 97 new), ruff clean.

**Note for later:** the `+12 h` and meridian logic lives here but the *formatting* half (ISO/JD/calendar → raw epoch) is still step 4. Until that lands, `set_iso_time_column` in `observations/nsdb_helpers.py` remains the producer of the epochs this module consumes.

### Step 5 — `frames.py` router and `aberration.py` (done)

`orbitdet.transformations.py` is **deleted**. Its responsibility is now split across three modules, so that the mathematics and the routing can be tested independently:

| Module | Owns |
|---|---|
| `preprocessing/precession.py` | The *orientation* of the axes — which equinox the axes point at. |
| `preprocessing/aberration.py` | The *kind* of direction — astrometric vs apparent. |
| `preprocessing/frames.py` | Only the routing between them, plus one frame-name vocabulary. |

#### `aberration.py`

Implements the exact relativistic IAU SOFA form, not the classical small-angle approximation $\mathbf{u}' = \mathbf{u} + \boldsymbol\beta$ — the two differ by ~0.001 arcsec for Earth's orbital motion, which is above the precision the pipeline targets.

Verified against ERFA's independent `eraAb`. The comparison needed care: `erfa.ab(u, v, s, bm1)` takes `s` as the **Sun–observer distance in au** (it also applies the Solar gravitational deflection) and `bm1` as the reciprocal Lorentz factor. Pushing the radial term to zero with `s → ∞` and passing the true $\sqrt{1-\beta^2}$ makes the two agree to **1.1e-16** — machine precision. Passing the classical `s = 1` leaves a 1.4e-12 residual, which is the Solar deflection term, not an error.

The transformation is exactly invertible (round trip 2.2e-16), zero-velocity is an exact identity, and the maximum displacement over the sky equals $|\boldsymbol\beta|$ (20.837 arcsec for Earth's orbital speed).

#### `frames.py`

Same public signatures as the old module, so the five call sites changed by one import line each:

| Call site | Function |
|---|---|
| `data/gaia.py` | `convert_radec_frame` (`angle_unit="rad"`, `ra_wrap=False`) |
| `data/voyager_data.py` | `convert_cartesian_frame` |
| `observations/voyager.py` | `convert_radec_frame` |
| `observations/absolute_ccd_nsdb.py` | `convert_radec_frame` |
| `observations/relative_xy_nsdb.py` | `convert_radec_frame` ×2 |

**Equivalence with the old module was verified numerically** on 300 random sky points across a century of epochs, and is the precondition for trusting the swap:

| Conversion | Max disagreement |
|---|---|
| `J2000` ↔ `ECLIPJ2000` | **0.0** (bit-identical) |
| `J2000` ↔ `B1950` | ≤3.1e-10 arcsec |
| `date` ↔ `J2000` | ≤5.1e-10 arcsec |
| `date` ↔ `ECLIPJ2000`, `date` ↔ `B1950`, `B1950` ↔ `date` | ≤5.1e-10 arcsec |
| cartesian `J2000` ↔ `ECLIPJ2000`, `J2000` ↔ `B1950` | **0.0** |

##### Bugs in the old code that this step fixes

1. **`Date → ECLIPJ2000` was wrong by 6.4e5 arcsec (180°).** `ECLIPJ2000` is an ecliptic frame with no equinox meaning, so the mean-of-date precession could not express it. `frames.py` chains through J2000 whenever a frame has no equinox meaning.

2. **`Date → B1950` was wrong by 12.6 arcsec.** Same cause: the FK4 leg needs SPICE's fixed rotation under the `spice` convention, not Astropy's E-term FK4.

3. **`J2000 → ICRS` silently skipped the frame bias**, returning `J2000` unchanged. The old code only applied the ICRS correction for *input* frames, and logged a warning admitting it. Astropy independently confirms the correct displacement is 566.9 mas in RA / 21.9 mas in Dec — which is what the new code produces.

4. **`ra_wrap=False` was ignored for ICRS input.** The old ICRS branch always returned `[0, 360)` because it delegated the convention to Astropy. The new `_convert_icrs` restores the principal interval explicitly. Only `gaia.py` passes `ra_wrap=False`, so this is a real behavioural fix on the Gaia path.

5. **Besselian equinoxes crashed or were silently mis-rotated.** `1905.0` and `1899.0` reach the FK4 branch, which then tried to hand the literal string `"1905.0"` to SPICE — `SpiceUNKNOWNFRAME`. Only B1950 has a SPICE counterpart, so `_requires_j2000_hub` now checks that the Besselian epoch is *exactly* 1950.0 before routing through SPICE; every other Besselian epoch goes through Astropy FK4.

6. **Unknown labels were classified as real equinoxes.** `parse_equinox` falls back to J2000 for labels it cannot read, which meant `classify_frame("nonsense")` returned `equinox`. `Equinox` now carries a `recognised` flag, so classification reports `UNKNOWN` honestly and `convert_radec_frame` raises rather than converting silently.

7. **`apparent` was classified as a true-of-date equinox.** This is a real-data bug, not a hypothetical: `reference_frame: apparent` appears in **48** configs and `astrometric` in **33**. `parse_equinox` accepted `apparent` as an of-date label and would have applied nutation to those 48 datasets while silently ignoring the aberration the label actually describes. `classify_frame` now checks the aberration vocabulary first, so `apparent` resolves to `FrameKind.KIND` and an aberration-only conversion raises with a clear message pointing at the right function.

8. **One NaN epoch aborted an entire batch.** Astropy rejects non-finite times outright, and the of-date frame was built from the raw epoch array *before* the NaN mask was applied. The mask is now computed first and the frames built from the filtered epochs.

##### Performance

The old `df.apply(..., axis=1)` loop was the pipeline's preprocessing bottleneck. The new implementation still loops the scalar SPICE calls (`pxform` is scalar-only) but vectorises all trigonometry:

| Path | 15 000 rows |
|---|---|
| old `apply(axis=1)` | 347 ms |
| new batched | 29 ms |

**~12x faster, with bit-identical output** — verified as 0.0 arcsec disagreement on the fixed-frame paths.

#### What is still *not* corrected

`reference_frame` (`apparent` ×48, `astrometric` ×33) is now *classified* correctly, but **nothing consumes it yet** — `convert_radec_frame` raises if asked to convert between the two kinds, because that needs an observer velocity the frame name cannot supply. Wiring `apparent` into a real aberration correction is a separate change with a real numerical effect (up to 20.8 arcsec), so it is deliberately left visible rather than slipped in here.

**Tests:** 218 across the three conversion modules (`test_preprocessing_frames.py`, `test_preprocessing_aberration.py`, `test_preprocessing_precession.py`). Full suite **883 passed**, ruff check and format clean.
