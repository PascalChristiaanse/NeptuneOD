# Relative Position-Angle & Separation Observation Model — Interaction Diagram

This document describes the runtime interaction of the
`relative_position_angle_and_separation_*_nsdb` dataset factory
(`src/orbitdet/observations/relative_position_angle_nsdb.py`), from the point where the
collection builder dispatches on the dataset `type`, through parsing, time/angle
conversion, per-observatory observation-set construction, and finally aggregation into
a Tudat `ObservationCollection` plus observation-model settings.

## Sequence diagram

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Collection builder
    participant Disp as create_observation_dataset()
    participant Reg as registry.get_factory()
    participant Fac as create_relative_position_angle_nsdb_dataset()
    participant NSDB as nsdb_helpers
    participant Help as helpers
    participant SOB as SystemOfBodies
    participant Anc as ancillary_settings
    participant Tudat as tudatpy (obs / model_settings)

    Caller->>Disp: create_observation_dataset(cfg, dataset_cfg, system_of_bodies)
    Disp->>Reg: get_factory(dataset_cfg.type)
    Reg-->>Disp: factory callable
    Disp->>Fac: factory(cfg, dataset_cfg, system_of_bodies)

    Note over Fac: --- Parse & normalise the raw table ---
    Fac->>Fac: pd.read_csv(file, sep=r"\s+", header=None, comment="#")
    Fac->>Fac: _format_columns(dataframe, format_columns)
    Note right of Fac: rename columns by 1-based index position

    Fac->>NSDB: set_iso_time_column(dataframe)
    Note over NSDB: JD  -> ISO-8601  (pd.to_datetime origin="julian")<br/>or Y/M/D[+h/m/s] -> ISO-8601
    NSDB-->>Fac: "iso_time" column

    Fac->>NSDB: set_position_angle_separation_columns(dataframe)
    Note over NSDB: position angle: deg + arcmin/60 -> radians (np.deg2rad)<br/>separation: arcsec -> radians (np.deg2rad(sep/3600))<br/>NSDB sentinel values masked (deg 998, arcmin 98, sep -98)
    NSDB-->>Fac: (POSITION_ANGLE_COLUMN, SEPARATION_COLUMN) [radians]

    Fac->>Fac: _observatories(dataset_cfg)
    Fac->>NSDB: resolve_observatory_codes(dataframe, observatories, telescope_index)
    Note over NSDB: single-observatory -> constant code (zero-padded)<br/>multi-observatory -> map selector column via telescope_index
    NSDB-->>Fac: Series of observatory codes

    Fac->>NSDB: group_rows_by_observatory(dataframe, codes)
    NSDB-->>Fac: [(station_name, sub_dataframe), ...]

    Note over Fac: --- Build one observation set per observatory ---
    loop each (station_name, group)
        Fac->>Fac: _build_set(cfg, dataset_cfg, SOB, group, station, angle_col, sep_col)

        Fac->>Help: add_observatory_to_SOB(cfg, SOB, station_name)
        Help->>SOB: get_ground_station(station_name)
        alt station missing
            Help->>Help: get_observatory_location(cfg, code)
            Help->>SOB: add_ground_station(Earth, station, [alt, lat, lon])
        end

        Fac->>Help: convert_time_to_seconds_since_j2000_TDB(group, station, SOB, time_scale)
        Note over Help: iso_time -> epoch (iso_string_to_epoch)<br/>timescale (ut1/utc/tt/tdb/GMT/CST/PST/local mean time)<br/>-> TDB, with station-longitude / zone offset<br/>writes "epoch_TDB"
        Help-->>Fac: epoch_TDB Series

        Fac->>Fac: target = first satellite, planet = dataset_cfg.planet.name
        Fac->>Fac: build link_ends (transmitter = planet origin,<br/>transmitter2 = target origin,<br/>receiver = station reference point, or Earth origin if geocentric)
        Fac->>Fac: _reference_frame(dataset_cfg)
        Note right of Fac: epoch_of_equinox -> PositionAngleReferenceFrame<br/>Date -> mean_of_date_iau_2006<br/>B1950 -> b1950, otherwise -> j2000

        Fac->>Anc: position_angle_ancillary_settings(reference_frame, use_stellar_aberration)
        Note right of Anc: use_stellar_aberration default True in factory<br/>True = apparent, False = astrometric
        Anc-->>Fac: ObservationAncillarySimulationSettings

        Fac->>Fac: link_definition = LinkDefinition(link_ends)

        opt position-angle rows present
            Fac->>Tudat: create_single_observation_set(position_angle_type, link_ends, values, epochs, receiver, ancillary)
            Fac->>Tudat: model_settings.position_angle(link_definition)
        end
        opt separation rows present
            Fac->>Tudat: create_single_observation_set(separation_distance_type, link_ends, values, epochs, receiver, ancillary)
            Fac->>Tudat: model_settings.separation_distance(link_definition)
        end
        Fac-->>Fac: accumulate (observation_sets, models)
    end

    Note over Fac: --- Aggregate ---
    alt exactly one observation set
        Fac-->>Disp: (SingleObservationSet, ObservationModelSettings)
    else multiple observation sets
        Fac->>Tudat: ObservationCollection(sets)
        Fac-->>Disp: (ObservationCollection, [ObservationModelSettings, ...])
    end
    Disp-->>Caller: (dataset, model_settings)
```

## Conversion responsibilities (at a glance)

| Concern | Where handled | Notes |
| --- | --- | --- |
| Time **format** (JD / Y-M-D / h-m-s → ISO-8601) | `nsdb_helpers.set_iso_time_column` | via `pd.to_datetime(..., origin="julian")` or component assembly |
| Time **standard / zone** (UT1, UTC, TT, TDB, GMT, CST, PST, local mean time → TDB seconds since J2000) | `helpers.convert_time_to_seconds_since_j2000_TDB` | uses `time_rep.default_time_scale_converter`; zone/local-mean offsets applied |
| Position-angle **units** (deg + arcmin → rad) | `nsdb_helpers.set_position_angle_separation_columns` | `np.deg2rad` |
| Separation **units** (arcsec → rad) | `nsdb_helpers.set_position_angle_separation_columns` | `np.deg2rad(sep / 3600)` |
| **Epoch of equinox** (pole convention) | `relative_position_angle_nsdb._reference_frame` | Date → IAU 2006 mean-of-date; B1950 → b1950; else → j2000 |
| **Apparent vs astrometric** (stellar aberration) | `ancillary.position_angle_ancillary_settings` | factory default `use_stellar_aberration=True` (apparent) |

## Notes and caveats

- **Equinox fallback is silent.** `_reference_frame` only recognises `Date`/`Of Date`
  and `B1950*`; every other value (`J2015.0`, `1899.0`, `no information`, ...) silently
  maps to `j2000` rather than warning or raising.
- **No coordinate rotation here.** Unlike the absolute RA/Dec factories (which call
  `convert_radec_frame` against `cfg.global_frame_orientation`), this factory only uses
  the equinox to choose the **north-pole convention** for the position angle; the
  observations are already angles, so no frame rotation of coordinates is performed.
- **Aberration is not data-driven yet.** `use_stellar_aberration` is not defined in any
  dataset YAML under `conf/datasets/**`, so it always defaults to `True`. The NSDB
  `reduction` field (e.g. "corrected for full relativistic aberration but not for
  relativistic light deflection") is parsed by `src/orbitdet/data/nsdb.py` but is not
  consumed by this factory.
- **Position angle vs separation split.** Rows with a valid angle and rows with a valid
  separation become *separate* `SingleObservationSet`s (and matching model settings);
  when only one component exists, a single set is returned instead of a collection.
