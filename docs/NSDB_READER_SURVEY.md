# NSDB reader: current functionality vs. the new parser

Survey of `src/orbitdet/data/nsdb.py` (`NSDBManager`, 596 lines) against the
unfinished prototype `src/orbitdet/data/nsdb_parsing.py` (`NSDBQuery`, 450 lines).

All numbers below come from running both implementations against the **83 real
metadata files in `data/raw/nm*/nm*.html`**, not from reading alone.

---

## 1. What the current reader does

`NSDBManager` is a *build-time* tool. It scrapes NSDB and writes Hydra configs;
it is not on the runtime observation path.

| Method | Responsibility |
| --- | --- |
| `download_dataset(name, dest)` | Fetches `{name}.txt` + `{name}.html` into `dest/{name}/`. **Refuses to overwrite** (`FileExistsError`) — the only reason `data/raw/` is trustworthy. |
| `_parse_contents_metadata(html, id)` | Splits into sections, parses each, merges, and fuses `type`. |
| `generate_hydra_configs(content, out, data_file)` | Writes `conf/datasets/generated/{identifier}.yaml`. |
| `_parse_contents(text)` | `planet`, `satellites`, `total number`, `type`, `dates`, `observatory`. |
| `_parse_informations(text)` | The astrometric metadata: `reference_frame`, `epoch_of_equinox`, `time_scale`, `reduction`, `coordinates`, `receptor`, `telescope`, `observers`, … |
| `_parse_telescope_entries(entries)` | Multi-telescope datasets; two formats (observatory-code prefix, or telescope-index with `code NNN` embedded). |
| `_attach_telescopes_to_observatories` | Matches telescopes to observatories by code. |
| `_build_telescope_index(format_columns, telescopes)` | Maps the telescope-index **data column** → observatory code. |

### Outputs the rest of the project depends on

- **`type` is fused**, not raw: `"absolute"` + `"absolute"` + `"CCD"` →
  `"absolute_absolute_CCD_nsdb"`. This string is the **factory registry key**
  (`@register_dataset_factory("absolute_absolute_CCD_nsdb")`) and appears
  verbatim in all 82 generated configs. It is a load-bearing contract.
- `format_columns` — 1-based index → description, and the key the factories
  actually use to name dataframe columns.
- `observatory` — list of `{code, name, telescopes[]}`.
- `telescope_index` — only **2/83** files actually produce one (`nm0019`,
  `nm0083`), the only two genuine multi-observatory datasets.
- Both spellings `centre_of_frame` **and** `center_of_frame` (deliberate).

### Two things it does not get right

1. **`reference_frame` is parsed but read by nothing.** No aberration is applied.
2. It parses metadata only. `_process_data` is a `pass` stub — the "TODO:
   Preprocess the dataframe for units, time, frame" is exactly what
   `preprocessing/` now covers.

---

## 2. Structural overlap with the new prototype

The prototype answers a **different question**: it is a *runtime* query client
(fetch → parse → preprocess → Tudat objects), where the current one is a
*build-time* config generator.

| Capability | `NSDBManager` (current) | `NSDBQuery` (prototype) |
| --- | --- | --- |
| Download / scrape | file-based, no-overwrite | `requests.get()` in-method, **no caching, no 404 retry** |
| HTML → sections | regex over all lines, tolerant | `bs4` + **exact-match list**, brittle |
| Generic `key: value` | inline, per-section | **pyparsing grammar — the real improvement** |
| Format section | regex `^(\d+)\.\s*(.+)$` | pyparsing `scan_string` |
| Domain parsing | `planet`/`satellites`/`dates`/`observatory` structured | **none** — keeps raw strings |
| `type` fusion | yes (`*_nsdb`) | **absent** |
| Multi-observatory | `telescope_index` + attach logic | **absent** |
| Hydra output | yes | absent |
| Tudat objects | no | stubs |
| Dataframe build | factories do it | `_get_observations` (buggy) |
| Tests | `test/test_nsdb_parse.py` (272 lines, passing) | **cannot run** — import path dead |

### The prototype's genuine advance

`_parse_contents_section` replaces ~140 lines of per-field `if lkey == …` with
one grammar. The key design choice is that the key regex is `[\w .]+`, which
**excludes `-`**. That matters: it means continuation lines such as
`188 - Z6: The 60-cm Zeiss Reflector` cannot be promoted to keys, because `188 -`
fails the key match. This is what makes `nm0083` work, and I verified it:

```text
telescope: ['083 - DL: The Tepfer Double Long-Focus Astrograph (D=40cm, F=550cm)',
            '188 - Z6: The 60-cm Zeiss Reflector (D=60cm, Cassegrain F=750cm)']
```

It also returns multi-line values as **lists** rather than joining them, which
fixes the current reader's whitespace-collapsing habit.

---

## 3. Bugs found in the current reader (measured, not guessed)

### 3.1 `References.` (plural) is not recognised — **49 of 83 files**

The section splitter's regex is `^(contents|informations|reference|comments|format)\.?$`.
NSDB uses the **plural** `References.` in 49 files, which does not match. The
heading is therefore treated as body text and appended to the previous section.

Consequence: **`reference` is never populated** in any of those 49 files, and
the `observatory` field in the Contents section — whose continuation-absorbing
loop runs until it sees a `:` — swallows the entire bibliography. In `nm0055`:

```text
observatory[1] = {'code': None,  'name': 'References.'}
observatory[2] = {'code': 1874,  'name': ''}
observatory[6] = {'code': 18743549, 'name': ''}   # a year and a page number, fused
```

Across all 83 files this produces **395 bogus entries at index ≥ 1**.

Heading spelling census:

| Spelling | Files | Matches current regex? | In prototype's list? |
| --- | --- | --- | --- |
| `References.` | **49** | ❌ | ❌ |
| `Reference.` | 26 | ✅ | ✅ |
| `Reference` (no period) | 5 | ✅ | ❌ |
| none | 3 | — | — |

### 3.2 Blast radius is smaller than it looks — `observatory[0]` is always clean

This is the important mitigating finding. The continuation loop starts at `i+1`
and `observatory[0]` comes from the `observatory:` line itself, so index 0 is
**never** polluted:

```text
observatory[0] plausible MPC code : 82
observatory[0] implausible        :  1
bogus entries at index >= 1       : 395
```

And the consumers only ever read index 0:

- `absolute_ccd_nsdb.py` → `dataset_cfg.observatory[0].code`
- `resolve_observatory_codes` → `observatories[0]["code"]` when there is no
  telescope index (81/83 files)

Only the 2 files that carry a `telescope_index` read past index 0, and **neither
uses `References.`** (nm0019 uses `Reference.`, nm0083 has no reference section).
So the corruption is real but currently latent — it will bite as soon as
anything iterates the observatory list.

### 3.3 `format_columns` count ≠ data column count — 8–9 of 83 files

| File | format entries | data columns |
| --- | --- | --- |
| `nm0001`, `nm0005` | 10 | 9 |
| `nm0011` | 9 | 8 |
| `nm0034`, `nm0035` | 7 | 8 |

Cause is real and physics, not parsing: NSDB writes **fixed-width, space-padded**
fields, and blank fields collapse under `sep=r"\s+"`. `nm0001` declares field 7
"Second of right ascension" but the row `44 10.474` has no seconds, so
declination slides left.

The current factory code survives because it **overlays** names onto existing
positional columns and bounds-checks (`if 0 <= pos < len(col_names)`). The
prototype would **raise**:

```python
df.columns = metadata[dataset]["format"].values()
# ValueError: Length mismatch: Expected axis has 9 elements, new values have 10
```

→ **9/83 datasets (11%) would crash.** This is the single most likely thing to
break first if the prototype is wired in as-is.

### 3.4 Small fallback artefacts

- `nm0036` has **no `Informations.` section** at all (it is a `References.` file
  with `Comments.` where Informations usually sits). Its refraction field lands
  under the raw key `diff._refraction` — 3 such keys across all files.
- Junk keys from bibliography text: `berkeley`, `moskva`,
  `lick_observatory_bulletin_;_no._39;_berkeley`, `no._404.,_berkeley`,
  `equatorial_observations_1893-1907` — **8 files**.
- `nm0011` data rows have variable field counts, so even a correct
  format-vs-data count is not sufficient; position within the line matters.

---

## 4. Bugs found in the prototype

The prototype is unfinished, and it shows in ways that are all fixable but none
of which are cosmetic:

| # | Issue | Evidence |
| --- | --- | --- |
| 1 | **Test file cannot run.** It loads `src/tudatpy/data/nsdb/nsdb_data.py`; that path resolves to `src/src/tudatpy/...` and `src/tudatpy/` does not exist. | `os.path.exists` → `False` |
| 2 | **`_parse_format` is greedy over trailing sample rows.** `scan_string` matches the real data row `1903  2 16  8 45 43  272.99   -99.99` as item number **1903**, then bleeds `272.99` in as a continuation. | 5/83 files produce non-sequential indices: `nm0055 → col 197`, `nm0076 → col 272`, `nm0037 → col 65`, `nm0044 → col 35`, `nm0051 → col 15` |
| 3 | **`_parse_html` hard-fails on missing sections**, and `nm0036` has no `Informations.` | `ValueError: Informations section is empty.` — 1/83 files |
| 4 | **`_parse_html` silently drops those files' data.** `nm0084`, `nm0090` use bare `Reference` (no period). It reports `OK` but yields no reference section — a silent wrong answer, worse than a crash. | 5/83 files |
| 5 | **No `type` fusion** → no factory registry key, and no way to write a config the pipeline can load. | 82 configs depend on the fused string |
| 6 | **No multi-observatory handling** (`planet`/`satellites`/`dates`/`observatory` left as raw strings) → cannot build relative-position observations. | `nm0019`, `nm0083` |
| 7 | **`_parse_contents_section` breaks at the first heading**, including headings it does not recognise, so it may stop early. | by inspection |
| 8 | **`requests.get()` with no timeout, no caching, no retry** — re-downloads 83 files per call. | `_get_observations` |
| 9 | **`pyparsing` is not in `environment.yml`** (installed as a transitive dep, 3.3.2). Relying on it needs an explicit pin. | `environment.yml` |
| 10 | ~15 stub methods (`table`, `size`, `to_tudat`, `__add__`, …) returning `None` — placeholder API surface. | by inspection |

**Update to #2:** this one is more important than it first appears. Because
`_parse_format` produces entries for **every** integer-leading line, the bogus
mapping is added to `format_columns`. But `format_columns` is exactly the key
the factories use to name columns — so bad indices there propagate straight into
observation data.

---

## 5. Verdict

**They are complementary, not competing.** The prototype is a better *lexer*;
the current reader is the *semantic layer* plus the build pipeline. The right
end state takes the prototype's grammar and keeps the current reader's domain
knowledge.

Concretely, the prototype's `_parse_contents_section` should replace the current
`_parse_contents`/`_parse_informations` line-splitting, and then feed the
existing `_parse_observatory_entry` / `_parse_telescope_entries` /
`_build_telescope_index` / `type`-fusion layer. Neither implementation alone is
shippable.

### Required fixes before the prototype can replace anything

1. **Fix the test import path.** Nothing else can be trusted until its 237 lines
   of tests actually execute.
2. **Recognise all four heading spellings**: `Reference.`, `References.`,
   `Reference`, `References.` — and make an unknown heading *not* terminate the
   current section, since `nm0036` legitimately has no `Informations.`.
3. **Bound `_parse_format`** to the region before the separator rule
   (`-----`). Every real file has one; stopping there removes the greedy
   sample-row match in one line and fixes all 5 bad files.
4. **Guard `df.columns = fmt.values()`** the way the current factories do —
   overlay positional names and bounds-check, rather than assigning a
   length-checked array. Without this **9/83 datasets raise**.
5. **Re-add `type` fusion, `planet`/`satellites`/`dates` structure, and the
   telescope-index logic** — or keep calling the current reader for them.
6. **Add `pyparsing` to `environment.yml`**, or drop it and keep the hand-rolled
   loop.
7. Decide the `requests` policy (timeout, retry, and whether to keep reading
   `data/raw/` rather than hitting the network from library code).

### The central-place question from your brief

The **generic** pieces worth extracting are the two grammar functions only:

- `_parse_contents_section` (generic `key: value` + indented continuations)
- `_parse_format` (generic `N. description`)

Everything else — `planets`, `satellites: N2-Nereid :174`, `dates: 1993-1998`,
`observatory: 083 - Name`, `total_number`, and the telescope/observatory
matching — is **NSDB-specific**, exactly as you suspected. That argues for a
`preprocessing/parsing.py` holding the two generic tokenisers, with all the
domain interpretation staying in an NSDB module.

I have not changed any code. Once you pick a direction I can implement it.

---

## 6. Implemented: the split into three classes

The prototype's grammar and the survey's domain logic are now merged into one
split, replacing `nsdb.py` (596 lines), `nsdb_parsing.py` (450) and
`nsdb_parsing_tests.py` (237, never runnable).

| Module | Class | Responsibility |
| --- | --- | --- |
| `data/nsdb_parser.py` | `NSDBParser`, `NSDBRecord` | Descriptor HTML in, a frozen record out. Pure: no network, no Tudat, no dataframe. |
| `data/nsdb_downloader.py` | `NSDBDownloader`, `NSDBDatasetFiles` | Fetch-once caching, modelled on `KernelManager`. |
| `data/nsdb_manager.py` | `NSDBManager` | YAML configs, collection config, Excel workbook. Delegates the other two. |

### Fixes applied

1. **Test file now runs.** The old one loaded `src/tudatpy/data/nsdb/nsdb_data.py`,
   which resolved to `src/src/tudatpy/…` and did not exist. The new tests import
   `orbitdet.data.nsdb_parser` normally: **181 tests** across
   `test/test_nsdb_parser.py` and `test/test_nsdb_download.py`, all passing.
2. **All four heading spellings recognised** — `Reference.`, `References.`,
   `Reference`, `References` — case-insensitively and with or without the
   period. Reference coverage went from **31/83 to 80/83**; the 3 remaining
   genuinely have no reference section.
3. **Blank lines terminate a field value.** This is what stops `nm0020`'s
   bibliography — which follows its observatory line with no heading — from
   being absorbed into the observatory list. Verified that no legitimate field
   value in the corpus spans a blank line.
4. **`_parse_format` stops at the separator rule.** Fixes the 5 datasets that
   produced bogus columns 15, 35, 65, 197 and 272. Every dataset now has exactly
   the indices `1..N`.
5. **Observatory codes bounded to three digits**, so bibliography such as
   `1874MNRAS__35___49D` cannot become an entry. Bogus entries at index ≥ 1 went
   from **395 to 0**.
6. **Three more observatory forms accepted**: `999 Bordeaux-Floirac` (no dash,
   `nm0018`), `H80 - Halsted observatory` (letter-prefixed MPC code, `nm0028`),
   and a bare `Cambridge` (`nm0051`, recorded as `code: None` rather than
   dropped). Unparsed lines went from **6 datasets to 0**.
7. **`telescope` entries require whitespace around the dash.** Without it,
   `26-inch refractor`, `40-inch` and `1-m reflectors` split into bogus entries.
   Structured telescopes now correctly appear in only the **2** multi-telescope
   datasets, down from 38.
8. **Discontinuous date ranges parsed separately.** `1898-1899, 1901-1902` no
   longer fuses into the year `18991901`; the span is the outer envelope.
9. **Missing `Informations.` heading recovered by vocabulary**, so `nm0036`'s
   frame, equinox and time scale are no longer lost.
10. **`df.columns = fmt.values()` length guard.** The parser never assigns a
    length-checked array; it exposes `format_columns` for the factories'
    existing bounds-checked overlay, so the 9 datasets whose declared column
    count differs from the whitespace-split count no longer raise.
11. **Downloads are atomic** — written to `.part` then moved — so an interrupted
    fetch cannot leave a file a later run trusts.
12. **`pyparsing` added to `environment.yml`** and the URLs no longer contain a
    double slash.

### One pre-existing bug this exposes

60 of the 82 committed configs in `conf/datasets/generated` store
``observatory`` as a **dict**, but the factories read ``observatory[0].code``
and so require a **list**. Those 60 configs cannot have been loadable. The new
parser always emits a list, in both the single- and multi-observatory cases.
Regenerating the configs will fix them.

### What is still not addressed

`_process_data` — time scale, precession, and the astrometric/apparent
correction — remains unimplemented in the reader. It is now correctly scoped,
because the pieces it needs exist in ``preprocessing``; wiring the frame and
aberration corrections into the NSDB path is the natural next step.
